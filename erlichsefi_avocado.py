import numpy as np
import pandas as pd
raw_data=pd.read_csv('../input/avocado.csv')
raw_data.head()
# parse date 
raw_data.Date=pd.to_datetime(raw_data.Date,format='%Y-%m-%d')
# drop the 'index columns'
raw_data.drop(columns=['Unnamed: 0'],inplace=True)
raw_data.head()
raw_data.isnull().sum()
raw_data.region.value_counts().plot(kind='pie',title='regions')
raw_data.type.value_counts().plot(kind='pie',title='type')
raw_data.year.value_counts().plot(kind='pie',title='type')
raw_data.groupby('year').Date.nunique().plot(kind='barh')
# let see the reationships between the numberic values
raw_data.drop(columns=['region','type']).corr(method='pearson')
# but what about the catagorial features.
raw_data.groupby('type').AveragePrice.mean().plot(kind='bar')
#clearly, orgranic avocados cost more. we will consider that as feature.
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(raw_data.type.drop_duplicates()) 
raw_data.type = label.transform(raw_data.type) 
raw_data.groupby('region').AveragePrice.mean().sort_values().plot(kind='bar')
# let see if we can explain it by the amount in the market
raw_data.groupby('region')['Total Volume'].sum().sort_values().plot(kind='bar')
# and so the region of the avocado is importent as well.
label = LabelEncoder()
label.fit(raw_data.region.drop_duplicates()) 
raw_data.region = label.transform(raw_data.region) 
# let see if the month is importent.
raw_data['month']=raw_data.Date.dt.month
raw_data.groupby('month')['AveragePrice'].sum().sort_values().plot(kind='bar')
# it seems that the start of the year the prices are higher, let compre it to the amount in the market
raw_data.groupby('month')['Total Volume'].sum().sort_values().plot(kind='bar')
x=raw_data.drop(columns=['Date','AveragePrice'])
y=raw_data['AveragePrice']
x.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
regr = RandomForestRegressor(max_depth=6, random_state=0,n_estimators=100)
regr = regr.fit(X_train, y_train)
regr.score(X_test,y_test)



from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6,8,10],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_ 
best_grid.score(X_test,y_test)
# let check we don't overfit.
best_grid.score(X_train, y_train)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# scale
minmax=MinMaxScaler()
scaled_x=minmax.fit_transform(x)
# reduce dimension to drawable. 
pca = PCA(n_components=2)
_x=pca.fit_transform(scaled_x)
principalDf = pd.DataFrame(data = _x, columns = ['principal component 1', 'principal component 2'])
principalDf['label']=y
p=principalDf.plot.scatter(x='principal component 1', y='principal component 2',c='label',colormap='viridis')

# let try to find the where are the yellow data points.
principalDf[principalDf.label>3].plot.scatter(x='principal component 1', y='principal component 2',c='label',colormap='viridis')
from sklearn.ensemble import AdaBoostRegressor
X_train, X_test, y_train, y_test = train_test_split(_x, y, test_size=0.3, random_state=42)
regr = AdaBoostRegressor()
regr=regr.fit(X_train, y_train)
regr.score(X_test,y_test)
# we will use the sacled sampled to faster the training 
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.3, random_state=42)
from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()

# Input - Layer
model.add(Dense(30, activation = "relu",kernel_initializer='normal', input_shape=(scaled_x.shape[1], )))
# Hidden - Layers
#model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(20, activation = "relu",kernel_initializer='normal'))
#model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(10, activation = "relu",kernel_initializer='normal'))
model.add(Dense(5, activation = "relu",kernel_initializer='normal'))

# Output- Layer
model.add(Dense(1,kernel_initializer='normal'))
model.summary()


#compiling the model
model.compile(
 optimizer = "adam",
 loss = "mean_squared_error",
 metrics = ['mae']
)

results = model.fit(
 X_train, y_train,
 epochs= 15,
 batch_size = 500,
 validation_data = (X_test,y_test)
)

true_prices=raw_data[raw_data.region==46].sort_values(by='Date').set_index('Date').AveragePrice
ml_prices=best_grid.predict(raw_data[raw_data.region==46].sort_values(by='Date').set_index('Date').drop(columns=['AveragePrice']))
dl_prices=model.predict(minmax.fit_transform(
raw_data[raw_data.region==46].sort_values(by='Date').set_index('Date').drop(columns=['AveragePrice'])))
import matplotlib.pyplot as plt
plt.figure(figsize=(25,25))
plt.plot(true_prices.values)
plt.plot(ml_prices)
plt.show()


# Blue is the true value , Orange is the ML , green is the DL
import matplotlib.pyplot as plt
plt.figure(figsize=(25,25))
plt.plot(true_prices.values)
plt.plot(dl_prices)
plt.show()
fake_raw_data=raw_data.copy()
fake_raw_data['buyed']=np.rint(np.random.rand(raw_data.shape[0],1))
fake_raw_data.head()

# preze the all layers but the last
for layer in model.layers[:-1]:
    layer.trainable = False
    
fake_x=fake_raw_data.drop(columns=['Date','AveragePrice','buyed'])
fake_y=fake_raw_data['buyed']
fake_minmax=MinMaxScaler()
fake_scaled_x=fake_minmax.fit_transform(fake_x)
fX_train, fX_test, fy_train, fy_test = train_test_split(fake_scaled_x, fake_y, test_size=0.3, random_state=42)

#compiling the model
model.compile(
 optimizer = "adam",
 loss = "mean_absolute_error",
 metrics = ['accuracy']
)

results = model.fit(
 fX_train, fy_train,
 epochs= 3,
 batch_size = 500,
 validation_data = (fX_test,fy_test)
)
