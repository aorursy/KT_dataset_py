import pandas as pd

from datetime import datetime, timedelta

import pandas as pd

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, make_scorer, r2_score, accuracy_score,classification_report,confusion_matrix

from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.linear_model import LinearRegression
def graph_distance(regressor, x, y, title=""):

  predictions = regressor.predict(x)

  differences = y - predictions

  distance = np.linalg.norm(y-predictions)

  print('Euclidean Distance :',distance)

  print('Mean Squared :',mean_squared_error(predictions,y))

  plt.plot(differences)

  plt.title(title)

  plt.ylim((-30,30))

  plt.legend(['Difference'])

  plt.show()
def plot_predictions(predictions=None,actual=None,x_start=1380,x_stop=40,title=''):

  x_stop = x_start+x_stop

  plt.plot(actual[x_start:x_stop])

  plt.grid(True, which='both')

  plt.title(title)

  plt.plot(predictions[x_start:x_stop])

  plt.legend(['open','Prediction'])

  plt.show()
ticks_in_future=2

# load dataset

df = pd.read_csv("../input/tsla-dal-withfeatures/Apr29_p3_.TSLA.fs5.15824.ds.csv")

df['datetime'] = pd.to_datetime(df['datetime'])

df['open_next']=df['open'].shift(-ticks_in_future) # this is the 'future' value

df.dropna(inplace=True)

#print(df.isnull().sum())

print('shape: %s \nstart date: %s \nend date: %s'%(df.shape,df['datetime'].head(1)[0],df['datetime'].tail(1)))

plt.plot(df['open'].head(50))

plt.plot(df['open_next'].head(50))

plt.legend(['open','to_predict'])

plt.show()

df[['open','open_next']].head()
plt.figure(1 , figsize = (17 , 8))

cor = sns.heatmap(df.iloc[:,:25].corr(), annot = True)
y = df['open_next'].to_numpy()

# we dont use all the features, because it makes everything worse ! 

#X = df.drop(columns=['open_next','datetime']).to_numpy()

X = df[['open', 'high', 'low', 'close', 'volume']].to_numpy()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,shuffle=False)

print('xtrain size: %s xtest size: %s'%(x_train.shape[0],x_test.shape[0]))
lr = LinearRegression()

lr.fit(x_train,y_train)

lr.coef_
lr.score(x_test,y_test)
graph_distance(lr, x_test, y_test, title="Linear Regression")

predictions = lr.predict(x_test)

plot_predictions(predictions,y_test,x_start=1350, x_stop=50, title="Linear Regression")





regressors = [AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor]

for r in regressors:

  reg = r(random_state=0, n_estimators=100)

  params = reg.get_params()

  if 'base_estimator' in params:

    reg = r(lr,random_state=0, n_estimators=100)

  name=reg.__class__.__name__

  print(name)

  reg.fit(x_train,y_train)

  graph_distance(reg, x_test, y_test, title=name)

  predictions = reg.predict(x_test)

  plot_predictions(predictions,y_test, x_start=1350, x_stop=50, title=name)

  #plot_predictions(predictions,x_start, title=name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
lr=LinearRegression()

lr.fit(X_train, y_train)

ypl=lr.predict(X_val)

print('valMSE for LinearRegression:',mean_squared_error(y_val, ypl))

ypt=lr.predict(X_test)

print('TestMSE for LinearRegression:',mean_squared_error(y_test, ypt)) # To check if the model is overfitting
regressors = [AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor]

for r in regressors:

  reg = r(random_state=0, n_estimators=100)

  params = reg.get_params()

  if 'base_estimator' in params:

    reg = r(lr,random_state=0, n_estimators=100)

  name=reg.__class__.__name__

  #print(name)

  reg.fit(X_train,y_train)

  ypv=reg.predict(X_val)

  print(name,':The MSE error for validation:',mean_squared_error(y_val, ypv))

  ypt=reg.predict(X_test)  

  print(name,':The MSE error for testing:',mean_squared_error(y_test, ypt))