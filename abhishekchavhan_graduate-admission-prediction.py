import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,BaggingRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data.head(10)
data.info()
data.columns
fig ,axes = plt.subplots()
axes.bar(data['Serial No.'] ,data['Chance of Admit '],color='salmon');
X = data.drop('Chance of Admit ',axis=1)
y = data['Chance of Admit ']
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)
np.random.seed(40)
model = GradientBoostingRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)
np.random.seed(40)
model = ExtraTreesRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)
np.random.seed(40)
model = BaggingRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)
model_gre = GradientBoostingRegressor()
model_gre.fit(X_train,y_train)
model_gre.score(X_test,y_test)
model_gre.get_params()
grid = {'max_depth':[1,2,3],
        'min_samples_leaf':[0.5,1],
        'max_features':['auto','sqrt'],
        'min_samples_leaf':[1,2,3],
        'min_samples_split':[1,2]}
model_gre_gs = GridSearchCV(estimator=model_gre,param_grid=grid,verbose=2,n_jobs=2)
model_gre_gs.fit(X_train,y_train)
model_gre_gs.score(X_test,y_test)
y_prediction = model_gre.predict(X)
y_prediction
Sub = pd.DataFrame()
Sub['Original'] = data['Chance of Admit ']
Sub['Predictions'] = y_prediction
Sub
Sub.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='salmon')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
y_prediction = model_gre.predict(X_test)
import sklearn.metrics as met
mse = met.mean_squared_error(y_test,y_prediction)
print('Mean Squared Error : ',mse)
r2 = met.r2_score(y_test,y_prediction)
print('R-square_score : ',r2)
rmse = np.sqrt(mse)
print('Root mean squared error : ',rmse)
sns.regplot(x = Sub['Original'], y = Sub['Predictions'],color='salmon');