import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
df=pd.read_csv('../input/car-data/car data.csv')
df.head()
df.shape
print(df['Seller_Type'].unique())

print(df['Fuel_Type'].unique())

print(df['Owner'].unique())

print(df['Transmission'].unique())
df.isnull().sum()
df.describe()
final_data=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_data.head()
final_data['Current_year']=2020
final_data.head()
final_data['no_year']=final_data['Current_year']-final_data['Year']
final_data.head()
final_data.drop(['Year','Current_year'],axis=1,inplace=True)
final_data.head()
final_data=pd.get_dummies(final_data,drop_first=True)
final_data.head()
sns.pairplot(final_data)
corrmat=final_data.corr()

top_corr_feat=corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(final_data[top_corr_feat].corr(),annot=True,cmap='RdYlGn')
X=final_data.iloc[:,1:]

y=final_data.iloc[:,0]
X.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor

model=ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor()
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]

max_features=['auto','sqrt']

max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=6)]

min_samples_splt=[2,5,10,15,100]

min_samples_leaf=[1,2,5,10]
from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_splt,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
rf_regr=RandomForestRegressor()
rf_rand=RandomizedSearchCV(estimator=rf_regr,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_rand.fit(X_train,y_train)
rf_rand.best_params_
pred=rf_rand.predict(X_test)
pred
sns.distplot(y_test-pred)
plt.scatter(y_test,pred)