### import all required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

df.shape
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
df.isnull().sum().sum()
df.describe()
df.columns
final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()
final_dataset['current_year']=2020
final_dataset.head()
final_dataset['no_year']=final_dataset['current_year']-final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.head()
final_dataset.drop(['current_year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.corr()
sns.pairplot(final_dataset)
corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='YlOrBr')
final_dataset.head()
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
X.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance
feat_importance.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)
X_train.shape
X_test.shape
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
# hyperparameter 

n_estimators=[int(x) for x in np.linspace(100,1200 ,12)]
print(n_estimators) 
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
from sklearn.model_selection import RandomizedSearchCV

## crate the random grid

random_grid={'n_estimators':n_estimators,
            'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf': min_samples_leaf}
print(random_grid)
rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error', n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)
prediction=rf_random.predict(X_test)
prediction
sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)
import pickle
file=open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random,file)
