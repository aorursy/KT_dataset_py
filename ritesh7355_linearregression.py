import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 
df=pd.read_csv('../input/air-quality-indexnew-delhi/Air Quality Index- Delhi.csv')
df.head()
## Check for null values



sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df=df.dropna()
X=df.iloc[:,:-1] # Independent values

y=df.iloc[:,-1] #Dependent Values
y.isnull().sum()
X.isnull().sum()
sns.pairplot(df)
df.corr()
corrmat=df.corr()

top_corr_features=corrmat.index

plt.figure(figsize=(20,20))

#plot Heatmap

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corrmat.index
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
X.head()
#plot graph of feature importances for better visualization

feat_importances=pd.Series(model.feature_importances_,index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
sns.distplot(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X,y)
regressor.coef_
regressor.intercept_
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))
from sklearn.model_selection import cross_val_score

score=cross_val_score(regressor,X,y,cv=5)
score.mean()
coeff_df=pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])

coeff_df
prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))

print('MSE:', metrics.mean_squared_error(y_test, prediction))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))