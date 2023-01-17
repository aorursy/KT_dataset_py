import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn-poster')
df=pd.read_csv('../input/insurance/insurance.csv')
df.sample(5)
df.info()
charges = df['charges'].groupby(df.region).mean().sort_values(ascending = True)
print(charges)

sns.barplot(x=charges, y=charges.head().index, palette='Blues')
plt.title("Average Health Cost for Different Region")
plt.show()
sns.boxplot(x=df['smoker'],y=df['charges'])
plt.title('Health Cost among Smoker and non Smoker')
plt.show()
sns.pairplot(data=df, hue='smoker')
sns.displot(df,x='charges',hue='smoker')
plt.show()

df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(df.region.drop_duplicates())
df.region = label.transform(df.region)
label.fit(df.sex.drop_duplicates())
df.sex = label.transform(df.sex)

label.fit(df.smoker.drop_duplicates())
df.smoker = label.transform(df.smoker)

df.head()

sns.heatmap(df.corr(),annot=True,cmap='Blues_r')
plt.title("Heatmap of Correlation")
plt.show()
'''df_copy=df.copy()
df['smoker']=df['smoker'].map({'yes':1,'no':0})
#df.drop('sex',axis=1,inplace=True)
df=pd.get_dummies(df,columns=['region','sex'])
df=df[df.columns[[0,1,2,3,5,6,7,8,9,10,4]]] #rearranging columns
df.head()'''
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X = df.drop('charges',axis=1)
y=df['charges']

X2=df.drop(['charges','sex','region'],axis=1) #dropping features with lowest impact.
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X
lin_reg1=LinearRegression()
lin_reg1.fit(X_train,y_train)

lin_reg2=LinearRegression()
lin_reg2.fit(X2_train,y2_train)

y_pred1=lin_reg1.predict(X_test)
y_pred2=lin_reg2.predict(X2_test)

df2=pd.DataFrame({'Actual':y_test,'Predicted':y_pred1})
sns.scatterplot(x=y_test,y=y_pred1)
plt.title("Predicted vs Actual cost as per Linear Model")
plt.show()
print("Metrics for Model 1")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print("R2 score: ",metrics.r2_score(y_test, y_pred1))
print("\n")
print("Metrics for Model 2")
print('Mean Absolute Error:', metrics.mean_absolute_error(y2_test, y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(y2_test, y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2_test, y_pred2)))
print("R2 score: ",metrics.r2_score(y2_test, y_pred2))
from sklearn.preprocessing import PolynomialFeatures
lin_reg=LinearRegression()
for i in [2,3,4,5]:
    poly_reg=PolynomialFeatures(degree=i)

    X_poly=poly_reg.fit_transform(X_train)

    lin_reg.fit(X_poly,y_train)
    y_pred_poly=lin_reg.predict(poly_reg.fit_transform(X_test))

    print("Degree: ",i)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_poly))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_poly))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_poly)))
    print("R2 score: ",metrics.r2_score(y_test, y_pred_poly))
    print("\n")
    
#Best degree 2
poly_reg=PolynomialFeatures(degree=2)

X_poly=poly_reg.fit_transform(X_train)

lin_reg.fit(X_poly,y_train)
y_pred_poly=lin_reg.predict(poly_reg.fit_transform(X_test))
y_pred_poly=lin_reg.predict(poly_reg.fit_transform(X_test))
df3=pd.DataFrame({'Actual':y_test,'Predicted':y_pred_poly})

sns.scatterplot(x=y_test,y=y_pred_poly)
plt.title("Predicted vs Actual cost as per Polynomial Regression")
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_poly))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_poly))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_poly)))
print("R2 score: ",metrics.r2_score(y_test, y_pred_poly))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(
    n_estimators = 1100,
    max_depth = 4,
    random_state = 1,
    max_leaf_nodes=1000
)

rfr.fit(X_train, y_train)

y_pred_rf=rfr.predict(X_test)
df3=pd.DataFrame({'Actual':y_test,'Predicted':y_pred_rf})

sns.scatterplot(x=y_test,y=y_pred_rf)
plt.title("Predicted vs Actual cost as per RandomForest Regressor")
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print("R2 score: ",metrics.r2_score(y_test, y_pred_rf))
rfr2 = RandomForestRegressor(
    n_estimators = 700,
    max_depth = 4,
    random_state = 1,
    max_leaf_nodes=1000
)

rfr2.fit(X2_train, y2_train)

y_pred2=rfr2.predict(X2_test)

print("Metrics for Model 2")
print('Mean Absolute Error:', metrics.mean_absolute_error(y2_test, y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(y2_test, y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2_test, y_pred2)))
print("R2 score: ",metrics.r2_score(y2_test, y_pred2))
y_pred_rf_train=rfr.predict(X_train)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred_rf_train))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred_rf_train))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_rf_train)))
print("R2 score: ",metrics.r2_score(y_train, y_pred_rf_train))

sns.scatterplot(x=y_train,y=y_pred_rf_train)
plt.title("Predicted vs Actual cost as per RandomForest Regressor on Training Data")

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

alphas = [0.1000001, 0.0001, 0.0001, 0.001, 0.01, 0.1,0.5, 0.0000002]

for a in alphas:
 ridge_reg = Ridge(alpha=a, normalize=True,fit_intercept=True,max_iter=1000).fit(X_train,y_train) 
 score = ridge_reg.score(X_test, y_test)
 pred_y = ridge_reg.predict(X_test)
 mse = metrics.mean_squared_error(y_test, pred_y) 
 print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(a, score, mse, np.sqrt(mse)))
y_pred_rd=ridge_reg.predict(X_test)
df3=pd.DataFrame({'Actual':y_test,'Predicted':y_pred_rd})

sns.scatterplot(x=y_test,y=y_pred_rd)
plt.title("Predicted vs Actual cost as per Ridge Regression")
plt.show()
from sklearn.linear_model import Lasso

alphas = [0.1000001, 0.0001, 0.0001, 0.001, 0.01, 0.1,0.5, 0.0000002]

for a in alphas:
    lasso_reg = Lasso(alpha=0.2, fit_intercept=True, normalize=True, precompute=True, max_iter=10000,
                  tol=0.0001, warm_start=False, positive=False, random_state=1, selection='cyclic'
                ).fit(X_train, y_train)
    y_pred_ls=lasso_reg.predict(X_test)
    mse = metrics.mean_squared_error(y_test, pred_y)
    score = lasso_reg.score(X_test, y_test)
    print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
        .format(a, score, mse, np.sqrt(mse)))
y_pred_ls=lasso_reg.predict(X_test)
df3=pd.DataFrame({'Actual':y_test,'Predicted':y_pred_ls})

sns.scatterplot(x=y_test,y=y_pred_ls)
plt.title("Predicted vs Actual cost as per Laso Regression")
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_ls))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_ls))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_ls)))
print("R2 score: ",metrics.r2_score(y_test, y_pred_ls))
y_pred_rf_train=rfr.predict(X)
df_all=pd.DataFrame({'Actual':y_test,'Predicted':y_pred_rf})
df_all.head(10)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor