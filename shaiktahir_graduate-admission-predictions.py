import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

import warnings 

warnings.filterwarnings('ignore')
import io 

df=pd.read_csv('../input/graduate-admissions/datasets_14872_228180_Admission_Predict.csv',index_col=['Serial No.'])

df.head()
print('The dataset has rows:{} and columns:{}'.format(df.shape[0],df.shape[1]))
df.info()
df.describe()
pd.DataFrame({'%Missing':df.isnull().sum()})
for i in df.columns:   # Skewness for each columns seems to be normal

  print(i,':',df[i].skew())
df.hist(figsize=(25,8))

plt.plot()
for i in df.columns:   # Unique elements in each column

  print(i,'\n:',df[i].unique())

  print('*'*100)
for i in df.describe(exclude='O').columns:   # No Heavy Outliers

    plt.subplots()

    plt.title(i)

    sns.boxplot(df[i])

    plt.show()
sns.pairplot(df)
df.corr()['Chance of Admit '].sort_values(ascending=False)
plt.figure(figsize=(25,8))

sns.heatmap(df.corr(),annot=True,fmt='0.2f',cmap='autumn')

plt.plot()
corr=df.corr()

cols=corr.nlargest(5,'Chance of Admit ').index

cm=np.corrcoef(df[cols].values.T)

plt.figure(figsize=(30,10))

sns.heatmap(cm,annot=True,fmt='0.2f',xticklabels=cols.values,yticklabels=cols.values)
from scipy.stats import zscore   # Scaling the data using zscore



X=df.drop(['Chance of Admit '],axis=1)

X=X.apply(zscore)

y=df['Chance of Admit ']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)



print('R2 for train:',lin_reg.score(X_train,y_train))

print('R2 for test:',lin_reg.score(X_test,y_test))
import statsmodels.api as sm



X_constant=sm.add_constant(X)

model=sm.OLS(y,X_constant).fit()

predictions=model.predict(X_constant)

model.summary()
df['Predictions']=model.predict(X_constant)

residuals=model.resid

residuals
plt.figure(figsize=(20,10))

ax=sns.residplot(df.Predictions,residuals,lowess=True,color='r')

ax.set(xlabel='Fitted Value',ylabel='Residuals',title='Residual VS Fitted Plot')

plt.show()
plt.figure(figsize=(20,10))

sns.distplot(residuals)  # Beacuse the residuals are normal we do not require any transformations
# Different Evaluation Metircs for Train and Test 

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error



print('R^2 for Train:',r2_score(y_train,lin_reg.predict(X_train))*100) # Same as above

print('R^2 for Test:',r2_score(y_test,lin_reg.predict(X_test))*100)

print('*'*50)



print('RMSE for Train:',np.sqrt(mean_squared_error(y_train,lin_reg.predict(X_train))))

print('RMSE for Test:',np.sqrt(mean_squared_error(y_test,lin_reg.predict(X_test))))

print('*'*50)





print('MAE for Train:',mean_absolute_error(y_train,lin_reg.predict(X_train))*100)

print('MAE for Test:',mean_absolute_error(y_test,lin_reg.predict(X_test))*100)

print('*'*50)



print('MAPE for Train:',np.mean(abs(y_train-lin_reg.predict(X_train)/y_train))*100)

print('MAPE for Test:',np.mean(abs(y_test-lin_reg.predict(X_test)/y_test))*100)

print('*'*50)



print('MPE for Train:',np.mean(y_train-lin_reg.predict(X_train)/y_train)*100)

print('MPE for Test:',np.mean(y_test-lin_reg.predict(X_test)/y_test)*100)
df1=df.copy(deep=True)

df1.drop(['Predictions'],axis=1,inplace=True)

df1.head()
cols=list(X.columns)

pmax=1



while (len(cols)>0):

    p=[]

    X_1=X[cols]

    X_1=sm.add_constant(X_1)

    model=sm.OLS(y,X_1).fit()

    p=pd.Series(model.pvalues.values[1:],index=cols)

    pmax=max(p)

    feature_with_p_max=p.idxmax()

    if (pmax > 0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE=cols

print(selected_features_BE)
X=df1.drop(['Chance of Admit ','University Rating','SOP'],axis=1)

X=X.apply(zscore)

y=df1['Chance of Admit ']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X.shape)

print(y.shape)
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)



print('R2 for train:',lin_reg.score(X_train,y_train))

print('R2 for test:',lin_reg.score(X_test,y_test))
# Different Evaluation Metircs for Train and Test 

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error



print('R^2 for Train:',r2_score(y_train,lin_reg.predict(X_train))*100) # Same as above

print('R^2 for Test:',r2_score(y_test,lin_reg.predict(X_test))*100)

print('*'*50)



print('RMSE for Train:',np.sqrt(mean_squared_error(y_train,lin_reg.predict(X_train))))

print('RMSE for Test:',np.sqrt(mean_squared_error(y_test,lin_reg.predict(X_test))))

print('*'*50)





print('MAE for Train:',mean_absolute_error(y_train,lin_reg.predict(X_train))*100)

print('MAE for Test:',mean_absolute_error(y_test,lin_reg.predict(X_test))*100)

print('*'*50)



print('MAPE for Train:',np.mean(abs(y_train-lin_reg.predict(X_train)/y_train))*100)

print('MAPE for Test:',np.mean(abs(y_test-lin_reg.predict(X_test)/y_test))*100)

print('*'*50)



print('MPE for Train:',np.mean(y_train-lin_reg.predict(X_train)/y_train)*100)

print('MPE for Test:',np.mean(y_test-lin_reg.predict(X_test)/y_test)*100)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=[variance_inflation_factor(X_1.values,j)  for j in range(X_1.shape[1])]    # We can see multicollinearity is present b/w features

pd.DataFrame({'vif':vif[1:]},index=X.columns).T 
from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve,classification_report

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score,KFold



lin_reg=LinearRegression()

knn= KNeighborsRegressor()

dtc= DecisionTreeRegressor(ccp_alpha=0.01) # to increase pruning and avoid overfitting

rfr= RandomForestRegressor()

svm= SVR()
models = []

models.append(('Lin-Reg', LinearRegression()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('DecisionTree', DecisionTreeRegressor(random_state=123)))

models.append(('RandomForest', RandomForestRegressor()))

models.append(('SVM', SVR()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# compare algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison',fontsize=20)

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(20,8)

plt.show()