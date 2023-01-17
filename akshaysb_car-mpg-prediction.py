import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/autompg-dataset/auto-mpg.csv')
df.head()
df['origin'].replace({1:'American',2:'European',3:'Japanese'},inplace=True)
df.info()
df.describe()
df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')
df['mpg'].plot(kind='kde')
df['cylinders'].plot(kind='kde')
df['displacement'].plot(kind='kde')
df['horsepower'].plot(kind='kde')
df['weight'].plot(kind='kde')
df['acceleration'].plot(kind='kde')
plt.figure(figsize=(8,8))
ax=sns.countplot(df['origin'])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))
plt.figure(figsize=(8,8))
ax=sns.barplot(x=df['origin'],y=df['weight'].median())
acc=(df.groupby('origin')['acceleration'].median())
print(acc)
acc.plot(kind='bar')
plt.ylabel('Avg Acceleration')
hp=(df.groupby('origin')['horsepower'].median())
print(hp)
hp.plot(kind='bar')
plt.ylabel('Avg. HP')
mpg=(df.groupby('origin')['mpg'].median())
print(mpg)
mpg.plot(kind='bar')
plt.ylabel('Avg Mpg')
plt.figure(figsize=(8,8))
ax=sns.countplot(df['model year'])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))
sns.scatterplot(x=df['weight'],y=df['mpg'])
sns.scatterplot(x=df['weight'],y=df['horsepower'])
sns.scatterplot(x=df['horsepower'],y=df['mpg'])
sns.scatterplot(x=df['acceleration'],y=df['horsepower'])
cor_mat=df.corr()
sns.heatmap(cor_mat,annot=True)
sns.pairplot(df,vars=['mpg','cylinders','displacement','horsepower','weight','acceleration'])
col=['origin','model year']
df=pd.get_dummies(data=df,drop_first=True,columns=col)
df.head()
df.drop('car name',axis=1,inplace=True)
imp=KNNImputer(missing_values=np.nan,n_neighbors=4)
df1=imp.fit_transform(df)
df=pd.DataFrame(df1,columns=df.columns)
df['horsepower'].unique()
# Base Model
x=df.drop('mpg',axis=1)
y=df['mpg']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=123)
x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()
vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T
x1=x.drop('horsepower',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=123)
x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()
vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T
x1=x1.drop('cylinders',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=123)
x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()
vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T
x1=x1.drop('displacement',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=123)
x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()
vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T
lr=LinearRegression()
model=lr.fit(x_train,y_train)
print(f'R^2 score for train: {lr.score(x_train, y_train)}')
print(f'R^2 score for test: {lr.score(x_test, y_test)}')
y_pred=lr.predict(x_test)
cv_results = cross_val_score(lr, x_train, y_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(GB_bias))
plt.plot(x_axis,GB_bias)
np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]
np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]
ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(ABLR_bias))
plt.plot(x_axis,ABLR_bias)
np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]
np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]
Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))
np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]
np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]
RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))
np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]
np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1234)
ss=StandardScaler()
x_s=ss.fit_transform(x)
x_trains=ss.fit_transform(x_train)
x_tests=ss.transform(x_test)
lr=LinearRegression()
model=lr.fit(x_trains,y_train)
print(f'R^2 score for train: {lr.score(x_trains, y_train)}')
print(f'R^2 score for test: {lr.score(x_tests, y_test)}')
cv_results = cross_val_score(lr, x_trains, y_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))
y_pred=lr.predict(x_tests)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_trains,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))
np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]
np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]
ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_trains,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))
np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]
np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]
Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))
np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]
np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]
RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_trains,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))
np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]
np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]
cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    x = x[cols]
    Xc = sm.add_constant(x)
    model = sm.OLS(y,Xc).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features = cols
print(selected_features)
X_new=x[selected_features]
X_new.head()
x_train1,x_test1,y_train1,y_test1=train_test_split(X_new,y,test_size=0.30,random_state=1234)
lr=LinearRegression()
model=lr.fit(x_train1,y_train1)
print(f'R^2 score for train: {lr.score(x_train1, y_train1)}')
print(f'R^2 score for test: {lr.score(x_test1, y_test1)}')
cv_results = cross_val_score(lr, x_train1, y_train1,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))
y_pred=lr.predict(x_test1)
mse=mean_squared_error(y_test1,y_pred)
rmse=np.sqrt(mse)
print(rmse)
GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))
np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]
np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]
ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))
np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]
np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]
Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))
np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]
np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]
RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))
np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]
np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]
x_news=ss.fit_transform(X_new)
x_train1s=ss.fit_transform(x_train1)
x_test1s=ss.transform(x_test1)
lr=LinearRegression()
model=lr.fit(x_train1s,y_train1)
print(f'R^2 score for train: {lr.score(x_train1s, y_train1)}')
print(f'R^2 score for test: {lr.score(x_test1s, y_test1)}')
cv_results = cross_val_score(lr, x_train1s, y_train1,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))
y_pred=lr.predict(x_test1s)
mse=mean_squared_error(y_test1,y_pred)
rmse=np.sqrt(mse)
print(rmse)
GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))
np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]
np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]
ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))
np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]
np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]
Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))
np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]
np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]
RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))
np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]
np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]
Rd=Ridge(alpha=0.5,normalize=True)
Ls=Lasso(alpha=0.1,normalize=True)
En=ElasticNet(alpha=0.01,l1_ratio=0.919,normalize=True)
models = []
models.append(('Ridge',Rd))
models.append(('Lasso',Ls))
models.append(('Elastic',En))
results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results = cross_val_score(model, x, y,cv=kfold, scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(np.sqrt(np.abs(cv_results))),np.std(np.sqrt(np.abs(cv_results)),ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results = cross_val_score(model, X_new, y,cv=kfold, scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(np.sqrt(np.abs(cv_results))),np.std(np.sqrt(np.abs(cv_results)),ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
df
x_final=df.drop('mpg',axis=1)
x_qr=x_final[['displacement','horsepower','weight','acceleration']]
qr=PolynomialFeatures(degree=2)
x_qr=qr.fit_transform(x_qr)
x_qr_df=pd.DataFrame(x_qr)
x_qr_df.head()
x_qr_df=x_qr_df.drop(0,axis=1)
idx=np.arange(x_final.shape[0])
y.index=idx
x_final.index=idx
x_qr_df=pd.concat([x_final,x_qr_df,y],axis=1)
x_qr_df.head()
x_qr_df.drop(['displacement','horsepower','weight','acceleration'],axis=1,inplace=True)
x_qr_df.columns
x_qr_df.columns=['cylinders', 'origin_European', 'origin_Japanese','model year_71',   'model year_72',   'model year_73',
                 'model year_74',   'model year_75',   'model year_76','model year_77',   'model year_78',   'model year_79',
                 'model year_80',   'model year_81',   'model year_82','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',
                 'f11','f12','f13','f14','mpg']
x_qr=x_qr_df.drop('mpg',axis=1)
y_qr=x_qr_df['mpg']
qr=LinearRegression()
models = []
models.append(('Ridge',Rd))
models.append(('Lasso',Ls))
models.append(('Elastic',En))
models.append(('Quadratic',qr))
results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results = cross_val_score(model, x_qr, y_qr,cv=kfold, scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(np.sqrt(np.abs(cv_results))),np.std(np.sqrt(np.abs(cv_results)),ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
xqr_train,xqr_test,yqr_train,yqr_test=train_test_split(x_qr,y_qr,test_size=0.30,random_state=1234)
cv_results = cross_val_score(qr, xqr_train, yqr_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))
y_pred=lr.predict(xqr_test)
mse=mean_squared_error(yqr_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))
np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]
np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]
ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))
np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]
np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]
Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))
np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]
np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]
RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))
np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]
np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]
xqr_s=ss.fit_transform(x_qr)
xqr_trains=ss.fit_transform(xqr_train)
xqr_tests=ss.transform(xqr_test)
cv_results = cross_val_score(qr, xqr_trains, yqr_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))
y_pred=qr.predict(xqr_tests)
mse=mean_squared_error(yqr_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)
GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))
np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]
np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]
ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))
np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]
np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]
Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))
np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]
np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]
RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))
np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]
np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]