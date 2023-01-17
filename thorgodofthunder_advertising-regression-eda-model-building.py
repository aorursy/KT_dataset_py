import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def plot_feature(df,col):

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)

    if df[col].dtype == 'int64':

        df[col].value_counts().sort_index().plot()

    else:

        mean = df.groupby(col)['Sales'].mean()

        df[col] = df[col].astype('category')

        levels = mean.sort_values().index.tolist()

        df[col].cat.reorder_categories(levels,inplace=True)

        df[col].value_counts().plot()

    plt.xticks(rotation=45)

    plt.xlabel(col)

    plt.ylabel('Counts')

    plt.subplot(1,2,2)

    

    if df[col].dtype == 'int64' or col == 'Sales':

        mean = df.groupby(col)['Sales'].mean()

        std = df.groupby(col)['Sales'].std()

        mean.plot()

        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values, \

                        alpha=0.1)

    else:

        sns.boxplot(x = col,y='Sales',data=df)

    plt.xticks(rotation=45)

    plt.ylabel('Sales')

    plt.show()    
def missingdata(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms= ms[ms["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return ms
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Validation score")



    plt.legend(loc="best")

    return plt
df = pd.read_csv('/kaggle/input/Advertising.csv')
df.head()
df.info()
#plt.figure(figsize=(12,7))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.isnull().sum()
df.describe()
df.corr()
plt.figure(figsize=(12,5))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
plt.figure(figsize=(12,5))

sns.heatmap(df.corr(),annot=False,cmap='viridis')

plt.show()
plt.figure(figsize=(12,5))

sns.heatmap(df.corr(),annot=True,cmap='Reds')

plt.show()
# Data back up we use it for Eda

df1 =  df.copy()

df2  = df.copy()

df3  = df.copy()
df1.head()
plot_feature(df1,'TV')
plot_feature(df1,'Radio')
plot_feature(df1,'Newspaper')
plt.style.use('dark_background')

df1.plot(x='TV',y='Sales',color='white')

plt.show()
plt.style.use('dark_background')

df1.plot(x='Radio',y='Sales')

plt.show()
plt.style.use('dark_background')

df1.plot(x='Newspaper',y='Sales')

plt.show()
plt.style.use('ggplot')

df1['TV'].value_counts().plot()

plt.show()
plt.style.use('ggplot')

df1['Radio'].value_counts().plot()

plt.show()
plt.style.use('ggplot')

df1['Newspaper'].value_counts().plot()

plt.show()
df2['TV'] = df2['TV'].apply(int)

df2['Radio'] = df2['Radio'].apply(int)

df2['Newspaper'] = df2['Newspaper'].apply(int)

df2['Sales'] = df2['Sales'].apply(int)
plt.style.use('ggplot')

df2['TV'].value_counts().plot()

plt.show()
plt.style.use('ggplot')

df2['Radio'].value_counts().plot()

plt.show()
plt.style.use('ggplot')

df2['Newspaper'].value_counts().plot()

plt.show()
plt.style.use('ggplot')

df2['Sales'].value_counts().plot()

plt.show()
df2['TV'].mean()
def tv_avg(tv):

    if tv >= 146.57:

        return "Higher"

    else:

        return "Lower"
plt.figure(figsize=(15,8))

plt.style.use('ggplot')

df2['TV'].apply(tv_avg).value_counts().plot(kind = "pie",legend=True)

plt.show()
df2['Radio'].mean()
def radio_avg(radio):

    if tv >= 22.79:

        return "Higher"

    else:

        return "Lower"
plt.figure(figsize=(15,8))

plt.style.use('ggplot')

df2['Radio'].apply(tv_avg).value_counts().plot(kind = "pie",legend=True)

plt.show()
df2['Newspaper'].mean()
def radio_avg(radio):

    if tv >= 30.1:

        return "Higher"

    else:

        return "Lower"
plt.figure(figsize=(15,8))

plt.style.use('ggplot')

df2['Newspaper'].apply(tv_avg).value_counts().plot(kind = "pie",legend=True)

plt.show()
df2['TV'].value_counts().sum()
def tv_sales(tv):

    if tv <= 50:

        return "drop sales"

    elif tv > 100 and tv <=239:

        return "Sales imporved"

    else:

        "avg Sales"
plt.figure(figsize=(10,5))

plt.style.use('ggplot')

df2['TV'].apply(tv_sales).value_counts().plot(kind='bar')

plt.show()
plt.figure(figsize=(10,5))

df2['TV'].plot(kind='hist',bins=50)

plt.show()
plt.figure(figsize=(10,5))

df2['Radio'].plot(kind='hist',bins=50)

plt.show()
plt.figure(figsize=(10,5))

df2['Newspaper'].plot(kind='hist',bins=50)

plt.show()
plt.figure(figsize=(10,5))

df2['Sales'].plot(kind='hist',bins=50)

plt.show()
df3.head()
for col in df3.columns:

    plt.figure(figsize=(19,10))

    sns.barplot(x=col,y='Sales',data=df3)

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.jointplot(x=col,y='Sales',data=df3,kind='reg')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.jointplot(x=col,y='Sales',data=df3,kind='hex')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.jointplot(x=col,y='Sales',data=df3,kind='hex',space=0,color='g')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.stripplot(x=col,y='Sales',data=df3,jitter=True,edgecolor='gray',size=10,palette='winter',orient='v')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.swarmplot(x=col,y='Sales',data=df3)

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='Sales',data=df3)

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.residplot(x=col,y='Sales',data=df3,lowess=True)

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.distplot(df3[col],color='red')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    plt.plot(col,'Sales',data=df3,color='orange')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    plt.bar(col,'Sales',data=df3,color='Orange')

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('Sales')

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.lineplot(x=col,y='Sales',data=df3)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('Sales')

    plt.show()
import scipy.stats as st

for col in df3.columns:

    plt.figure(figsize=(18,9))

    st.probplot(df3[col],plot=plt)

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='Sales',data=df3)

    sns.pointplot(x=col,y='Sales',data=df3,color='Black')

    plt.tight_layout()

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.boxplot(data=df3)

    sns.stripplot(data=df3,jitter=True,edgecolor='gray')

    plt.tight_layout()

    plt.ylabel('Sales')

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    plt.scatter(x=col,y='Sales',data=df3)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('Sales')

    plt.axhline(15,color='Black')

    plt.axvline(50,color='Black')

    plt.show()
for col in df3.columns:

    plt.figure(figsize=(18,9))

    sns.kdeplot(data=df3)

    plt.tight_layout()

    plt.show()
sns.pairplot(df3)

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.boxplot(df3.Sales)

plt.subplot(1,2,2)

sns.distplot(df3.Sales,bins=20)

plt.show()
q = df3.Sales.describe()

print(q)

IQR    = q['75%'] - q['25%']

Upper  = q['75%'] + 1.5 * IQR

Lower  = q['25%'] - 1.5 * IQR

print("the upper and lower outliers are {} and {}".format(Upper,Lower))
rows =2



cols = 2



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = df3.columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.distplot(df3[col[index]],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()
rows = 2

cols = 2



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = df3.columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.regplot(x=df3[col[index]],y=df3['Sales'],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()

        
# back to orginal data

df.head()
def normalize (x): 

    return ( (x-np.min(x))/ (max(x) - min(x)))
df = df.apply(normalize)
X = df.drop(['Sales'],axis=1)
y = df[['Sales']]
X.head()
y.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn import metrics

from sklearn.ensemble import GradientBoostingRegressor

import time

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

from sklearn.model_selection import learning_curve

import statsmodels.api as sm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
cross_val_score(LinearRegression(),X_train,y_train).mean()
cross_val_score(RandomForestRegressor(n_estimators=100),X_train,y_train).mean()
cross_val_score(GradientBoostingRegressor(),X_train,y_train).mean()
cross_val_score(SVR(),X_train,y_train).mean()
lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

r2_score(y_test,pred)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train,y_train)

pred = rf.predict(X_test)

r2_score(y_test,pred)
gb = GradientBoostingRegressor()

gb.fit(X_train,y_train)

pred = gb.predict(X_test)

r2_score(y_test,pred)
svm = SVR()

svm.fit(X_train,y_train)

pred = svm.predict(X_test)

r2_score(y_test,pred)
X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train,X_train_sm).fit()

lr.summary()
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
rf.feature_importances_
feature_importance = pd.Series(rf.feature_importances_,index=X_train.columns)

feature_importance.sort_values()

feature_importance.plot(kind='barh',figsize=(8,6))

plt.show()
plt.figure(figsize=(7,8))

features = X_train.columns

importances = rf.feature_importances_

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='Blue', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
from sklearn.feature_selection import SelectFromModel
fe = SelectFromModel(RandomForestRegressor(n_estimators=100))
fe.fit(X_train,y_train)
fe.get_support()
a = X_train.columns[fe.get_support()]
a
lr = LinearRegression()

lr.fit(X_train[a],y_train)

pred = lr.predict(X_test[a])

r2_score(y_test,pred)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train[a],y_train)

pred = rf.predict(X_test[a])

r2_score(y_test,pred)
gb = GradientBoostingRegressor()

gb.fit(X_train[a],y_train)

pred = gb.predict(X_test[a])

r2_score(y_test,pred)
from sklearn.feature_selection import RFE
def run_Linear(X_train, X_test, y_train, y_test):

    clf = LinearRegression()

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print('R2 Score: ', r2_score(y_test, pred))

    print('MSE:',metrics.mean_squared_error(pred,y_test))

    rms = np.sqrt(metrics.mean_squared_error(pred, y_test))

    print('RMSE:',rms)
def run_randomForest(X_train, X_test, y_train, y_test):

    clf = RandomForestRegressor(n_estimators=100, random_state=100, n_jobs=-1)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print('R2 Score: ', r2_score(y_test, pred))

    print('MSE:',metrics.mean_squared_error(pred,y_test))

    rms = np.sqrt(metrics.mean_squared_error(pred, y_test))

    print('RMSE:',rms)
def run_Gradient(X_train, X_test, y_train, y_test):

    clf = GradientBoostingRegressor(n_estimators=100, random_state=100)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print('R2 Score: ', r2_score(y_test, pred))

    print('MSE:',metrics.mean_squared_error(pred,y_test))

    rms = np.sqrt(metrics.mean_squared_error(pred, y_test))

    print('RMSE:',rms)
for index in range(1,4):

    fe = RFE(LinearRegression(), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_Linear(X_train_b, X_test_b, y_train, y_test)

    print()
for index in range(1,4):

    fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_randomForest(X_train_b, X_test_b, y_train, y_test)

    print()
for index in range(1,4):

    fe = RFE(GradientBoostingRegressor(random_state=100), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_Gradient(X_train_b, X_test_b, y_train, y_test)

    print()
fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = 2)

fe.fit(X_train, y_train)

X_train_b = fe.transform(X_train)

X_test_b = fe.transform(X_test)

print('Selected Feature: ', 2)

run_randomForest(X_train_b, X_test_b, y_train, y_test)

print()
fe.get_support()
a = X_train.columns[fe.get_support()]
a
lr = LinearRegression()

lr.fit(X_train[a],y_train)

pred = lr.predict(X_test[a])

r2_score(y_test,pred)
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train[a],y_train)

pred = rf.predict(X_test[a])

r2_score(y_test,pred)
gb = GradientBoostingRegressor()

gb.fit(X_train[a],y_train)

pred = gb.predict(X_test[a])

r2_score(y_test,pred)
svm = SVR()

svm.fit(X_train[a],y_train)

pred = svm.predict(X_test[a])

r2_score(y_test,pred)
plt.style.use('default')
title = "Learning Curves (Linear Regression)"

cv = 5

plot_learning_curve(lr, title, X_train[a], 

                    y_train, ylim=(0.7, 1.0), cv=cv, n_jobs=-1);
title = "Learning Curves (Random Forest)"

cv = 5

plot_learning_curve(rf, title, X_train[a], 

                    y_train, ylim=(0.7, 1.0), cv=cv, n_jobs=-1);
title = "Learning Curves (Gradient Boosting)"

cv = 5

plot_learning_curve(gb, title, X_train[a], 

                    y_train, ylim=(0.7, 1.0), cv=cv, n_jobs=-1);
title = "Learning Curves (Gradient Boosting)"

cv = 5

plot_learning_curve(svm, title, X_train[a], 

                    y_train, ylim=(0.7, 1.0), cv=cv, n_jobs=-1);
plt.figure(figsize=(10,5))

plt.style.use('default')

lr = LinearRegression()

lr.fit(X_train[a],y_train)

pred = lr.predict(X_test[a])

r2_score(y_test,pred)

plt.scatter(y_test,pred)

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(y_test-pred)

plt.show()
plt.figure(figsize=(13,5))

c = [i for i in range(1,61,1)]

fig = plt.figure() 

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") 

plt.plot(c,pred, color="red",  linewidth=2.5, linestyle="-") 

fig.suptitle('Actual and Predicted', fontsize=20)               

plt.xlabel('Index', fontsize=18)                               

plt.ylabel('Price', fontsize=16)

plt.show()