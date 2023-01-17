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

        mean = df.groupby(col)['price'].mean()

        df[col] = df[col].astype('category')

        levels = mean.sort_values().index.tolist()

        df[col].cat.reorder_categories(levels,inplace=True)

        df[col].value_counts().plot()

    plt.xticks(rotation=45)

    plt.xlabel(col)

    plt.ylabel('Counts')

    plt.subplot(1,2,2)

    

    if df[col].dtype == 'int64' or col == 'price':

        mean = df.groupby(col)['price'].mean()

        std = df.groupby(col)['price'].std()

        mean.plot()

        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values, \

                        alpha=0.1)

    else:

        sns.boxplot(x = col,y='price',data=df)

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
df = pd.read_csv('/kaggle/input/housing-simple-regression/Housing.csv')
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.isnull().sum()
df.info()
df.describe()
df.corr()
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=False,cmap='viridis')

plt.show()
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=True,cmap='Reds')

plt.show()
df1 = df.copy()

df2 = df.copy()
df1.head()
for col in df1:

    plot_feature(df1,col)

    plt.show()
df2.head()
df2.columns
num = ['area','bedrooms','bathrooms','stories','parking']
cat = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
num
cat
plt.style.use('ggplot')

for col in num:

    plt.figure(figsize=(15,6))

    sns.barplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.barplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.boxplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.boxplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.violinplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.violinplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.jointplot(x=col,y=df2['price'],data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.jointplot(x=col,y=df2['price'],data=df2,kind='hex',color='g')

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.stripplot(x=col,y=df2['price'],data=df2,jitter=True,edgecolor='gray',size=10,palette='winter',orient='v')

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.factorplot(x=col,y='price',data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.residplot(x=col,y='price',data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.distplot(df2[col],color='red')

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    plt.plot(col,'price',data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    plt.bar(col,'price',data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    plt.bar(col,'price',data=df2,color='B')

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.lineplot(x=col,y='price',data=df2)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
import scipy.stats as st

for col in num:

    plt.figure(figsize=(18,9))

    st.probplot(df2[col],plot=plt)

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.barplot(x=col,y='price',data=df2)

    sns.pointplot(x=col,y='price',data=df2,color='Black')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.barplot(x=col,y='price',data=df2)

    sns.pointplot(x=col,y='price',data=df2,color='Black')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(15,6))

    sns.boxplot(x=col,y='price',data=df2)

    sns.pointplot(x=col,y='price',data=df2,color='Black')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.boxplot(x=col,y='price',data=df2)

    sns.pointplot(x=col,y='price',data=df2,color='Black')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.boxenplot(x=col,y='price',data=df2)

    sns.pointplot(x=col,y='price',data=df2,color='Black')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.boxplot(x=col,y='price',data=df2)

    #sns.pointplot(x=col,y='price',data=df2,color='Black')

    sns.stripplot(x=col,y='price',data=df2,jitter=True,edgecolor='gray')

    plt.tight_layout()

    plt.show()
for col in cat:

    plt.figure(figsize=(15,6))

    sns.boxenplot(x=col,y='price',data=df2)

    sns.pointplot(x=col,y='price',data=df2,color='Black')

    sns.stripplot(x=col,y='price',data=df2,jitter=True,edgecolor='gray')

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(18,9))

    plt.scatter(x=col,y='price',data=df2)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('price')

    plt.axhline(15,color='Black')

    plt.axvline(50,color='Black')

    plt.show()
for col in num:

    plt.figure(figsize=(18,9))

    sns.kdeplot(data=df2[col])

    plt.tight_layout()

    plt.show()
for col in num:

    plt.figure(figsize=(12,7))

    plt.plot(df2[col].value_counts())

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
plt.style.use('dark_background')

for col in num:

    plt.figure(figsize=(12,7))

    df2.plot(x=col,y='price')

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
plt.style.use('ggplot')

for col in cat:

    plt.figure(figsize=(12,7))

    df2.plot(x=col,y='price')

    plt.xlabel(col)

    plt.ylabel('price')

    plt.tight_layout()

    plt.show()
sns.pairplot(df2)

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.boxplot(df2.price)

plt.subplot(1,2,2)

sns.distplot(df2.price,bins=20)

plt.show()
q = df2.price.describe()

print(q)

IQR    = q['75%'] - q['25%']

Upper  = q['75%'] + 1.5 * IQR

Lower  = q['25%'] - 1.5 * IQR

print("the upper and lower outliers are {} and {}".format(Upper,Lower))
rows =2



cols = 2



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = df2[num].columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.distplot(df2[col[index]],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()

rows = 2

cols = 2



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = df2[num].columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.regplot(x=df2[col[index]],y=df2['price'],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()
df.head()
def binary_mapping(x):

    return x.map({'yes':1,'no':0})
map = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
map
df[map] = df[map].apply(binary_mapping)
df.head()
cat = ['furnishingstatus']
cat
df = pd.get_dummies(df,cat,drop_first=True)
df.head()
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

df_columns = df.columns

df = scalar.fit_transform(df)

df = pd.DataFrame(df)

df.columns = df_columns

df.head()
X = df.drop(['price'],axis=1)
y = df[['price']]
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

from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import RFE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
cross_val_score(LinearRegression(),X_train,y_train).mean()
cross_val_score(RandomForestRegressor(n_estimators=100),X_train,y_train).mean()
cross_val_score(SVR(),X_train,y_train).mean()
cross_val_score(GradientBoostingRegressor(),X_train,y_train).mean()
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
df.corr()
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
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

plt.barh(range(len(indices)), importances[indices], color='Black', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
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
svm = SVR()

svm.fit(X_train[a],y_train)

pred = svm.predict(X_test[a])

r2_score(y_test,pred)
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
X_train.shape
for index in range(1,14):

    fe = RFE(LinearRegression(), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_Linear(X_train_b, X_test_b, y_train, y_test)

    print()
for index in range(1,14):

    fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_randomForest(X_train_b, X_test_b, y_train, y_test)

    print()
for index in range(1,14):

    fe = RFE(GradientBoostingRegressor(random_state=100), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_Gradient(X_train_b, X_test_b, y_train, y_test)

    print()
fe = RFE(LinearRegression(), n_features_to_select = 12)

fe.fit(X_train, y_train)

X_train_b = fe.transform(X_train)

X_test_b = fe.transform(X_test)

print('Selected Feature: ', 12)

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
X_train_sm = sm.add_constant(X_train[a])

lr = sm.OLS(y_train,X_train_sm).fit()

lr.summary()
lr = LinearRegression()

rfe = RFE(lr, 8)             

rfe = rfe.fit(X_train, y_train)

print(rfe.support_)          

print(rfe.ranking_) 
b=X_train.columns[rfe.get_support()]

b
lr = LinearRegression()

lr.fit(X_train[b],y_train)

pred = lr.predict(X_test[b])

r2_score(y_test,pred)
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)





hyper_params = [{'n_features_to_select': list(range(1, 14))}]



lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)             



model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')

plt.show()
lr = LinearRegression()

rfe = RFE(lr, 10)           

rfe = rfe.fit(X_train, y_train)

print(rfe.support_)          

print(rfe.ranking_) 
a = X_train.columns[fe.get_support()]
a
lr = LinearRegression()

lr.fit(X_train[a],y_train)

pred = lr.predict(X_test[a])

r2_score(y_test,pred)
X_train_sm = sm.add_constant(X_train[a])

lr = sm.OLS(y_train,X_train_sm).fit()

lr.summary()
lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

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
X_train.drop('bedrooms',axis=1,inplace=True)

X_test.drop('bedrooms',axis=1,inplace=True)
lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

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
X_train.drop('furnishingstatus_semi-furnished',axis=1,inplace=True)

X_test.drop('furnishingstatus_semi-furnished',axis=1,inplace=True)
lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

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
for index in range(1,12):

    fe = RFE(LinearRegression(), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_Linear(X_train_b, X_test_b, y_train, y_test)

    print()
for index in range(1,12):

    fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_randomForest(X_train_b, X_test_b, y_train, y_test)

    print()
for index in range(1,12):

    fe = RFE(GradientBoostingRegressor(random_state=100), n_features_to_select = index)

    fe.fit(X_train, y_train)

    X_train_b = fe.transform(X_train)

    X_test_b = fe.transform(X_test)

    print('Selected Feature: ', index)

    run_Gradient(X_train_b, X_test_b, y_train, y_test)

    print()
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)





hyper_params = [{'n_features_to_select': list(range(1, 12))}]



lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)             



model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')

plt.show()
rows = 3

cols = 3



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = X_train.columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.regplot(x=df[col[index]],y=df['price'],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()
plt.style.use('default')

lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

r2_score(y_test,pred)

title = "Learning Curve (Linear Regression)"

cv = 5

plot_learning_curve(lr, title, X_train, 

                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1);
title = "Learning Curve (Random Forest)"

cv = 5

plot_learning_curve(rf, title, X_train, 

                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1);
title = "Learning Curve (Gradient Boosting)"

cv = 5

plot_learning_curve(gb, title, X_train, 

                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1);
title = "Learning Curve (SVM)"

cv = 5

plot_learning_curve(svm, title, X_train, 

                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1)
plt.figure(figsize=(10,5))

plt.style.use('default')

lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

r2_score(y_test,pred)

plt.scatter(y_test,pred)

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(y_test-pred)

plt.show()
plt.figure(figsize=(13,5))

c = [i for i in range(1,165,1)]

fig = plt.figure() 

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") 

plt.plot(c,pred, color="red",  linewidth=2.5, linestyle="-") 

fig.suptitle('Actual and Predicted', fontsize=20)               

plt.xlabel('Index', fontsize=18)                               

plt.ylabel('Price', fontsize=16)

plt.show()