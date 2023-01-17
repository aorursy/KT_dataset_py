# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

fffff=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
ty=test["LotFrontage"]
ty=ty.to_dict()
for i,j in ty.items():

    test["LotFrontage"].iloc[i]=j
train.head()
train.info()
a=[]

def missvalue(t):

    for i in t.columns:

        if(t[i].isnull().sum()>0):

              a.append(i)

            
missvalue(train)

a
a=[]

missvalue(test)

a
a=[]

missvalue(train)

num=[]

cat=[]

def separate(train,p):

    for i in p:

        if(train[i].dtypes=='O'):

            cat.append(i)

        else:

            num.append(i)

    
separate(train,a)
cat
num
num2=[]

num1=[]

def cor(k):

    corr=train.corr()

    for l in k:

        m=0

        t=corr[l].sort_values(ascending=False)

        t=t.to_dict()

        for i,j in t.items():

            if(j>0.5):

                print(i,j)

                m=m+1

        if(m>1):

            num1.append(l)

        else:

            num2.append(l)

                

           

        print("*"*40)
cor(num)
num1
num2
def fig(k,c):

     for i in k:

        sns.boxplot(i,data=train)

        plt.show()

        



fig(num2,train)
plt.scatter(train['GarageYrBlt'],train['YearBuilt'])
def discrete(a,b,c):

    t=b[a].isnull()

    t=t.to_dict()

    k=[]

    for i,j in t.items():

        if j==True:

            k.append(i)

    p=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

    m=pd.DataFrame(p[c].loc[k])

    p.drop(k,inplace=True)

    u=p[c].mode()

    p.fillna(u,inplace=True)

    t1=p[c].values

    t2=p[a].values

    t1=t1.reshape(-1,1)

    from sklearn.linear_model import LinearRegression

    lin_reg=LinearRegression()

    lin_reg.fit(t1,t2)

    l=m.values

    y=lin_reg.predict(l)

    y = [round(x) for x in y]

    q=0

    for i,j in t.items():

        if j==True:

               b[a].iloc[i]=y[q]

               q=q+1  
discrete("GarageYrBlt",train,"YearBuilt")
train["GarageYrBlt"].isnull().sum()
train['LotFrontage'].fillna(69,inplace=True)

train['LotFrontage'].isnull().sum()
train['MasVnrArea'].mean()
train['MasVnrArea'].fillna(0,inplace=True)
cat
for i in cat:

    sns.countplot(i,data=train)

    plt.show()

    t=train[i].isnull().sum()

    print(t)
train.drop(['MiscFeature','Fence','PoolQC','Alley'],axis=1,inplace=True)
k=['MiscFeature','Fence','PoolQC','Alley']



cat = [element for element in cat if element not in k]
cat
len(train.columns)
def categorical(ca,tttt):

    for al in ca:

        t=tttt[al].isnull()

        t=t.to_dict()  

        k=[]

        for i,j in t.items():

            if j==True:

                 k.append(i)

        p=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

        p.drop(['MiscFeature','Fence','PoolQC','Alley'],axis=1,inplace=True)

        z=pd.DataFrame(p.iloc[k])

        z.drop([al],axis=1,inplace=True)

        p.drop(k,inplace=True)

        q=p[al].value_counts()

        q=q.to_dict()

        m=1

        for i,j in q.items():

            q[i]=m

            m=m+1

        p.replace({al:q},inplace=True)

        t1=p[al].values

        p.drop([al],axis=1,inplace=True)

        p.fillna(0,inplace=True)

        def split(a):

            num=a.select_dtypes(include=[np.number]) 

            cat=a.select_dtypes(exclude=[np.number])

            cat=pd.get_dummies(cat)

            return num,cat

         

        x,y=split(p)

        scaler = preprocessing.StandardScaler()

        def scaling(x,y):

             features_scaled=scaler.fit_transform(x.values)

             q=y.values

             vk=np.concatenate((q,features_scaled),axis=1)

             return vk

        features=scaling(x,y)

        (x_train,x_test,y_train,y_test) = train_test_split(features,t1, train_size=0.75, random_state=42)

        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=100, random_state=42)

        rnd_clf.fit(x_train,y_train)

        z.fillna(0,inplace=True)

        r,s=split(z)

        for w in y.columns:

            for o in s.columns:

                if(w==o):

                    c=1

                    break

                else:

                    c=2

            if(c==2):

                s[w]=0

        for o in s.columns:

                for w in y.columns:

                    if(o==w):

                        c=1

                        break

                    else:

                        c=2

                if(c==2):

                    s.drop(o,axis=1,inplace=True)

     

        for oe in r.columns:

            for we in x.columns:

                if(we==oe):

                    c=1

                    break

                else:

                    c=2

            if(c==2):

                r.drop(oe,axis=1,inplace=True)

        for we in x.columns:

            for oe in r.columns:

                if(we==oe):

                    c=1

                    break

                else:

                    c=2

            if(c==2):

                r[we]=0

            

                    

       

        features=scaling(r,s)

        y_pred=rnd_clf.predict(features)

        h=[]

        for l in range(len(y_pred)):

                u=[i for i,j in q.items() if j==y_pred[l]]

                for tyyy in u:

                    h.append(tyyy)

        x=0

        for i,j in t.items():

                if j==True:

                       tttt[al].iloc[i]=h[x]

                       x=x+1  

        
categorical(cat,train)
len(train.columns)
train.isnull()
test.info()
a=[]

missvalue(test)

num=[]

cat=[]

separate(test,a)

num2=[]

num1=[]

cor(num)

discrete('BsmtFinSF1',test,"BsmtFullBath")

discrete('TotalBsmtSF',test,"1stFlrSF")

discrete('BsmtFullBath',test,"BsmtFinSF1")
test["YearBuilt"].mode()


discrete('GarageCars',test,"GarageArea")
discrete('GarageArea',test,"GarageCars")
test["YearBuilt"].isnull().sum()
test["GarageYrBlt"].isnull().sum()
train["GarageYrBlt"].unique()
train["YearBuilt"].unique()
test["GarageYrBlt"].fillna(2005,inplace=True)
fig(num2,train)
test["LotFrontage"]
test["LotFrontage"].fillna(test["LotFrontage"].median(),inplace=True)
test["MasVnrArea"].fillna(test["MasVnrArea"].median(),inplace=True)
test["BsmtFinSF2"].fillna(test["BsmtFinSF2"].mode(),inplace=True)

test["BsmtUnfSF"].fillna(test["BsmtUnfSF"].median(),inplace=True)

test["BsmtHalfBath"].fillna(test["BsmtHalfBath"].mode(),inplace=True)
cat
for i in cat:

    sns.countplot(i,data=test)

    plt.show()

    t=test[i].isnull().sum()

    print(t)
test.drop(['MiscFeature','Fence','PoolQC','Alley'],axis=1,inplace=True)

k=['MiscFeature','Fence','PoolQC','Alley']



cat = [element for element in cat if element not in k]
cat
len(test.columns)
test['MasVnrType'].value_counts()

def categori(ca,tttt):

    for al in ca:

        t=tttt[al].isnull()

        t=t.to_dict()  

        k=[]

        for i,j in t.items():

            if j==True:

                 k.append(i)

        p=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

        p.drop(['MiscFeature','Fence','PoolQC','Alley'],axis=1,inplace=True)

        z=pd.DataFrame(p.iloc[k])

        z.drop([al],axis=1,inplace=True)

        p.drop(k,inplace=True)

        q=p[al].value_counts()

        q=q.to_dict()

        m=1

        for i,j in q.items():

            q[i]=m

            m=m+1

        p.replace({al:q},inplace=True)

        t1=p[al].values

        p.drop([al],axis=1,inplace=True)

        p.fillna(0,inplace=True)

        def split(a):

            num=a.select_dtypes(include=[np.number]) 

            cat=a.select_dtypes(exclude=[np.number])

            cat=pd.get_dummies(cat)

            return num,cat

         

        x,y=split(p)

        scaler = preprocessing.StandardScaler()

        def scaling(x,y):

             features_scaled=scaler.fit_transform(x.values)

             q=y.values

             vk=np.concatenate((q,features_scaled),axis=1)

             return vk

        features=scaling(x,y)

        (x_train,x_test,y_train,y_test) = train_test_split(features,t1, train_size=0.75, random_state=42)

        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=100, random_state=42)

        rnd_clf.fit(x_train,y_train)

        z.fillna(0,inplace=True)

        r,s=split(z)

        for w in y.columns:

            for o in s.columns:

                if(w==o):

                    c=1

                    break

                else:

                    c=2

            if(c==2):

                s[w]=0

        for o in s.columns:

                for w in y.columns:

                    if(o==w):

                        c=1

                        break

                    else:

                        c=2

                if(c==2):

                    s.drop(o,axis=1,inplace=True)

     

        for oe in r.columns:

            for we in x.columns:

                if(we==oe):

                    c=1

                    break

                else:

                    c=2

            if(c==2):

                r.drop(oe,axis=1,inplace=True)

        for we in x.columns:

            for oe in r.columns:

                if(we==oe):

                    c=1

                    break

                else:

                    c=2

            if(c==2):

                r[we]=0

            

                    

       

        features=scaling(r,s)

        y_pred=rnd_clf.predict(features)

        h=[]

        for l in range(len(y_pred)):

                u=[i for i,j in q.items() if j==y_pred[l]]  

                for tyyy in u:

                    h.append(tyyy)

        x=0

        for i,j in t.items():

                if j==True:

                       tttt[al].iloc[i]=h[x]

                       x=x+1  

        
categori(cat,test)
len(test.columns)
num=[]

cat=[]

def sep(tv):

    for i in tv.columns:

        if(tv[i].dtypes=='O'):

            cat.append(i)

        else:

            num.append(i)
sep(train)
cont=[]

disc=[]

def numsep():

    for i in num:

        t=train[i].value_counts()

        qw=len(t)

        if qw>20:

            cont.append(i)

        else:

            disc.append(i)

        


numsep()
cont
disc
for i in cont:

      train[i].hist()

      plt.xlabel(i)

      plt.show()
sss=['LotFrontage','LotArea','1stFlrSF','GrLivArea']

for f in sss:

    train[f]=np.log(train[f])
disc
cont
train['LotFrontage'].hist()
num=[]

cat=[]

sep(test)

cont=[]

disc=[]

numsep()
for i in cont:

    test[i].hist()

    plt.xlabel(i)

    plt.show()


ss=['LotFrontage','LotArea','1stFlrSF','GrLivArea']

for q in ss:

    test[q]=np.log(test[q])
pp=[]

corr=train.corr()

t=corr["SalePrice"].sort_values(ascending=False)

t=t.to_dict()

for i,j in t.items():

    if j>0.5 or (0.2>j and j>-0.2):

        print(i,j)

        print("*"*40)

        if 0.2>j and j>-0.2:

            pp.append(i)

pp
num=[]

sep(train)
for i in num:

    for j in num:

        train["new"]=train[j]/train[i]

        corr=train.corr()

        t=corr["new"].sort_values(ascending=False)

        t=t.to_dict()

        for r,s in t.items():

            if (s>0.6 or s<-0.6) and r=='SalePrice':

                print(i,j)

                print(s)

                print("*"*40)



                

       



        

train.drop(['new'],axis=1,inplace=True)
def add(w):

    w["LotFrontage OverallQual"]=w["OverallQual"]/w["LotFrontage"]

    w["LotArea OverallQual"]=w["OverallQual"]/w["LotArea"]

    w["OverallCond OverallQual"]=w["OverallQual"]/w["OverallCond"]

    w["YearBuilt OverallQual"]=w["OverallQual"]/w["YearBuilt"]

    w["YearBuilt TotalBsmtSF"]=w["TotalBsmtSF"]/w["YearBuilt"]

    w["YearBuilt GarageCars"]=w["GarageCars"]/w["YearBuilt"]

    w["YearBuilt GarageArea"]=w["GarageArea"]/w["YearBuilt"]

    w["YearRemodAdd OverallQual"]=w["OverallQual"]/w["YearRemodAdd"]

    w["YearRemodAdd TotalBsmtSF"]=w["TotalBsmtSF"]/w["YearRemodAdd"]

    w["YearRemodAdd GrLivArea"]=w["GrLivArea"]/w["YearRemodAdd"]

    w["YearRemodAdd GarageCars"]=w["GarageCars"]/w["YearRemodAdd"]

    w["YearRemodAdd GarageArea"]=w["GarageArea"]/w["YearRemodAdd"]

    w["1stFlrSF OverallQual"]=w["OverallQual"]/w["1stFlrSF"]

    w["1stFlrSF TotalBsmtSF"]=w["TotalBsmtSF"]/w["1stFlrSF"]

    w["LowQualFinSF ScreenPorch"]=w["ScreenPorch"]/w["LowQualFinSF"]

    w["GrLivArea OverallQual"]=w["OverallQual"]/w["GrLivArea"]

    w["GrLivArea YrSold"]=w["YrSold"]/w["GrLivArea"]

  

  

    w["GarageYrBlt OverallQual"]=w["OverallQual"]/w["GarageYrBlt"]

    w["GarageYrBlt TotalBsmtSF"]=w["TotalBsmtSF"]/w["GarageYrBlt"]

    w["GarageYrBlt GarageCars"]=w["GarageCars"]/w["GarageYrBlt"]

    w["GarageYrBlt GarageArea"]=w["GarageArea"]/w["GarageYrBlt"]

  

    w["YrSold OverallQual"]=w["OverallQual"]/w["YrSold"]

    w["YrSold TotalBsmtSF"]=w["TotalBsmtSF"]/w["YrSold"]

    w["YrSold GrLivArea"]=w["GrLivArea"]/w["YrSold"]

    w["YrSold GarageArea"]=w["GarageArea"]/w["YrSold"]

    w["GarageYrBlt TotalBsmtSF"]=w["TotalBsmtSF"]/w["GarageYrBlt"]

    w["GarageYrBlt GarageCars"]=w["GarageCars"]/w["GarageYrBlt"] 

    
add(train)
train.drop(pp,axis=1,inplace=True)
add(test)
len(test.columns)
test.drop(pp,axis=1,inplace=True)
len(test.columns)
len(train.columns)
label=train["SalePrice"].values

label
train.drop(["SalePrice"],axis=1,inplace=True)
trtr=["LowQualFinSF ScreenPorch"]

train.drop(trtr,axis=1,inplace=True)
num=[]

cat=[]

sep(train)
def split(a):

    num=a.select_dtypes(include=[np.number]) 

    cat=a.select_dtypes(exclude=[np.number])

    cat=pd.get_dummies(cat)

    return num,cat
x,y=split(train)
scaler=StandardScaler()

def scaling(x,y):

     features_scaled=scaler.fit_transform(x.values)

     q=y.values

     vk=np.concatenate((q,features_scaled),axis=1)

     return vk
features=scaling(x,y)
np.random.seed(1234)

(x_train,x_test,y_train,y_test) = train_test_split(features,label, train_size=0.75, random_state=42)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)
scores = cross_val_score(lin_reg, x_test,y_test,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)

tree_rmse_scores.mean()
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(x_train,y_train)
y_pred = tree_reg.predict(x_test)

lin_mse = mean_squared_error(y_test,y_pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
scores = cross_val_score(tree_reg, x_test,y_test,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
tree_rmse_scores.mean()
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(x_train,y_train)
y_pred = forest_reg.predict(x_test)

lin_mse = mean_squared_error(y_test,y_pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")

svm_reg.fit(x_train, y_train)

y_pred = svm_reg.predict(x_test)

svm_mse = mean_squared_error(y_test,y_pred)

svm_rmse = np.sqrt(svm_mse)

svm_rmse
from sklearn.model_selection import GridSearchCV



param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)

grid_search.fit(x_train,y_train)
yt=grid_search.best_estimator_
yt
y_pred = yt.predict(x_test)

lin_mse = mean_squared_error(y_test,y_pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
k=[1,2,3,4,5,100]

for i in k:

    svm=SVR(kernel="linear",epsilon=i)

    svm.fit(x_train,y_train)

    y_pred = svm_reg.predict(x_test)

    svm_mse = mean_squared_error(y_test,y_pred)

    svm_rmse = np.sqrt(svm_mse)

    print(svm_rmse,i)

    print("*"*40)


deg=[2,3,4,5,6]

c=[1,3,10,15,20,100]



for j in deg:

    for ci in c:

            svm=SVR(kernel="rbf",degree=j,C=ci,gamma="scale")

            svm.fit(x_train,y_train)

            scores = cross_val_score(svm, x_test,y_test,

                         scoring="neg_mean_squared_error", cv=10)

            rmse_scores = np.sqrt(-scores)

            print(rmse_scores.mean(),j,ci)
deg=[2,3,4,5,6]

c=[1,3,10,15,20,100]

epsilon=[1.5,2,3]

for j in deg:

    for ci in c:

        for e in epsilon:

                svm=SVR(kernel="rbf",degree=j,C=ci,epsilon=e,gamma="scale")

                svm.fit(x_train,y_train)

                scores = cross_val_score(svm, x_test,y_test,

                             scoring="neg_mean_squared_error", cv=10)

                rmse_scores = np.sqrt(-scores)

                print(rmse_scores.mean(),j,ci,e)
deg=[2,3,4,5,6]

c=[1,3,10,15,20,100]

for j in deg:

    for ci in c:

                svm=SVR(kernel="rbf",degree=j,C=ci,epsilon=0.1,gamma="scale")

                svm.fit(x_train,y_train)

                scores = cross_val_score(svm, x_test,y_test,

                             scoring="neg_mean_squared_error", cv=10)

                rmse_scores = np.sqrt(-scores)

                print(rmse_scores.mean(),j,ci)
deg=[2,3,4,5,6]

c=[1,3,10,15,20,100]

gamma=[0.01, 0.03, 0.1]

for j in deg:

    for ci in c:

        for k in gamma:

                svm=SVR(kernel="rbf",degree=j,C=ci,gamma=k)

                svm.fit(x_train,y_train)

                scores = cross_val_score(svm, x_test,y_test,

                             scoring="neg_mean_squared_error", cv=10)

                rmse_scores = np.sqrt(-scores)

                print(rmse_scores.mean(),j,ci,k)
deg=[2,3,4,5,6]

c=[1,3,10,15,20,100]

gamma=[0.01, 0.03, 0.1]

for j in deg:

    for ci in c:

        for k in gamma:

                svm=SVR(kernel="linear",degree=j,C=ci,gamma=k)

                svm.fit(x_train,y_train)

                scores = cross_val_score(svm, x_test,y_test,

                             scoring="neg_mean_squared_error", cv=10)

                rmse_scores = np.sqrt(-scores)

                print(rmse_scores.mean(),j,ci,k)
svm=SVR(kernel="linear",C=100)

svm.fit(x_train,y_train)

scores = cross_val_score(svm, x_test,y_test,

                             scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean(),j,ci,k)
c=[100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,5500,5550]

for i in c:

    svm=SVR(kernel="linear",C=i)

    svm.fit(x_train,y_train)

    scores = cross_val_score(svm, x_test,y_test,

                                 scoring="neg_mean_squared_error", cv=10)

    rmse_scores = np.sqrt(-scores)

    print(rmse_scores.mean(),i)
tree_reg = DecisionTreeRegressor(max_depth=2,random_state=42)

tree_reg.fit(x_train,y_train)

scores = cross_val_score(tree_reg, x_test,y_test,

                                 scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
m=[1,2,3,4,5,6,7,8,9,100,150,200,250,300]



for i in m:

    tree_reg = DecisionTreeRegressor(max_depth=i,random_state=42)

    tree_reg.fit(x_train,y_train)

    scores = cross_val_score(tree_reg, x_test,y_test,

                                 scoring="neg_mean_squared_error", cv=10)

    rmse_scores = np.sqrt(-scores)

    print(rmse_scores.mean(),i)
m=[1,2,3,4,5,6,7,8,9,100,150,200,250,300]

mi=[2,3,4]

for i in m:

    for j in mi:

        tree_reg = DecisionTreeRegressor(max_depth=i,min_samples_split=j,random_state=42)

        tree_reg.fit(x_train,y_train)

        scores = cross_val_score(tree_reg, x_test,y_test,

                                     scoring="neg_mean_squared_error", cv=10)

        rmse_scores = np.sqrt(-scores)

        print(rmse_scores.mean(),i,j)
mi=[2,3,4]

for j in mi:

    for k in range(2,100):

            tree_reg = DecisionTreeRegressor(min_samples_split=j,max_leaf_nodes=k,random_state=42)

            tree_reg.fit(x_train,y_train)

            scores = cross_val_score(tree_reg, x_test,y_test,

                                         scoring="neg_mean_squared_error", cv=10)

            rmse_scores = np.sqrt(-scores)

            print(rmse_scores.mean(),j,k)
mi=[2,3,4]

max=[1,2,3,4,5,6,7,8,9,10,11,12,13]

for m in max:

    for j in mi:

        for k in range(10,20):

                tree_reg = DecisionTreeRegressor(min_samples_split=j,max_leaf_nodes=k,max_depth=m,random_state=42)

                tree_reg.fit(x_train,y_train)

                scores = cross_val_score(tree_reg, x_test,y_test,

                                             scoring="neg_mean_squared_error", cv=10)

                rmse_scores = np.sqrt(-scores)

                print(rmse_scores.mean(),j,k)
rnd_clf = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1,random_state=42)

rnd_clf.fit(x_train, y_train)

scores = cross_val_score(rnd_clf, x_test,y_test,scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
rnd_clf = RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=42)

rnd_clf.fit(x_train, y_train)

scores = cross_val_score(rnd_clf, x_test,y_test,scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
n_estimators=[10,20,30,40]

for i in n_estimators:

    rnd_clf = RandomForestRegressor(n_estimators=i,random_state=42)

    rnd_clf.fit(x_train, y_train)

    scores = cross_val_score(rnd_clf, x_test,y_test,scoring="neg_mean_squared_error", cv=10)

    rmse_scores = np.sqrt(-scores)

    print(rmse_scores.mean())
n_estimators=[50,60,70,80,90,100]

max=[8,9,10,11,12]

for i in n_estimators:

    for j in max:

        rnd_clf = RandomForestRegressor(n_estimators=i,random_state=42,max_features=j)

        rnd_clf.fit(x_train, y_train)

        scores = cross_val_score(rnd_clf, x_test,y_test,scoring="neg_mean_squared_error", cv=10)

        rmse_scores = np.sqrt(-scores)

        print(rmse_scores.mean())
import xgboost
xgb_reg = xgboost.XGBRegressor(random_state=42)

xgb_reg.fit(x_train, y_train)
scores = cross_val_score(xgb_reg, x_test,y_test,scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
params={

    "learning_rate":[0.05,0.10,0.15,0.20,0.30],

    "max_depth":[3,4,5,6,7,8,9,10,12,15,20,30,40,50],

    "min_child_weight":[1,3,5,7,9,11,12,13,14,16,19,25,30],

    "gamma":[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1],

    "colsample_bytree":[0.3,0.4,0.5,0.6,0.7,0.8],

}
rs=RandomizedSearchCV(xgb_reg,param_distributions=params,n_iter=10,scoring='neg_mean_squared_error',cv=5)
rs.fit(x_train,y_train)
rs.best_params_
rs.best_estimator_
xgb=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.4, gamma=0.1, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.2, max_delta_step=0, max_depth=12,

             min_child_weight=30, monotone_constraints='()',

             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=42,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
scores = cross_val_score(xgb, x_test,y_test,scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
test.drop(trtr,axis=1,inplace=True)
num=[]

cat=[]

sep(test)

x1,y1=split(test)

num=[]

cat=[]

sep(test)

x,y=split(train)

a=[]

for i in y.columns:

    n=-1

    for j in y1.columns:

        n=n+1

        if i==j:

            break

        elif n==222:

            a.append(i)
for i in a:

    y1[i]=y[i]
len(y1.columns)
features1=scaling(x1,y1)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(features1)
fffff.head()
r = pd.Series(y_pred,name="SalePrice")
fffff.tail
r
submission = pd.concat([pd.Series(range(1461,2920),name = "Id"),r],axis = 1)
submission
submission.to_csv("regression.csv",index=False)