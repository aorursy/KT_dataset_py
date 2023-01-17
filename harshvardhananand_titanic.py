from sklearn.linear_model import SGDClassifier as sgd
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier as xgboost
from xgboost.sklearn import XGBRegressor as xgboostR

ss = StandardScaler()

def model(x,y):
    x, testx, y, testy = train_test_split(x, y, test_size=0.1, random_state=4355)
    all_models = ['sgd','svc','knn','rf','dtc', 'lr', 'LDA', 'xgboost']
    models = [sgd,SVC, knn,rf,dtc,LogisticRegression, LDA, xgboost ]
    for i,j in zip(models,all_models):
        try:
            model = i(random_state=344,n_components=5)
        except:
            try:
                model = i(random_state=344)
            except:
                model = i()
        model.fit(x,y)
        trsc = model.score(x,y)
        sc = model.score(testx,testy)
        print(f'For {j} we have train score of {trsc} and test score of {sc}')
        
def modelR(x,y):
    x, testx, y, testy = train_test_split(x, y, test_size=0.1, random_state=4355)
    all_models = ['svr','rfr','dtr', 'lreg', 'xgboostR']
    models = [SVR,rfr,dtr,LinearRegression, xgboostR ]
    for i,j in zip(models,all_models):
        try:
            model = i(random_state=344,n_components=5)
        except:
            try:
                model = i(random_state=344)
            except:
                model = i()
        model.fit(x,y)
        trsc = model.score(x,y)
        sc = model.score(testx,testy)
        print(f'For {j} we have train score of {trsc} and test score of {sc}')      
        
        
def scaler(df):
    df = df.apply(lambda x: (x-x.mean())/x.std())
    return df
        
def pca(xd,n):
    pca = PCA(n_components=n)
    xd = pca.fit(xd).transform(xd)
    print(pca.explained_variance_ratio_)
    return xd

def setfig(w=10,h=7):
    return plt.figure(figsize=[w,h])
            
def predS(model, test_data):
    pred = model.predict(test_data)
    test_data = pd.read_csv('../input/titanic/test.csv')
    td = {'PassengerId':test_data.PassengerId, 'Survived':pred}
    pd.DataFrame(td).to_csv('Submission.csv',index=False)
import matplotlib.pyplot as plt
import seaborn as sns
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# original data 
odata = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
odata.head(3)
odata.isna().sum()
test_data.isna().sum()
len(odata)
177/891
odata.info()
odata = odata.drop(['PassengerId','Ticket','Cabin'],axis=1)
odata.isna().sum()
odata.head()
data = odata.copy()
import re
# To select the max number of repetation
data.Name.str.extract(' (\w{1,8})\. ').isna().sum()
# All unique name suffixes.
pd.Series(data.Name.str.extract(' (\w{1,8})\. ').values.flatten()).unique()
data['suffix'] = data.Name.str.extract(' (\w{1,8})\. ')

data.drop('Name', axis=1, inplace=True)  # Now name is not required

data.head()
data.suffix.value_counts().plot.bar()
data.suffix = data.suffix.apply(lambda x: x if (x in ['Mr', 'Miss', 'Mrs', 'Master']) else 'rare')
data.suffix.value_counts().plot.bar()
data.isna().sum()
import math
print(sorted(set(data.Fare.apply(math.ceil))))
def faregrp(fare):
    if 0<=fare<=50:
        return 1
    if 50<fare<=100:
        return 2
    if 100<fare<=150:
        return 3
    if 150<fare<=200:
        return 4
    if 200<fare<=250:
        return 5
    if fare>250:
        return 6
data['grouped_fare'] = data.Fare.apply(faregrp)
setfig()
sns.heatmap(data.corr(), annot=True)
narows = data[data.Age.isna()]
narows
tempdata = data.dropna()
x = tempdata[['Parch', 'SibSp', 'Pclass']]
y = tempdata.Age

modelR(x,y)

agepred = rfr()
agepred.fit(x, y)

# Using Random forest to precict Null age
predicted = agepred.predict(narows[['Parch', 'SibSp', 'Pclass']])
data.Age.fillna(pd.Series(predicted, index=narows.index), inplace=True)
def agegrp(age):
    if 0<= age <=10:
        return 1
    if 10< age <=18:
        return 2
    if 18< age <=25:
        return 3
    if 25< age <=35:
        return 4
    if 35< age <=47:
        return 5
    if 47< age <=60:
        return 6
    if 60< age <=70:
        return 7
    if 70< age <=80:
        return 8
data['age_grp'] = data.Age.apply(agegrp)
data.Age[data.Age<=18].count()
data.Age[data.Survived==1][data.Age<=18].count()
data.head()
data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)
data.Embarked.mode()
data.head()
data.Pclass.value_counts().plot.bar()
data.SibSp.value_counts().plot.bar()
data.SibSp[data.Survived==1].value_counts().plot.bar(color='green')
data.SibSp[data.Survived==0].value_counts().plot.bar(color='red')
data.Parch.value_counts().plot.bar()
data.Embarked.value_counts().plot.bar()
data.age_grp.value_counts().plot.bar()
data['total_family_size'] = data.SibSp+data.Parch
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
data.corr()
data['familysize-fare+age+pclass'] = data.total_family_size - data.Fare+data.Age+data.Pclass
data.drop('total_family_size', axis=1, inplace=True)
data.corr()
data['age_fare'] = data.Fare-data.Age
data['agegrp_fare'] = data.Fare*data.age_grp

data.corr()
data.isna().sum()
data.suffix.unique()
dict(list(zip(['Mr', 'Mrs', 'Miss', 'Master', 'rare'], [1,2,3,4,5])))
mapping = dict(list(zip(['Mr', 'Mrs', 'Miss', 'Master', 'rare'], [1,2,3,4,5])))
data.suffix = data.suffix.map(mapping)

data.corr()
data['suff+fare'] = data.suffix+data.Fare
data['suff+Pclass'] = data.suffix-data.Pclass
data['suff-fs-fare+age+cls'] = data.suffix-data['familysize-fare+age+pclass']
data['suff+age_fare'] = data.suffix+data.age_fare
data['suff+agegrp_fare'] = data.suffix+data.agegrp_fare
# data.drop('age_grp', axis=1, inplace=True)
data.corr()
data.isnull().sum()
data
data.Embarked = data.Embarked.map({'C':1,'S':2, 'Q':3})
data['embarked_suffix'] = data.Embarked/data.suffix
# data.drop('Pclass', axis=1, inplace=True)
# data.drop('Embarked', axis=1, inplace=True)
data.corr()
data['pclass-suff+age'] = data.Pclass-data['suff+age_fare']
data['pclass/suff'] = data.Pclass/data.suffix
# data['pclass-suff'] = data.Pclass-data.suffix
data.corr()
data.Sex = data.Sex.map({'male':0, 'female':1})
data.corr()
plt.figure(figsize=[20, 9])
sns.heatmap(data.corr(), annot=True)
odata.Fare.median()
# sns.pairplot(data)
data.head()
v0 = data.copy()
data = v0.copy()
col_have_ol = ['Age', 'Fare','familysize-fare+age+pclass',
               'age_fare','agegrp_fare','suff+fare',
               'suff-fs-fare+age+cls','suff+age_fare',
               'pclass-suff+age','embarked_suffix','pclass/suff','suff+agegrp_fare']

# data[col_have_ol].describe().min()
have_neg = {}

for i in col_have_ol:
    if data[col_have_ol].describe().min()[i]<=0:
        min_val = data[col_have_ol].describe().min()[i]
        have_neg[i] = min_val
        data[i] = data[i].apply(lambda x: np.log10(x+1-min_val))
    else:
        data[i] = data[i].apply(lambda x: np.log10(x+1))# using x+1 so that 0 can be converted to 1 as log(0) is nan
    data[i] = ss.fit_transform(data[[i]])
have_neg
# sns.pairplot(data)
v1 = data.copy()
# Correlated Data
corrdata = data[data.corr().Survived[data.corr().Survived<-0.3].index.append(data.corr().Survived[data.corr().Survived>0.3].index).tolist()]
corrdata.head()
plt.figure(figsize=[20, 9])
sns.heatmap(corrdata.corr(), annot=True)
data.head()
alldata = data.copy()
x = alldata.drop('Survived', axis=1)
x = x[sorted(x)]
y = alldata.Survived
alldata.head()
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.1, random_state=4355)
xgb = xgboost(n_estimators=8, max_depth=3, random_state=345)
xgb.fit(trainx, trainy)

print(xgb.score(trainx, trainy))
print(xgb.score(testx, testy))

impcol = x.columns[xgb.feature_importances_.astype(bool)]  # <----------
x = x[impcol]

# XGBoost feature selected
impx = x.copy()  # For final model  <----------
impy = y.copy() # for final model   <----------

model(x,y)

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.1, random_state=4355)
xgb = xgboost(n_estimators=8, max_depth=3, random_state=345)
xgb.fit(trainx, trainy)

print(xgb.score(trainx, trainy))
print(xgb.score(testx, testy))
alldata1 = corrdata.copy()
x = alldata1.drop('Survived', axis=1)
x = x[sorted(x)]
y = alldata.Survived

corx = x.copy()   # For final model
cory = y.copy()   # For final model
model(x, y)
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.1, random_state=4355)
xgb = xgboost(n_estimators=8, max_depth=3, random_state=345)
xgb.fit(trainx, trainy)

print(xgb.score(trainx, trainy))
print(xgb.score(testx, testy))
# important columns

cor_impcol = x.columns[xgb.feature_importances_.astype(bool)]
x = x[x.columns[xgb.feature_importances_.astype(bool)]]
cor_impcol
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
trainx, testx, trainy, testy = train_test_split(corx[cor_impcol], cory, test_size=0.1, random_state=4355)
# trainx, testx, trainy, testy = train_test_split(impx, impy, test_size=0.1, random_state=4355)

inpshape = trainx.shape
inpshape
model = Sequential([
    Dense(10, activation='relu', input_shape=inpshape),
    Dense(13, activation='relu'),
    Dropout(0.3),
    Dense(17, activation='relu'),
    Dropout(0.3),
    Dense(13, activation='relu'),
    Dense(10, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])

model.summary()

model.compile(metrics=['accuracy'], loss=['binary_crossentropy'])
history = model.fit(trainx, trainy, batch_size=32, epochs=40, validation_split=0.1)
hist = history.history
history.history.keys()
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(['loss','val_loss'])
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
model.evaluate(testx, testy)
class Test():
    def __init__(self, data):
        self.data = data
    def get_data(self):
        self.data = self.data.drop(['PassengerId','Ticket','Cabin'],axis=1)  
        self.create_suffix()
        self.grp_fare()   
        self.grpage()
        self.data.Embarked.fillna('S', inplace=True)
        self.feature_generation()
        
        col_have_ol = ['Age', 'Fare','familysize-fare+age+pclass',
                       'age_fare','agegrp_fare','suff+fare',
                       'suff-fs-fare+age+cls','suff+age_fare',
                       'pclass-suff+age','embarked_suffix','pclass/suff',
                       'suff+agegrp_fare']
        
        have_negative = { 
                     'Fare': 0.0,
                     'familysize-fare+age+pclass': -476.3292,
                     'age_fare': -66.225,
                     'agegrp_fare': 0.0,
                     'suff-fs-fare+age+cls': -68.225,
                     'suff+age_fare': -65.225,
                     'pclass-suff+age': -479.3292
                   }
        
        for i in col_have_ol:
            if i in have_negative:
                self.data[i] = self.data[i].apply(lambda x: np.log10(x+1-have_negative[i]))
            else:
                self.data[i] = self.data[i].apply(lambda x: np.log10(x+1))# using x+1 so that 0 can be converted to 1 as log(0) is nan
            self.data[i] = ss.transform(self.data[[i]])
            
        
        return self.data
    
    def create_suffix(self):
        self.data['suffix'] = self.data.Name.str.extract(' (\w{1,8})\. ')
        self.data.drop('Name', axis=1, inplace=True)  # Now name is not required
        self.data.suffix = self.data.suffix.apply(lambda x: x if (x in ['Mr', 'Miss', 'Mrs', 'Master']) else 'rare')
        
    def faregrp(self, fare):
        if 0<=fare<=50:
            return 1
        if 50<fare<=100:
            return 2
        if 100<fare<=150:
            return 3
        if 150<fare<=200:
            return 4
        if 200<fare<=250:
            return 5
        if fare>250:
            return 6
        else:
            return 14.4542 # median of Fare of training Data
        
    def grp_fare(self):
        self.data.fillna(pd.Series(14.4542, index=[152]), inplace=True) # only one data have Fare=NaN
        self.data['grouped_fare'] = self.data.Fare.apply(self.faregrp)
        
    def agegrp(self,age):
        if 0<= age <=10:
            return 1
        if 10< age <=18:
            return 2
        if 18< age <=25:
            return 3
        if 25< age <=35:
            return 4
        if 35< age <=47:
            return 5
        if 47< age <=60:
            return 6
        if 60< age <=70:
            return 7
        if 70< age <=80:
            return 8
        else:
            return -1

    def grpage(self):
        narows = self.data[self.data.Age.isna()]
        predicted = agepred.predict(narows[['Parch', 'SibSp', 'Pclass']])
        self.data.Age.fillna(pd.Series(predicted, index=narows.index), inplace=True)
        self.data['age_grp'] = self.data.Age.apply(self.agegrp)
        
    def feature_generation(self):
        self.data['total_family_size'] = self.data.SibSp + self.data.Parch
        self.data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

        self.data['familysize-fare+age+pclass'] = self.data.total_family_size - self.data.Fare + self.data.Age + self.data.Pclass
        self.data.drop('total_family_size', axis=1, inplace=True)

        self.data['age_fare'] = self.data.Fare - self.data.Age
        self.data['agegrp_fare'] = self.data.Fare * self.data.age_grp
        
        mapping = dict(list(zip(['Mr', 'Mrs', 'Miss', 'Master', 'rare'], [1,2,3,4,5])))
        self.data.suffix = self.data.suffix.map(mapping)

        self.data['suff+fare'] = self.data.suffix+self.data.Fare
        self.data['suff+Pclass'] = self.data.suffix-self.data.Pclass
        self.data['suff-fs-fare+age+cls'] = self.data.suffix-self.data['familysize-fare+age+pclass']
        self.data['suff+age_fare'] = self.data.suffix+self.data.age_fare
        self.data['suff+agegrp_fare'] = self.data.suffix+self.data.agegrp_fare

        self.data.Embarked = self.data.Embarked.map({'C':1,'S':2, 'Q':3})
        
        self.data['embarked_suffix'] = self.data.Embarked/self.data.suffix

        self.data['pclass-suff+age'] = self.data.Pclass-self.data['suff+age_fare']
        self.data['pclass/suff'] = self.data.Pclass/self.data.suffix

        self.data.Sex = self.data.Sex.map({'male':0, 'female':1})
import pandas as pd
import numpy as np
test_data = Test(pd.read_csv('../input/titanic/test.csv'))
test_data = test_data.get_data()
test_data.head()
# All those columns that are in train but not in test. Those columns will be added

for i in impx.columns:
    if i not in test_data.columns:
        print(i)
#         test_data[i] = np.zeros(len(test_data))
# All those columns that are in test but not in train. Those columns will be dropped

for i in test_data.columns:
    if i not in impx.columns:
        print(i)
#         test_data.drop(i, axis=1, inplace=True)
test_data = test_data[sorted(test_data)]
# test_data.isna().sum()[test_data.isna().sum()!=0]
# Final Model training data

Fmodel = xgboost(n_estimators=8, max_depth=3, random_state=345)
Fmodel.fit(impx, impy)

print(Fmodel.score(impx, impy))
predS(Fmodel, test_data[impx.columns])
# predS(model, test_data[impx.columns])