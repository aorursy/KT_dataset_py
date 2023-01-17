import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

t = pd.read_csv('../input/train.csv')
ts = pd.read_csv('../input/test.csv')

data = pd.concat([t.drop('Survived',axis=1),ts])
data.shape
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
ts.head(2)
#Title <-Name
def name_clean(df):
    df['Name_len'] = df.Name.apply(lambda x:len(x))
    df.Name_len = pd.cut(df.Name_len,10)
    Others = ['Khalil,','Brito,','Palmquist,','Dr.','Rev.', 'y', 'Planke,','Impe,', 'Col.', 'Gordon,', 'Major.', 'Carlo,', 'Velde,','Melkebeke,', 'Cruyssen,', 'Messemaeker,', 'Pelsmaeker,','Shawah,', 'Capt.','Don.', 'der', 'Jonkheer.', 'the', 'Steen,','Walle,', 'Mulder,', 'Billiard,']
    df['Last_Name'] = df.Name.str.split(' ').apply(lambda x:x[0])
    df['Title'] = df.Name.str.split(' ').apply(lambda x:x[1])
    df.Title.replace(['Mlle.','Mme.','Ms.'],['Miss.','Mrs.','Mr.'],inplace=True)
    df.Title.replace(Others,'Other', inplace=True)
    df.drop('Name', axis = 1, inplace=True)
    return df
#Age
#     age_group = [-100,5,10,20,30,200]
#     group_name = ['Baby','Child','Teen','Adult','Senior']
def age_clean(df):
#    df.Age = df.Age.fillna(data.Age.mean())
    df.Age = pd.cut(df.Age.astype(int),5)
    return df
#SibSp+Prch ->Family+Alone
def family(df):
    df['Family']=df.SibSp+df.Parch
    df['Alone']= (df.Family<=1).as_matrix().astype(int)
    df.drop(['SibSp','Parch'], axis=1, inplace=True)
    return df
#Ticket
#    df['Ticket_lett'] = df.Ticket.apply(lambda x:str(x)[0])
def ticket_clean(df):
    df['Ticket_len'] = df.Ticket.apply(lambda x:len(x))
    df.drop('Ticket',axis=1,inplace=True)
    return df
#Fare
#     df.Fare[(df.Fare==0)&(df.Pclass==1)]=df.Fare[df.Pclass==1].mean()
#     df.Fare[(df.Fare==0)&(df.Pclass==2)] = df.Fare[df.Pclass==2].mean()
#     df.Fare[(df.Fare==0)&(df.Pclass==3)] = df.Fare[df.Pclass==3].mean()
#     Fare = [-1000,5,20,40,100,300,1000]
#     Fare_cat = ['a','b','c','d','e','f']
def fare_clean(df):
#    df.Fare = df.Fare.fillna(data.Fare.median())
    df.Fare = pd.cut(df.Fare,10)
    return df
#Cabin
#df = t.copy()
def cabin_clean(df):
    df.Cabin.fillna('N',inplace=True)
    df.Cabin = df.Cabin.apply(lambda x:x[0])
    return df
#cabin_clean(df)
#Embarked
def embarked(df):
    df.Embarked.fillna(data.Embarked.mode()[0],inplace=True)
    return df
def clean_features(df):
    df = name_clean(df)
#    df = age_clean(df)
    df = family(df)
#    df = fare_clean(df)
    df = ticket_clean(df)
    df = cabin_clean(df)
    df = embarked(df)
    return df
#Transform data

train_c1 = clean_features(t.copy())
test_c1 = clean_features(ts.copy())
train_c1.Name_len.min()
from sklearn import preprocessing
def lebelEnCoding(df_train, df_test):
    Features = ['Sex','Cabin','Embarked','Name_len','Last_Name','Title','Family','Alone','Ticket_len']
    df = pd.concat([df_train[Features], df_test[Features]])
    
    for F in Features:
        le = preprocessing.LabelEncoder()
        le.fit(df[F])
        df_train[F] = le.transform(df_train[F])
        df_test[F] = le.transform(df_test[F])
    return df_train, df_test
train_c2, test_c2 = lebelEnCoding(train_c1.copy(),test_c1.copy())
train_c2.head()
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler

def age_fare(df):
    return pd.DataFrame(NuclearNormMinimization().fit_transform(df))
#    return pd.DataFrame(KNN(k=5).fit_transform(df))
train_c3 = age_fare(train_c2.copy())
test_c3 = age_fare(test_c2.copy())
train_c3.columns = train_c2.columns
test_c3.columns = test_c2.columns
# ind = train_c2[train_c2.isnull().sum(axis=1)>0].index
# train_c3.iloc[ind]
def f_clean(df):
    df = age_clean(df)
    df = fare_clean(df)
    return df
train_c4 = f_clean(train_c3.copy())
test_c4 = f_clean(test_c3.copy())
def lebelEnCoding1(df_train, df_test):
    Features = ['Age','Fare']
#    Features = ['Sex','Cabin','Embarked','Name_len','Last_Name','Title','Family','Alone','Ticket_len']
    df = pd.concat([df_train[Features], df_test[Features]])
    
    for F in Features:
        le = preprocessing.LabelEncoder()
        le.fit(df[F])
        df_train[F] = le.transform(df_train[F])
        df_test[F] = le.transform(df_test[F])
    return df_train, df_test
train,test = lebelEnCoding1(train_c4.copy(),test_c4.copy())
from sklearn.model_selection import train_test_split
X = train.drop(['Survived','PassengerId'],axis=1)
Y = train.Survived

X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 118, max_depth=5, random_state=10)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, accuracy_score

# score = make_scorer(accuracy_score)
# rf = RandomForestClassifier()
# parameters = {'n_estimators':[118,119,120],
#               'criterion':['entropy','gini'],
#               'max_depth':[5,6,7],
#               'min_samples_split':[7,8],
#               'min_samples_leaf':[1,2,3],
#               'max_features':['auto','sqrt','log2'],
#               'random_state':[8,10,12,15,40]}
# grid = GridSearchCV(rf,parameters,scoring=score)
# grid.fit(X_train,y_train)

# grid.best_score_

# grid.best_params_
rf = RandomForestClassifier(criterion='entropy',
 max_depth= 5,
 max_features='log2',
 min_samples_leaf= 1,
 min_samples_split= 4,
 n_estimators= 540,
 oob_score=True,
 random_state=10)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
rf.fit(X,Y)
result = rf.predict(test.drop('PassengerId',axis=1))
results = pd.DataFrame({'PassengerId':test.PassengerId.astype(int), 'Survived':result.astype(int)})
results.to_csv('My_sub1.csv', index=False)