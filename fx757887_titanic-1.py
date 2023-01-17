# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import StandardScaler

# GridSearchCV

from sklearn.model_selection import GridSearchCV



from sklearn.neighbors import KNeighborsClassifier #KNN





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

data_df=train_df.append(test_df)
data_df.head()
#Nameの整理

data_df["Title"]=data_df["Name"]



#"Name"の最初の文字列抜出

for name_string in data_df["Name"]:

    data_df["Title"]=data_df["Name"].str.extract('([A-Za-z]+)\.', expand=True)
data_df["Title"].unique()
#重複する敬称をまとめる   

mapping={'Mlle':'Miss','Major':'Mr', 'Col':'Mr', 'Sir':'Mr','Don':'Mr','Mme':'Miss','Jonkheer':'Mr','Lady':'Mrs','Capt':'Mr','Countess':'Mrs','Ms':'Miss','Dona':'Mrs'}

data_df.replace({'Title':mapping},inplace=True)
data_df["Title"].unique()
titles=['Dr','Master','Miss','Mr','Mrs','Rev']

for title in titles:

    age_to_impute=data_df.groupby('Title')['Age'].median()[titles.index(title)]

    data_df.loc[(data_df['Age'].isnull())&(data_df['Title']==title), 'Age']=age_to_impute

    

train_df['Age']=data_df['Age'][:891]

test_df['Age']=data_df['Age'][891:]



data_df.drop('Title', axis=1, inplace=True)
data_df.head()
#「Family_Size」の生成←Parch（兄弟、配偶者）+SibSp（親、子供）使用

data_df['Family_Size']=data_df['Parch']+data_df['SibSp']



train_df['Family_Size']=data_df['Family_Size'][:891]

test_df['Family_Size']=data_df['Family_Size'][891:]
data_df['Family_Size'].head()
data_df.head(2)
data_df["Fare"].isnull().sum()
#FARE BINS生成

data_df['Fare'].fillna(data_df['Fare'].median(), inplace=True)
data_df["Fare"].isnull().sum()
data_df["Fare"].describe()
counts=data_df["Fare"].value_counts()
counts.plot.bar()
#分割、qcut　指定した数分割

data_df["FareBin"]=pd.qcut(data_df["Fare"],5)
data_df["FareBin"]
data_df["Fare"].plot.hist(title="Fare")
label=LabelEncoder()

data_df['FareBin_Code']=label.fit_transform(data_df['FareBin'])
data_df["FareBin_Code"].describe()
train_df['FareBin_Code']=data_df['FareBin_Code'][:891]

test_df['FareBin_COde']=data_df['FareBin_Code'][891:]



train_df.drop(['Fare'],1,inplace=True)

test_df.drop(['Fare'], 1, inplace=True)
#年齢ビンを作る

data_df['AgeBin']=pd.qcut(data_df['Age'], 5)
data_df["AgeBin"]
label=LabelEncoder()

data_df['AgeBin_Code']=label.fit_transform(data_df['AgeBin'])



train_df['AgeBin_Code']=data_df['AgeBin_Code'][:891]

test_df['AgeBin_Code']=data_df['AgeBin_Code'][891:]



train_df.drop(['Age'], 1, inplace=True)

test_df.drop(['Age'], 1, inplace=True)
#性別のマッピング、データクリーニング

train_df['Sex'].replace(['male','female'], [0,1], inplace=True)

test_df['Sex'].replace(['male','female'], [0,1], inplace=True)



train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

test_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
train_df.head()
train_df.describe()
train_df.isnull().sum()
#Traing

#X and yの生成

X=train_df.drop('Survived',1)

y=train_df['Survived']

X_test=test_df.copy()
#Scalingスケーリング

std_scaler=StandardScaler()

X=std_scaler.fit_transform(X)

#X_test=std_scaler.fit_transform(X_test)

X_test=std_scaler.transform(X_test)

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = list(range(1,50,5))

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 

                cv=10, scoring = "roc_auc")

gd.fit(X, y)

print(gd.best_score_)

print(gd.best_estimator_)
gd.best_estimator_.fit(X,y)

y_pred=gd.best_estimator_.predict(X_test)

print(y_pred)
knn=KNeighborsClassifier(algorithm="auto",leaf_size=26,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=6,p=2,weights='uniform')

knn.fit(X,y)

y_pred=knn.predict(X_test)

print(y_pred)
#Make Submission

temp=pd.DataFrame(pd.read_csv("/kaggle/input/titanic/test.csv")['PassengerId'])

temp['Survived']=y_pred

temp.to_csv("submission3.csv",index=False)