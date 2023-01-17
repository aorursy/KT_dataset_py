# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#reading csv as dataframes

import os

import pandas as pd

from pandas import DataFrame

os.chdir("../input/")

df_tr= DataFrame.from_csv("train.csv",header=0,sep=',',index_col='PassengerId')

df_tr.head()
#checking for a missing value in the dataframe

print(df_tr.count())

#the column age is missing 177 values

#the column Embarked is missing 2 values

#these 2 columns can be used as features

#but the column Cabin has 687 missing values and hence cant be used as a feature

#creating a data frame to fill true or false if data is present or absent correspondingly

a = pd.isnull(df_tr)

a.tail()

em_miss_index = a[a['Embarked']==True].index.values.tolist()

df_tr.loc[em_miss_index]
age_miss_index = a[a['Age']==True].index.values.tolist()

df_tr.loc[age_miss_index]
age_miss=df_tr.iloc[age_miss_index]

live = age_miss[age_miss['Survived']==1]

temp = len(live.index)

print("Alive & no age details %d" % temp)

print("Dead & no age details %d" % (177-temp))
import sklearn

import pandas as pd

import numpy as np

from sklearn import cross_validation,tree

#checking for NaN in age

np.isnan(df_tr['Age'][df_tr['Age'].isnull()==True])
#filling average age in place of NaN

df_tr['Age'].fillna(df_tr['Age'].mean(),inplace=True)

#see if a row has a null

#df_tr.ix[pd.isnull(df_tr).any(1)==True]
#checking for NaN in age

np.isnan(df_tr['Age'][df_tr['Age'].isnull()==True])
#checking for NaN in sex

np.isnan(df_tr['Age'][df_tr['Age'].isnull()==True])
df_tr['Sex'][df_tr['Sex']=='female'] = 1

df_tr['Sex'][df_tr['Sex']=='male'] = 0 

df_tr['Fare'].astype(float)

df_tr['Sex'].astype(int)

df_tr['Age'].astype(float)

df_tr['Pclass'].astype(int)

df_tr.tail()
#checking for NaN in sex

np.isnan(df_tr['Age'][df_tr['Age'].isnull()==True])
features = df_tr.drop(df_tr[['Survived','Name','Ticket','Cabin','SibSp','Parch','Embarked']], axis=1)

label = df_tr['Survived']

features.ix[pd.isnull(df_tr).any(1)==True].tail()

# Dropping all the rows which donot have gender information

#features.drop(features.index[features['Sex'].apply(np.isnan)],inplace=True,axis=0)
X_train,X_test,y_train,y_test = cross_validation.train_test_split(features,label,test_size=0.2)

clf = tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)
## Now apply the model on to test data set

df_te = DataFrame.from_csv("test.csv", sep=',', header=0, index_col='PassengerId')

df_te.head()
#bringing data to same input type features set

print(df_te.count())
np.isnan(df_te['Age'][df_te['Age'].isnull()==True]).count()
#filling age NaN with mean of age values

df_te['Age'].fillna(df_te['Age'].mean(),inplace=True)
np.isnan(df_te['Age'][df_te['Age'].isnull()==True]).count()
# replacing female with 1 and male with 0 and converting to int and float

df_te['Sex'][df_te['Sex']=='female'] = 1

df_te['Sex'][df_te['Sex']=='male'] = 0

df_te['Fare'].fillna(0,inplace=True)

df_te['Fare'].astype(float)

df_te['Pclass'].astype(int)

df_te['Age'].astype(float)

df_te['Sex'].astype(int)

df_te.head()
te_features = df_te.drop(df_te[['Cabin','Name','Ticket','Embarked','SibSp','Parch']],axis=1)

te_features.head()
import numpy as np

te_features['Sex'][te_features['Sex'].isnull()==True]
op_df = DataFrame(clf.predict(te_features), index=df_te.index.values)

op_df.columns=['Survived']

op_df.tail()
te_re = DataFrame.from_csv("gendermodel.csv", sep=',', header = 0, index_col = 'PassengerId')

te_re.head()
# Comparing outputs

df_chk = pd.merge(op_df,te_re)