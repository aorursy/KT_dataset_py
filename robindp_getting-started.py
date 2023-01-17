# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output .
RANDOM_STATE = np.random.RandomState(seed=42)
df = pd.read_csv('/kaggle/input/titanic/train.csv')

print(df.columns)

print(df.head())

print(df.describe())

original_df = df.copy()
#We will drop the Id, Name and Ticket columns first 

df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

df.head()
#check missing data

for column in df.columns:

    print((column,df[column].isna().sum()))

print("Total instances: "+str(len(df)))
#Cabin has to much missing data, we will drop that column

df.drop('Cabin',axis=1,inplace=True)
#Age has 20% missing data, we can drop those rows, or fill them in with an estimate (maybe just the overall average)

#For now we will drop those rows



#First we perform a simple check that we are not dropping a disproportinate set of the survival rate.

print('all')

print(df['Survived'].value_counts(normalize=True))

print('age unknown')

print(df[pd.isnull(df['Age'])]['Survived'].value_counts(normalize=True))
#this is safe to drop, we will however work further with 20% less data samples

df = df[pd.notnull(df['Age'])]

AVERAGE_AGE = df['Age'].mean()
#We also drop the 2 rows with missing embarkment info

df = df[pd.notnull(df['Embarked'])]
#Encode the categorical variables

from sklearn.preprocessing import LabelEncoder

enc_df = pd.DataFrame()

sex_enc = LabelEncoder()

sex_enc.fit(df['Sex'])

print(sex_enc.classes_)

print(sex_enc.transform(sex_enc.classes_))

df['Sex'] = sex_enc.transform(df['Sex'])
df = pd.get_dummies(df, drop_first=True)

df.head()
#Now we prepare the dataset for our prediction models

from sklearn.model_selection import train_test_split

X = df.drop('Survived',axis=1)

y = df['Survived']

# as a last step we will normalize the data of every column to [0,1] with simple MinMax Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)

X = pd.DataFrame(scaler.transform(X),columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=RANDOM_STATE,solver='lbfgs').fit(X_train, y_train)

print('Accuracy of logistic regression: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.svm import LinearSVC

svm = LinearSVC(random_state=RANDOM_STATE).fit(X_train, y_train)

print('Accuracy of support vector machine: {:.2f}'.format(svm.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier



randfor = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE).fit(X_train, y_train)

print('Accuracy of random forest: {:.2f}'.format(randfor.score(X_test, y_test)))
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(6,4,2), random_state=RANDOM_STATE, max_iter=1000).fit(X_train, y_train,)

print('Accuracy of mlp: {:.2f}'.format(mlp.score(X_test, y_test)))
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

def preprocess(dataframe):

    id = dataframe['PassengerId']

    dataframe.drop(['Cabin','PassengerId','Name','Ticket'],axis=1,inplace=True)

    dataframe['Age'] = dataframe['Age'].fillna(AVERAGE_AGE)

    dataframe['Embarked'] = dataframe['Embarked'].fillna('C')

    dataframe['Sex'] = sex_enc.transform(dataframe['Sex'])

    dataframe['Embarked_Q'] = dataframe['Embarked'] == 'Q'

    dataframe['Embarked_C'] = dataframe['Embarked'] == 'C'

    dataframe.drop('Embarked',axis=1,inplace=True)

    dataframe = pd.DataFrame(scaler.transform(dataframe),columns = dataframe.columns)

    dataframe.loc[dataframe['Age']<0.0,'Age']=0.0

    dataframe.fillna(0.0,inplace=True)

    return dataframe, id

df_test, test_id = preprocess(df_test)

df_test.head()
out = pd.DataFrame()

out['PassengerId'] = test_id

out['Survived'] = logreg.predict(df_test)

out.to_csv('submission_logreg.csv',index=False)

out = pd.DataFrame()

out['PassengerId'] = test_id

out['Survived'] = svm.predict(df_test)

out.to_csv('submission_svm.csv',index=False)
out = pd.DataFrame()

out['PassengerId'] = test_id

out['Survived'] = randfor.predict(df_test)

out.to_csv('submission_randfor.csv',index=False)
out = pd.DataFrame()

out['PassengerId'] = test_id

out['Survived'] = mlp.predict(df_test)

out.to_csv('submission_mlp.csv',index=False)