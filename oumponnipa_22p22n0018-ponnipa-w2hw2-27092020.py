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
from collections import Counter

from sklearn import preprocessing

import copy

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import recall_score,precision_score,f1_score
df = pd.read_csv('/kaggle/input/titanic/train.csv')

testDF = pd.read_csv('/kaggle/input/titanic/test.csv')

df.info()
df.head(5)
df = df.drop(['Name', 'Ticket', 'Cabin'], axis='columns')
ctr = Counter(df['Embarked'])

print("Embarked feature most common 2 data points:", ctr.most_common(2))



print("Age feature mean value:", np.mean(df['Age'].dropna()))
df['Embarked'].fillna('S', inplace=True)



df['Age'].fillna(30, inplace=True) # 29.69... does not specify a valid age, round it
encoder = preprocessing.LabelEncoder()



embarkedEncoder = copy.copy(encoder.fit(df['Embarked']))

df['Embarked'] = embarkedEncoder.transform(df['Embarked'])



sexEncoder = copy.copy(encoder.fit(df['Sex']))

df['Sex'] = sexEncoder.transform(df['Sex'])
df.describe()
df.info()
testDF.head(5)
testDF.info()
testDF.drop(['Name', 'Ticket'], axis='columns', inplace=True)
ctr = Counter(testDF['Cabin'])

print(f'Cabin feature most common values:', ctr.most_common(4))



meanAge = np.mean(testDF['Age'])

print(f'Mean age for the age feature:', meanAge)
# drop the Cabin feature and fill perform mean substitution for missing records in the Age feature

testDF.drop('Cabin', axis='columns', inplace=True)

testDF['Age'].fillna(30, inplace=True)
testDF['Embarked'] = embarkedEncoder.transform(testDF['Embarked'])

testDF['Sex'] = sexEncoder.transform(testDF['Sex'])
testDF.info()
testDF['Fare'].fillna(np.mean(testDF['Fare']), inplace=True)

testDF.info()
def writeCSV(predictions):

    outputDF = pd.DataFrame(np.column_stack([testDF['PassengerId'], predictions]), columns=['PassengerId', 'Survived'])

    outputDF.to_csv('./predictions.csv', index=False)
X = df.drop('Survived', axis='columns')

Y = df['Survived']

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.9)
from sklearn.model_selection import KFold

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor



skf = KFold(n_splits=5) 



tree_model = tree.DecisionTreeClassifier()

gnb_model = GaussianNB()

clf = RandomForestClassifier(n_estimators=50,n_jobs=1,max_depth=10)



TRecall = []

NRecall = []

RRecall = []

i = 0



for train_fold, valid_fold in skf.split(df):    

    i = i+1

    print("Fold" , i)

    f_train = df.loc[train_fold] # Extract train data with cv indices

    f_valid = df.loc[valid_fold] # Extract valid data with cv indices

    X = f_train.drop(['Survived'], axis=1)

    y = f_train["Survived"]

    T_model = tree_model.fit(X,y)

    N_model = gnb_model.fit(X,y)

    R_model = clf.fit(X,y)

    t_pred = T_model.predict(testX)

    n_pred = N_model.predict(testX)

    r_pred = R_model.predict(testX)

    Measure = []

    Recall = []

    print("Decision Tree")

    print('Recall: %.3f' %(recall_score(testY,t_pred)))

    print('Precision:%.3f' %(precision_score(testY,t_pred)))

    Measure.append(f1_score(testY,t_pred))

    print('F-Measure:%.3f' %(f1_score(testY,t_pred)))

    print("\t")

    print("Naïve Bayes")

    print('Recall:%.3f' %(recall_score(testY,n_pred)))

    print('Precision: %.3f' % (precision_score(testY,n_pred))) 

    Measure.append(f1_score(testY,n_pred))

    print('F-Measure:%.3f' %(f1_score(testY,n_pred)))

    print("\t")

    print("Neural Network")

    print('Recall:%.3f' %(recall_score(testY,r_pred)))

    print('Precision:%.3f' %(precision_score(testY,r_pred)))

    Measure.append(f1_score(testY,r_pred))

    print('F-Measure:%.3f' %(f1_score(testY,r_pred)))

    print("\t")

    print('Average F-Measure:%.3f' %(sum(Measure) / 3))

    print("--------------------------------------------------")