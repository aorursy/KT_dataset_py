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
!pip install feature_engine
import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.info()
test.info()
sns.distplot(train['Age'])
sns.distplot(train['Fare'])
train.head(20)
def column_to_categorical(df_raw, columns):

    df = df_raw.copy()

    

    for column in columns:

        df[column] = df[column].astype('O')

    

    return df



def prepreprocess(df_raw, train = True):

    df = df_raw.copy()

    

    df=df[~pd.isna(df['Embarked'])]

    

    df['Cabin'].fillna('N', inplace=True)

    df['Cabin_Letter'] = df['Cabin'].str[0]

    df['Cabin_Num'] = df['Cabin'].apply(lambda x: x.split(' ')[0][1:])

    df['Cabin_Num'] = df['Cabin_Num'].apply(lambda x : 0 if x == '' else int(x))

    df['Cabin_Num'].fillna(0, inplace=True)

    df['Sex_male'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    df['Sex_female'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)

    

    columns = ['Cabin_Letter', 'Cabin_Num', 'Embarked', 'Pclass']

    

    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Sex'], axis = 1)

    

    df = column_to_categorical(df, columns)

    

    if train:

        X = df.drop(['Survived'], axis = 1)

        y = df[['Survived']]

    else:

        X = df

        y = None



    return X, y
X_train_simple, y_train = prepreprocess(train)

X_test_simple, _ = prepreprocess(test, False)
X_train_simple.info()
X_test_simple.info()
y_id = y_test['PassengerId']

y_test = y_test.drop(['PassengerId'], axis = 1)
from feature_engine import categorical_encoders as ce, missing_data_imputers as mdi
replacer = mdi.RandomSampleImputer()
replacer.fit(X_train_simple)
X_train_simple = replacer.transform(X_train_simple)

X_test_simple = replacer.transform(X_test_simple)
sns.distplot(X_train_simple['Age'])
X_train_simple.info()
encoder = ce.CountFrequencyCategoricalEncoder(encoding_method='frequency',

                                            variables=['Cabin_Letter','Cabin_Num', 'Embarked', 'Pclass'])
encoder.fit(X_train_simple)
X_train = encoder.transform(X_train_simple)

X_test = encoder.transform(X_test_simple)
X_train.fillna(0, inplace = True)

X_test.fillna(0, inplace = True)
#X_train_simple = column_to_categorical(X_train_simple, [])

#X_test_simple = column_to_categorical(X_test_simple, ['Embarked', 'Pclass'])
#encoder2 = ce.OneHotCategoricalEncoder(variables=[ 'Embarked', 'Pclass'])
#encoder2.fit(X_train_simple)
#X_train = encoder2.transform(X_train_simple)

#X_test = encoder2.transform(X_test_simple)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.svm import LinearSVC, SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
clfs = [('random_forest', RandomForestClassifier()), 

        ('ada_boost', AdaBoostClassifier()), 

        ('gaussian_nb', GaussianNB()), 

        ('svc', SVC()), 

        ('decision_tree', DecisionTreeClassifier()), 

        ('logistic_reg', LogisticRegression()), 

        ('knn', KNeighborsClassifier())]



best_clf = None

best_clf_acc = 0.0



for clf in clfs:

    print(clf[0])

    

    #clf[1].fit(X_train, y_train['Survived'])

    #y_pred = clf[1].predict(X_test)

    acc_array = cross_val_score(clf[1], X_train, y_train['Survived'], cv=10)

    #acc = accuracy_score(y_test, y_pred)

    #f1 = f1_score(y_test, y_pred)

    acc = np.mean(acc_array)

    acc_std = np.std(acc_array)

    

    if acc > best_clf_acc:

        best_clf_acc = acc

        best_clf = clf[1]

        best_clf_name = clf[0]

    

    print('Acuracia: '+str(acc))

    print('Acuracia desvio: '+str(acc_std))

    #print('F1-score: '+str(f1))

    #print(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
best_clf_name
# if best_clf_name == 'svc':

#     parameters = {'kernel':['linear', 'rbf'], 

#                   'C':[1, 2, 5, 10],

#                   'gamma': ['scale', 'auto']

#                  }

#     clf = GridSearchCV(best_clf, parameters, cv=10)

# elif best_clf_name == 'random_forest':

#     parameters = {'n_estimators': [10,50, 100, 150],

#                   'criterion': [ 'gini', 'entropy'],

#                   'min_samples_leaf': [1, 2, 5, 10]

#                  }

#     clf = GridSearchCV(best_clf, parameters, cv=10)

# else:

clf = best_clf
clf.fit(X_train, y_train['Survived'])

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test['Survived'], y_pred)

f1 = f1_score(y_test['Survived'], y_pred)
print('Acuracia: '+str(acc))

print('F1-score: '+str(f1))

print(confusion_matrix(y_test[['Survived']], y_pred))
submission = pd.DataFrame()

submission['PassengerId'] = y_id

submission['Survived'] = y_pred
submission.to_csv(r'Submission.csv', index = False, header = True)