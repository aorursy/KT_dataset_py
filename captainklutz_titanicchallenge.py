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
import pandas as pd

training = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

df=training

training['train_test'] = 1

test['train_test'] = 0

test['Survived'] = np.NaN

all_data = pd.concat([training,test])

all_data.head()
#find numerical and non-numerical data

all_data._get_numeric_data()

num_cols = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']

cat_cols = ['Name','Sex','Ticket','Cabin','Embarked']

#visualise numerical data better

import matplotlib.pyplot as plt

for x in num_cols:

    plt.hist(all_data[x])

    plt.title(x)

    plt.show()
#understand how numerical columns correlate to eachother

import seaborn as sns

df_num = all_data[num_cols]

sns.heatmap(df_num.corr())
#check survival rates along variables

pd.pivot_table(all_data,index='Survived',values=['Age','Pclass','Parch','SibSp'])

pd.pivot_table(all_data,index='Survived',columns='Pclass',values='Ticket',aggfunc='count')

pd.pivot_table(all_data,index='Survived',columns='Name',values='Ticket',aggfunc='count')

pd.pivot_table(all_data,index='Survived',columns='Sex',values='Ticket',aggfunc='count')
#divide the test data

from sklearn.model_selection import train_test_split

y = df['Survived']

X = df.drop('Survived',axis=1)

y.head()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

             

# Imputation
#apply Encoding to splits by creating pipeline

import numpy as np

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_cols),

        ('cat', categorical_transformer, cat_cols)

    ])



rfc = RandomForestClassifier(random_state = 1)

xgbc = XGBClassifier(random_state =1)

svc = SVC(probability = True)

#Cross Validate data through pipeline - 

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import RandomizedSearchCV 





# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', rfc)

                             ])

#my_pipeline.head()

my_pipeline.fit(X_train,y_train)

preds1 = my_pipeline.predict(X_valid)

mean_absolute_error(y_valid,preds1)

test = pd.read_csv('../input/titanic/test.csv')

test['train_test'] = 0

test.head()

preds2 = my_pipeline.predict(test)

print(preds2)



final_data_3 = {'PassengerId': test.PassengerId, 'Survived': preds2}

submission_3 = pd.DataFrame(data=final_data_3)

submission_3.to_csv('sumbission_randomForestClassifiers.csv', index=False)
# Bundle preprocessing and XGboost modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', xgb)

                             ])



scores2 = cross_val_score(my_pipeline,X_train, y_train,

                              cv=5)





print("MAE scores:\n", scores2.mean())
# Bundle preprocessing and SVC modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', svc)

                             ])



scores2 =  cross_val_score(my_pipeline, X_train, y_train,

                              cv=5)



print("MAE scores:\n", scores2.mean())
#build voting classifier

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('rf',rf),('svc',svc),('xgb',xgb)], voting = 'soft') 



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', voting_clf)

                             ])



scores2 = cross_val_score(my_pipeline,X_train, y_train,

                              cv=5)



print("MAE scores:\n", scores2.mean())