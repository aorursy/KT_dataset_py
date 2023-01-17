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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
drop_column = ['PassengerId','Cabin', 'Ticket']

df.drop(drop_column, axis=1, inplace = True)
df['Age'].fillna(df['Age'].median(), inplace = True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

df['Fare'].fillna(df['Fare'].median(), inplace = True)

df
from sklearn.preprocessing import LabelEncoder



label = LabelEncoder()



df['Sex_Code'] = label.fit_transform(df['Sex'])

df['Embarked_Code'] = label.fit_transform(df['Embarked'])
from sklearn.model_selection import train_test_split

train_feature = ['Sex_Code','Pclass', 'Embarked_Code', 'SibSp', 'Parch', 'Age', 'Fare']
from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import recall_score, precision_score, f1_score



import warnings

warnings.simplefilter("ignore")

def train(model_fn):

    for fold in range(5):

        clf = model_fn()



        df_train = df[df.index % 5 != fold]

        df_test = df[df.index % 5 == fold]

        x_train = df_train[train_feature]

        y_train = df_train['Survived']

        clf.fit(x_train, y_train)



        df_test_0 = df_test[df_test['Survived']==0]

        x_test_0 = df_test_0[train_feature]

        y_test_0 = df_test_0['Survived'].to_numpy()

        predict_0 = clf.predict(x_test_0)



        df_test_1 = df_test[df_test['Survived']==1]

        x_test_1 = df_test_1[train_feature]

        y_test_1 = df_test_1['Survived'].to_numpy()

        predict_1 = clf.predict(x_test_1)

        print(f'Fold: {fold}')



        print(f'Survived recall {recall_score(y_pred=predict_1, y_true=y_test_1)}')

        print(f'Not survived recall {recall_score(y_pred=predict_0, y_true=y_test_0, pos_label=0)}')



        print(f'Survived precision {precision_score(y_pred=predict_1, y_true=y_test_1)}')

        print(f'Not survived precision {precision_score(y_pred=predict_0, y_true=y_test_0, pos_label=0)}')



        print(f'Survived f1 {f1_score(y_pred=predict_1, y_true=y_test_1)}')

        print(f'Not survived f1 {f1_score(y_pred=predict_0, y_true=y_test_0, pos_label=0)}')



        print(f'Average F1 {f1_score(y_pred=np.concatenate([predict_1, predict_0]), y_true=np.concatenate([y_test_1, y_test_0]))}')

        print()

print('Decision Tree')

train(DecisionTreeClassifier)
print('Naïve Bayes')

train(GaussianNB)
print('Neural Network')

train(MLPClassifier)