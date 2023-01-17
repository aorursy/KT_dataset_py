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

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ROCAUC

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()
df.isna().sum()/df.shape[0]*100 #Check proprtion of null value for each column
df = df.dropna()

df.columns
df = df[['Survived', 'Pclass','Sex', 'Age', 'SibSp',

       'Parch',  'Fare', 'Embarked']]

df['Embarked'] = df['Embarked'].apply(str)

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

df['Embarked'] = le.fit_transform(df['Embarked'])

df.head()
X,y = df.loc[:, df.columns != 'Survived'], df.loc[:, df.columns == 'Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train
model = RandomForestClassifier()

visualizer = ROCAUC(model, classes=[0,1])



visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show() 
trees=model.estimators_

feature_imporance_tree_wise = []

for each_tree in trees:

    feature_imporance_tree_wise.append(each_tree.feature_importances_)

feature_imporance_tree_wise = pd.DataFrame(feature_imporance_tree_wise,

                                         columns = [ 'Pclass','Sex', 'Age', 'SibSp',

       'Parch',  'Fare', 'Embarked'] )

feature_imporance_tree_wise.head()
import seaborn as sns

sns.set(style="whitegrid")



ax = sns.barplot(x=feature_imporance_tree_wise.columns, y=feature_imporance_tree_wise.mean(),

                yerr = feature_imporance_tree_wise.std()).set_title("Feature Importance Plot")

                
