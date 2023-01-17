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



import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from  sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
titanic_df = pd.read_csv("../input/titanic/train.csv")
list(titanic_df.columns)
sns.stripplot(x="Survived",y="Age",data=titanic_df )
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, titanic_df['Fare'].max()))



facet.add_legend()
sns.countplot("Sex",hue="Survived",data=titanic_df)
sns.factorplot('Pclass','Survived',data=titanic_df)



plt.show()
sns.countplot("SibSp",hue="Survived",data=titanic_df)
sns.factorplot('Parch','Survived',data=titanic_df)



plt.show()
sns.countplot("Embarked",hue="Survived",data=titanic_df)
titanic_df[~titanic_df["Embarked"].isnull()]
cat_cols = ['Pclass','Sex','SibSp',

 'Parch',

 'Ticket','Embarked']
for i in cat_cols:

    titanic_df[i]=titanic_df[i].astype("category")

    titanic_df[i]=titanic_df[i].cat.codes
titanic_df
X_train,X_test=train_test_split(titanic_df,test_size=0.4)
Y_train=X_train["Survived"]



Y_test=X_test["Survived"]
X_train=X_train.drop(["Survived","Name","PassengerId","Cabin","Embarked"],axis=1)





X_test=X_test.drop(["Survived","Name","PassengerId","Cabin","Embarked"],axis=1)
titanic_df['Age'].isnull().values.any()