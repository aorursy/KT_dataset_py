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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()

train.info()
test.head()

test.info()
#Categorical Variable: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp and Parch

#Numerical Variable: Age, PassengerId and Fare

import matplotlib.pyplot as plt

import seaborn as sns

cols=train.columns

#removing passengerid and cabin are unessential, hence dropping the columns

drop_columns = ['PassengerId','Cabin','Ticket']

train_df = train.drop(drop_columns, axis=1)

train_df.columns
#Plotting countplot for categorical variabes

%matplotlib inline

categorical_variable = ["Survived","Pclass","Sex","SibSp","Parch",'Embarked']

for x in categorical_variable:

    # get feature

    var = train_df[x]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    #visualize

    sns.countplot(x=var,data=train_df,palette="hls")

    plt.title(x)

    plt.show()
#Plotting distribution plot for numerical values

sns.distplot(train_df.Fare)
sns.distplot(train_df.Age)
#checking for NA/Null values

train_df.isnull().sum() #Age and Embarked are having na's
#To fill the na values with mean imputation

import statistics as st

new_train = train_df.fillna({

    'Embarked': st.mode(train_df['Embarked']),

    'Age' : np.mean(train_df['Age'])

}) #mode imputation for embarked(embarked value)



new_train

new_train.isnull().sum() #all zero shows no more null values
# Using corr value for feature engineering

#Heat map for all values

sns.heatmap(new_train.corr(),annot=True)
from sklearn.ensemble import RandomForestClassifier

y = train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)

pred = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")