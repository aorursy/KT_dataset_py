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
titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
#Describe key statistics
titanic_df.describe()
#Calculating missing values of non-numeric columns
#Python treats NaN as not missing
(pd.isnull(titanic_df['Cabin'])).sum()
#Print top 5 rows
titanic_df.head(5)
#Understanding the distribution of age to fill mising values
titanic_df.hist(column="Age",        # Column to plot
              figsize=(8,8),         # Plot size
              color="blue")  
#Filling with median
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
#Idenfifying uniques values of Sex
titanic_df.Sex.unique()
titanic_df.loc[titanic_df['Sex']=='male','Sex'] = 0
titanic_df.loc[titanic_df['Sex']=='female','Sex'] = 1
#Similarly for Embarked
titanic_df.Embarked.unique()
#Finding the most common value to impute NaN
titanic_df.Embarked.value_counts()
#Imputing with S
titanic_df.loc[titanic_df['Embarked'].isnull()==True,'Embarked']='S'
titanic_df.loc[titanic_df['Embarked']=='S','Embarked']=0
titanic_df.loc[titanic_df['Embarked']=='C','Embarked']=1
titanic_df.loc[titanic_df['Embarked']=='Q','Embarked']=2

#Using Linear Regression and Cross Validation 
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
alg = LinearRegression()
kf = KFold(titanic_df.shape[0],n_folds=3,random_state=1)
predictions=[]
for train, test in kf:
    train_predictors = (titanic_df[predictors].iloc[train,:])
    train_target = titanic_df['Survived'].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic_df[predictors].iloc[test,:])
    predictions.append(test_predictions)
                                   


predictions = np.concatenate(predictions,axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions < 0.5] = 0
accuracy = len(titanic_df.loc[titanic_df.Survived == predictions,'Survived'])/len(titanic_df.Survived)
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
alg= LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg,titanic_df[predictors],titanic_df['Survived'],cv=3)
#Repeating the steps on test file

test_df['Age']=test_df['Age'].fillna(titanic_df['Age'].median())
test_df.loc[test_df['Sex']=='male','Sex']=0
test_df.loc[test_df['Sex']=='female','Sex']=1

test_df.loc[test_df['Embarked'].isnull()==True,'Embarked']='S'
test_df.loc[test_df['Embarked']=='S','Embarked']=0
test_df.loc[test_df['Embarked']=='C','Embarked']=1
test_df.loc[test_df['Embarked']=='Q','Embarked']=2

test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())
#Using the entire training data to make predictions
alg = LogisticRegression(random_state=1)
alg.fit(titanic_df[predictors],titanic_df['Survived'])
predictions = alg.predict(test_df[predictors])
submission = pd.DataFrame({
           "PassengerId": test_df["PassengerId"],
        "Survived": predictions
    })
with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):
    print(submission)
