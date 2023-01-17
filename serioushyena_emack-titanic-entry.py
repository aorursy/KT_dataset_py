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



test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv("/kaggle/input/titanic-competion-data/train.csv")
test_shape = test.shape

train_shape = train.shape



print('Output is in (row, col)')

print('test.csv: ', test_shape) 

print('train.csv: ', train_shape) 
import matplotlib.pyplot as plt



sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

train_pivot = train.pivot_table(index='Pclass', values='Survived')



train_pivot.plot.bar()

plt.show()
print(train["Age"].describe())
survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



# define the age cutoff for age classification

cut_points = [-1,0,18,100] # the ages are in a list

label_names = ["Missing","Child","Adult"] # the age categories are in a list



train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df

    

cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"] 



train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)   



pivot = train.pivot_table(index="Age_categories", values="Survived")

pivot.plot.bar()
train["Pclass"].value_counts()
train["Pclass"].head(12)
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")



train.head()
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")



train = create_dummies(train,"Sex")

test = create_dummies(test,"Sex")



train = create_dummies(train,"Age_categories")

test = create_dummies(test,"Age_categories")
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior'] # The x variables



lr.fit(train[columns], train['Survived']) # The y or target variable we want to predict
holdout = test # from now on we will refer to this

               # dataframe as the holdout data



from sklearn.model_selection import train_test_split



columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']



all_X = train[columns]

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.2,random_state=0)
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X) # here we make our predictions

accuracy = accuracy_score(test_y, predictions)



print("Our model's accuracy is: ", accuracy)
from sklearn.model_selection import cross_val_score

import numpy as np



lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)



accuracy = np.mean(scores)

print('Scores = :', scores)

print('Accuracy = :', accuracy)
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']

lr = LogisticRegression()

lr.fit(all_X,all_y)

holdout_predictions = lr.predict(holdout[columns])
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



submission.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')

sub.head()