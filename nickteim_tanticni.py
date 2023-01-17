# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



#grafs

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

train.info()
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = train

X_test_full = test



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['Survived'], inplace=True)

y = X.Survived              

X.drop(['Survived'], axis=1, inplace=True)

X.drop(['Name'], axis=1, inplace=True)

X.drop(['Ticket'], axis=1, inplace=True)

#X.drop(['Fare'], axis=1, inplace=True)





#complete missing age with median

train['Fare'].fillna(train['Fare'].median(), inplace = True)

X_test_full['Fare'].fillna(X_test_full['Fare'].median(), inplace = True)





X.drop(['Cabin'], axis=1, inplace=True)

#X.drop(['Embarked'], axis=1, inplace=True)

#X.drop(['Age'], axis=1, inplace=True)

   

    

    #complete missing age with median

train['Age'].fillna(train['Age'].median(), inplace = True)

X_test_full['Age'].fillna(X_test_full['Age'].median(), inplace = True)

   

    

    

    

    

    # Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Low cardinality means that the column contains a lot of “repeats” in its data range.

# Examples of categorical variables are race, sex, age group, and educational level. 

# While the latter two variables may also be considered in a numerical manner by using exact values for age 

# and highest grade completed

# nunique() function to find the number of unique values over the column axis. So when it finds over 10 uniqe 

# values and the cname is a 

# dtype 'object' which means Data type objects are useful for creating structured arrays. 

# A structured array is the one which contains different types of data.



### one line meaning of above####

## for cname in a dataframes column shall return a value to 'low_cardinality_cols' if there are more then 10 uniqe values

## and the dtype shall be a object which is a structured array that can have different types of data (lik; int, float string ect.)







# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

### for cname (every value, one at the time) in dataframe for columns return a value to 'numeric_cols' if the 

### dtype= int64 or float64. 



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
train.isnull().sum()
print(train.head())

print('-' * 100)

print(test.head())
y.head()
train['Sex'] = np.where(train['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female

test['Sex'] = np.where(test['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female
train.Sex.head()
train.Sex.isnull().sum()
sns.scatterplot(x=train['Sex'],y=train['Age'])
train.head()


# Color-coded scatter plot w/ regression lines

sns.lmplot(x='Fare',y='Parch', hue='Sex', data=train)
train.describe()
#train.Age.isnull().sum()


#train.Age.isnull().sum()
#train.Survived.head()
print(len(train))

train = train.dropna(subset=['Embarked'])

test = test.dropna(subset=['Embarked'])

print(len(train))
X.head()
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

#from xgboost import XGBRegressor





model2 = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=1)

model = GradientBoostingClassifier(random_state=1)

#model = DecisionTreeClassifier(random_state=1)

#model=SGDClassifier(random_state=1)

#model=ExtraTreesClassifier(random_state=1)

#model = XGBRegressor()







model.fit(X_train, y_train)

predictions = model.predict(X_test)





output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")


print('model accuracy score',model.score(X_valid, y_valid))
model2.fit(X_train, y_train)

y_predictions = model2.predict(X_test)

print('model1 accuracy score',model2.score(X_valid, y_valid))
if len(output) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(output)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
X_train.head()
X_valid.head()
ss=X_valid
ss.set_index('Pclass')
# Heatmap showing average game score by platform and genre

plt.figure(figsize=(20,20))

# Add title

plt.title("Average score")



sns.heatmap(data=X_valid,annot=True)
