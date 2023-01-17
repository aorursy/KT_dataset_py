# data processing and analysis

import pandas as pd 

print("pandas version: {}". format(pd.__version__))



# scientific computing

import numpy as np 

print("NumPy version: {}". format(np.__version__))



# scientific and publication-ready visualization

import matplotlib 

print("matplotlib version: {}". format(matplotlib.__version__))



# scientific and publication-ready visualization 

import seaborn as sns

print("seaborn version: {}". format(sns.__version__))



# machine learning algorithms

import sklearn 

print("scikit-learn version: {}". format(sklearn.__version__))



# machine learning algorithms

import statsmodels

print("statsmodels version: {}". format(statsmodels.__version__))



# scientific computing and advance mathematics

import scipy as sp 

print("SciPy version: {}". format(sp.__version__))



from sklearn import tree

from sklearn.metrics import accuracy_score, confusion_matrix
#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
data = pd.read_csv("../input/titanic/train.csv")



# a dataset should be broken into 3 splits: train, test, and (final) validation

# we will split the train set into train and test data in future sections

data_val  = pd.read_csv("../input/titanic/test.csv")



# to play with our data, create copy

data1 = data.copy(deep = True)



# however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data1, data_val]
data1.head()
data_val.head()
data1.shape
data_val.shape
data1.info()
data1.describe()
print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())
# Data description

data.describe(include = 'all')
for dataset in data_cleaner:    

    # age: median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    # embarked: mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    # fare: median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

    # drop Cabin as it has 687 as null out of 891 (approx 77% of data)

    dataset.drop('Cabin', axis=1, inplace=True)
print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())
# List of variables to map



varlist =  ['Sex']



# Defining the map function

def binary_map(x):

    return x.map({'male': 1, "female": 0})



# Applying the function to the housing list

for dataset in data_cleaner:

    dataset[varlist] = dataset[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data1['Embarked'], prefix='Embarked', drop_first=True)

    

# Adding the results to the master dataframe

data1 = pd.concat([data1, dummy1], axis=1)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data_val['Embarked'], prefix='Embarked', drop_first=True)

    

# Adding the results to the master dataframe

data_val = pd.concat([data_val, dummy1], axis=1)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data1['Pclass'], prefix='Pclass', drop_first=True)

    

# Adding the results to the master dataframe

data1 = pd.concat([data1, dummy1], axis=1)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data_val['Pclass'], prefix='Pclass', drop_first=True)

    

# Adding the results to the master dataframe

data_val = pd.concat([data_val, dummy1], axis=1)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data1['Sex'], prefix='Male', drop_first=True)

    

# Adding the results to the master dataframe

data1 = pd.concat([data1, dummy1], axis=1)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data_val['Sex'], prefix='Male', drop_first=True)

    

# Adding the results to the master dataframe

data_val = pd.concat([data_val, dummy1], axis=1)
data1['FamilySize'] = data1['SibSp'] + data1['Parch'] + 1

data1.head(2)
data_val['FamilySize'] = data_val['SibSp'] + data_val['Parch'] + 1

data_val.head(2)
PassengerId = data_val.PassengerId
# Renaming the column 

data1= data1.rename(columns={ 'Male_1' : 'Male'})

data_val= data_val.rename(columns={ 'Male_1' : 'Male'})
drop_column = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Fare', 'Embarked']

data1.drop(drop_column, axis=1, inplace = True)
# Checking for outliers in the continuous variables

cont_col = data1['Age']



# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

cont_col.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Let's see the correlation matrix 

plt.figure(figsize = (10,5))        # Size of the figure

sns.heatmap(data1.corr(),annot = True)

plt.show()
# Plot the scatter plot of the data



sns.pairplot(data1, x_vars=['Age'], y_vars='Survived',size=4, aspect=1, kind='scatter')

plt.show()
#graph distribution of quantitative data

plt.figure(figsize=[4,5])



#plt.subplot(231)

plt.boxplot(x=data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age')
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Age']==0]['Age']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age ')

plt.ylabel('# of Passengers')

plt.legend()
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['SibSp'], data1[data1['Survived']==0]['SibSp']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('SibSp Histogram by Survival')

plt.xlabel('SibSp')

plt.ylabel('# of Passengers')

plt.legend()
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['Parch'], data1[data1['Survived']==0]['Parch']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Parch Histogram by Survival')

plt.xlabel('Parch')

plt.ylabel('# of Passengers')

plt.legend()
plt.figure(figsize=[10,4])



plt.hist(x = [data1[data1['Survived']==1]['Male'], data1[data1['Survived']==0]['Male']], 

         stacked=False, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Sex Histogram by Survival')

plt.xlabel('Sex')

plt.ylabel('# of Passengers')

plt.legend()
# Plot the heatmap of the data to show the correlation



sns.heatmap(data1.corr(), cmap="YlGnBu", annot = True)

plt.show()
from sklearn.model_selection import train_test_split



# Separating target column from other features



target = 'Survived'



y = data1[target]

x = data1.drop(columns = target)



# Train and Test dataset split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)





from sklearn.ensemble import RandomForestClassifier



y = data1["Survived"]



#features = ["Age", "Sex_male", "Embarked_S", "Embarked_Q"]

features = ["Age", "Male","Pclass_2", "Pclass_3","FamilySize"]

X = pd.get_dummies(data1[features])

X_test = pd.get_dummies(x_test[features])
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

print("Score: ",score)
X_val = pd.get_dummies(data_val[features])

predictions_test = model.predict(X_val)

output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions_test})

output.to_csv('my_submission_RFC.csv', index=False)

print("Your submission was successfully saved!")