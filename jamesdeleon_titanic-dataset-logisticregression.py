# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Addditional Classes



# What would happen if all variables except for 'male', 'Q', and 'S' were standardized?

# 'male', 'Q', and 'S' are binary values and don't need to be standardized.

# code derived from http://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html



import numpy as np



class StandardScaler():

    def __init__(self):

        pass



    def fit(self, X):

        self.mean_ = np.mean(X, axis=0)

        self.scale_ = np.std(X - self.mean_, axis=0)

        return self



    def transform(self, X):

        return (X - self.mean_) / self.scale_



    def fit_transform(self, X):

        return self.fit(X).transform(X)
# EXPLORATORY DATA ANALYSIS
titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_train.head()
titanic_train.info()

# There seems to be missing values from the 'Age' and 'Cabin' columns.

# 'Age' Values could be compensated for by adding in placeholder values

# There is too much missing from the 'Cabin' column to add any sort of pattern-based placeholder.
sns.countplot(x = 'Survived', data = titanic_train, hue = 'Sex')

# An overwhelming amount of males did not survive
sns.countplot(x = 'Survived', data = titanic_train, hue = 'Pclass')

# An overwhelming amount of the 3rd class passengers did not survive
sns.boxplot(x = 'Pclass', y = 'Age', data = titanic_train, hue = 'Sex')

# First class passengers were generally older than second and third class passengers

# The third class passengers were generally the youngest demographic.
sns.countplot(x = 'SibSp', data = titanic_train)

# Most passengers were single or most likely were travelling with their spouse
sns.heatmap(titanic_train.isnull(), yticklabels = False, cbar=False, cmap='viridis')
# DATA CLEANING
# Feature Engineering with names...

# Split by title (Mr., Mrs., Dr., etc...)

print(titanic_train['Name'][0])

name_arr = titanic_train['Name'][0].split()

print(name_arr[1])





# The second index is the title of the person's name.
titanic_train['Name']
# This function will extract the title from the person's name and replace the name entirely with the title



def input_name(column):

    arr = column.split()

    return arr[1]



# titanic_train['Name'] = titanic_train[['Name']].apply(input_name, axis = 1)



titanic_train['Title'] = titanic_train['Name'].apply(lambda x: x.split(',')[1].split('.')[0])

titanic_test['Title'] = titanic_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
titanic_train.head()
titanic_train['Title'].value_counts()
titanic_test['Title'].value_counts()
# Assign numbers to each title



def assign_title(col):

    x = [" Mr", " Miss", " Mrs", " Master", " Dr", " Rev", " Mlle", " Col", " Major", " Don", " Jonkheer", " the Countess", " Lady", " Sir", " Ms", " Mme", " Capt", " Dona"]

    

    if col == x[0]:

        return 1

    elif col == x[1]:

        return 2

    elif col == x[2]:

        return 3

    elif col == x[3]:

        return 4

    elif col == x[4]:

        return 5

    elif col == x[5]:

        return 6

    elif col == x[6]:

        return 7

    elif col == x[7]:

        return 8

    elif col == x[8]:

        return 9

    elif col == x[9]:

        return 10

    elif col == x[10]:

        return 11

    elif col == x[11]:

        return 12

    elif col == x[12]:

        return 13

    elif col == x[13]:

        return 14

    elif col == x[14]:

        return 15

    elif col == x[15]:

        return 16

    elif col == x[16]:

        return 17

    elif col == x[17]:

        return 18

    else:

        return 0
titanic_train['Name_Title_Num'] = titanic_train['Title'].apply(assign_title)

titanic_test['Name_Title_Num'] = titanic_test['Title'].apply(assign_title)
titanic_train.head(n = 10)
# Drop the extraneous info



titanic_train.drop('Name', axis = 1, inplace = True)

titanic_train.drop('Title', axis = 1, inplace = True)



titanic_test.drop('Name', axis = 1, inplace = True)

titanic_test.drop('Title', axis = 1, inplace = True)
sns.heatmap(titanic_train.isnull())
# Looking back at the boxplot, ages can be inferred based on the class of the passenger

# a function will be created to assign the mean class age to a passenger of a specific class



def input_age(columns):

    age = columns[0]

    pclass = columns[1]

    sex = columns[2]

    

    if pd.isnull(age):

        if pclass == 1 and sex == 'male':

            return 40

        elif pclass == 1 and sex == 'female':

            return 35

        elif pclass == 2 and sex == 'male':

            return 30

        elif pclass == 2 and sex == 'female':

            return 28

        elif pclass == 3 and sex == 'male':

            return 25

        elif pclass == 3 and sex == 'female':

            return 21

        else:

            return 30 #mean of all separated ages

    else:

        return age
titanic_train['Age'] = titanic_train[['Age', 'Pclass', 'Sex']].apply(input_age, axis = 1)



titanic_test['Age'] = titanic_test[['Age', 'Pclass', 'Sex']].apply(input_age, axis = 1)
titanic_train.info()
titanic_test.info()
# Remove the rest of the minor missing values ('Embarked')

#titanic_train.dropna(axis = 0, inplace = True)

#titanic_test.dropna(axis = 0, inplace = True)
print(titanic_train.info())

# all the data entries and columns have values associated to them

# A ML_model could have been created to predict the ages of the passengers with an unknown name, 

# maybe that can be revisited.



print(titanic_test.info())
# since 'sex' is a binary column, each value can be represented as such

# convert the 'sex' col into a binary indicator column



sex_train = pd.get_dummies(titanic_train['Sex'], drop_first = True) 

# drop_first = True to take away redundant info

# The first column was a perfect predicotr of the second column



# The same could be done to the 'embark' col

embark_train = pd.get_dummies(titanic_train['Embarked'], drop_first = True)

# Removing one column could remove the 'perfect predictor' aspect
sex_test = pd.get_dummies(titanic_test['Sex'], drop_first = True) 

embark_test = pd.get_dummies(titanic_test['Embarked'], drop_first = True)
# Combine the indicator columns with the original dataset and then remove the original columns that were adjusted

titanic_train = pd.concat([titanic_train, sex_train, embark_train], axis = 1)

titanic_train.head()
titanic_train.drop(['Sex', 'Embarked', 'Ticket'], axis = 1, inplace = True)



titanic_train.head()
titanic_test = pd.concat([titanic_test, sex_test, embark_test], axis = 1)

titanic_test.head()
titanic_test.drop(['Sex', 'Embarked', 'Ticket'], axis = 1, inplace = True)



titanic_test.head()
# ML Models
# Cabin Model
# Feature Engineering with 'Cabin' numbers

# A separate ML model will be utilized for Cabin numbers

# A KNN algorithm will predict the cabin deck... (A, B, C, ..., G)
def cabin_extract(col):

    if pd.notnull(col):

        x = col.split(" ")

        return list(x[0])[0]

    else:

        pass



titanic_train['Cabin_Letter'] = titanic_train['Cabin'].apply(cabin_extract)

titanic_test['Cabin_Letter'] = titanic_test['Cabin'].apply(cabin_extract)
def cabin_num(col):

    if pd.notnull(col):

        if col == 'A':

            return 1

        elif col == 'B':

            return 2

        elif col == 'C':

            return 3

        elif col == 'D':

            return 4

        elif col == 'E':

            return 5

        elif col == 'F':

            return 6

        elif col == 'G':

            return 7

        else:

            return 0

    else:

        pass



titanic_train['Cabin_Number'] = titanic_train['Cabin_Letter'].apply(cabin_num)

titanic_test['Cabin_Number'] = titanic_test['Cabin_Letter'].apply(cabin_num)
titanic_train.head(n = 30)
titanic_test.head(n = 30)
# Remove Extraneous columns

titanic_train.drop('Cabin', axis = 1, inplace = True)

titanic_train.drop('Cabin_Letter', axis = 1, inplace = True)



titanic_test.drop('Cabin', axis = 1, inplace = True)

titanic_test.drop('Cabin_Letter', axis = 1, inplace = True)
titanic_train.head()
X_train_cabin = titanic_train[titanic_train['Cabin_Number'].notnull()].drop(['Survived', 'Cabin_Number'], axis = 1)

y_train_cabin = titanic_train[titanic_train['Cabin_Number'].notnull()]['Cabin_Number']

X_test1_cabin = titanic_train[titanic_train['Cabin_Number'].isnull()].drop(['Survived', 'Cabin_Number'], axis = 1)

X_test2_cabin = titanic_test[titanic_test['Cabin_Number'].isnull()].drop('Cabin_Number', axis = 1)
import numpy as np

from collections import Counter

import math as m



class K_Nearest_Neighbors:

    

    def __init__(self, k):

        self.k = k

        

    def fit(self, X, y):

        # no training process, just a storage process

        self.X_train = np.array(X)

        self.y_train = np.array(y)

    

    def predict(self, X):

        pred_labels = [self._predict(x) for x in X]

        return np.array(pred_labels)

        

    def _predict(self, x):

        # Compute distances for all samples

        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        

        # Focus on k nearest neighbors (sort) and grab labels

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        

        # majority vote -> classify label

        classify = Counter(k_nearest_labels).most_common(1)

        return classify[0][0]

    

    def _euclidean_distance(self, x1, x2):

        # return np.sqrt(np.sum((x1 - x2) ** 2))

        # return np.linalg.norm(x1 - x2)

        return m.sqrt(m.fsum((x1 - x2) ** 2))
test = K_Nearest_Neighbors(k = 5)

test.fit(X_train_cabin, y_train_cabin)

pred1 = test.predict(np.array(X_test1_cabin))

pred2 = test.predict(np.array(X_test2_cabin))
print(pred1)

print(pred2)



# Find an error rate by isolating training data to only titanic_train non_null values...
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train_cabin, y_train_cabin, test_size = 0.03)#, random_state = 2)
test1 = K_Nearest_Neighbors(k = 40)

test1.fit(X_train, y_train)

pred = test.predict(np.array(X_test))
#%matplotlib inline

#import matplotlib.pyplot as plt



for i in range(1, 200):

    knn = K_Nearest_Neighbors(k = i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(np.array(X_test))



    error = np.mean(pred_i != y_test) # avg error rate



    print("k value: " + str(i) + ', Error: ' + str(error))



#plt.figure(figsize = (10,6))

#plt.plot(range(1,200), error_rate, color = 'black', linestyle = 'dashed', marker = 'o', 

#        markerfacecolor = 'red', markersize = 10)

#plt.title('Error Rate vs. K Value')

#plt.xlabel('K value')

#plt.ylabel('Error')



#plt.show()

    

# Settle with a k-value of 20 because with all test allocations that is 

# where the error is most likely to decrease the most
for i in range(0, pred1.size):

    titanic_train['Cabin_Number'].fillna(pred1[i], inplace = True)
#for i in range(0, pred2.size):

#    titanic_test['Cabin_Number'].fillna(pred2[i], inplace = True)
# Survived Model
X_train = titanic_train.drop('Survived', axis = 1)

y_train = titanic_train['Survived']



X_test = titanic_test
# Standardize Values to speed up the algorithm

# Which values to standardize?

X_train.head()
# scale X_train and X_test



scale = StandardScaler()

scaled_xtrain = scale.fit_transform(X_train.drop(['male', 'S', 'Q'], axis = 1))

scaled_xtest = scale.fit_transform(X_test.drop(['male', 'S', 'Q'], axis = 1))



scaled_xtrain.head()
# replace original values with scaled train and test variables



X_train_scaled = pd.concat([scaled_xtrain, X_train['male'], X_train['S'], X_train['Q']], axis = 1)

X_test_scaled = pd.concat([scaled_xtest, X_test['male'], X_test['S'], X_test['Q']], axis = 1)



X_train_scaled.head()
# A simple model and multivariant will be combined into one model

# if show_progress and (num_features == 1): show temp_plot



#cost_function

# -1 * np.sum((y * np.log10(p)) + ((1 - y) * np.log10(1 - p)))



import numpy as np

import matplotlib.pyplot as plt



class LogisticRegression:

    

    def __init__(self, X, y, rate, show_progress):

        self.X_train = X

        self.y_train = y

        self.learning_rate = rate

        self.show_progress = show_progress

        

        self.num_samples, self.num_features = self.X_train.shape

        

        self.weights = np.zeros(self.num_features)

        self.bias = 0

        

        self.cost_history = []

        self.iter_history = []

        

    

    def _sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    

    

    def _predict(self, X_input):

        linear_model = np.dot(X_input, self.weights) + self.bias

        y_predict = self._sigmoid(linear_model)

        return y_predict

    

    

    def cost_function(self, X, y):

        predictions = self._predict(self.X_train)

        cost = -1 * np.sum(((y * np.log(predictions)) + ((1 - y) * np.log(1 - predictions))))

        return cost

        

    

    def update_weights(self):

        y_hat = self._predict(self.X_train)

        

        dw = (1 / self.num_samples) * np.dot(self.X_train.T, (y_hat - self.y_train))

        db = (1 / self.num_samples) * np.sum(y_hat - self.y_train)

            

        self.weights -= self.learning_rate * dw

        self.bias -= self.learning_rate * db

        

        return self.weights, self.bias

    

    

    def train(self, iterations):

        for i in range(0, iterations):

            weight, bias = self.update_weights()

            

            cost = self.cost_function(self.X_train, self.y_train)

            self.cost_history.append(cost)

            self.iter_history.append(i)

            

            if (self.show_progress == True) and (i % 1000 == 0):

                print('Iter: ' + str(i) + ', Error: ' + str(cost))

            

        return weight, bias, cost

    

    

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        y_pred = self._sigmoid(linear_model)

        y_classify = [1 if i >= 0.5 else 0 for i in y_pred]

        return np.array(y_classify)

    

    

    def error_plot(self):

        plt.title('Cost Function Visual')

        plt.xlabel('Iterations')

        plt.ylabel('Error')

        plt.plot(self.iter_history, self.cost_history, color = 'black')

        plt.show()
test = LogisticRegression(X = X_train_scaled, y = y_train, rate = 0.1, show_progress = True)
test.train(iterations = 30000)
test.error_plot()
y_predict = test.predict(X_test_scaled)
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": y_predict

    })



submission.to_csv('submission.csv', index = False)