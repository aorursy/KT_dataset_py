# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing relevant packages



import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import sklearn



from pandas import Series, DataFrame

from sklearn import preprocessing



#Encoders

from sklearn.preprocessing import LabelEncoder #Ordinal Features

from sklearn.preprocessing import OneHotEncoder #Nominal Features



#Modelling and Evaluation

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict



import keras 

from keras.models import Sequential #Initialize

from keras.layers import Dense      #Layers





from sklearn import metrics

#Suppress Warnings 

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score



np.random.seed(100) #results reproducible
#Modeling Part 1 - Logistic Regression



#Assumptions for Logistic Regression include 

#A). (Rule #1): Data is free of missing values - Check missing values, discard or impute to get a comprehensive dataset.

#B). Predictant/Response/Target variable is BINARY OR ORDINAL (ordered values)

#C). (Rule #2): Predictors are indepedent of each other. Correlation leads to overfitting in ML models.

#D). (Rule #3): At least 50 values per variable for reliable results.



#When to USE IT

# - Binary Target Variable

# - Best for clean, well behaved data.



#Modeling Part 2 - Artificial Neural Network



#Activaton Functions to be used (Since it has two classes - Survived or Not Survived)

# -ReLU: Only activates the function is if it above a certain qty. When input is 0, output is 0 but when above a certain threshold,

# it has linear relationship with the dependent variable. Returns all +ve values for +ve values.    





#The general steps to be followed include - 



# General Steps: Data Exploration, Data Wrangling, Modeling, Evaluation.

# -Identify the Predictor and Response Variable (Survived). The target variable should be a Binary Target Variable.

# -Explore Continuous Features (Nulls, Missing, Outliers, Distribution etc).

# -Explore Categorical Features

# -Feature Engineering: Fix any wrangling issues with features. Encode features, combine features etc.

# -Drop/subset relevant variable for modeling. 

# -Split dataset using sklearn's test_train_split

# -Modeling: Choose appropriate parameters, build the model and feed the data.

# -Test prediction showing a passenger will survive or not
#Import the data 

titanic_train = pd.read_csv ('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv ('/kaggle/input/titanic/train.csv')
#Combine Data

titanic = pd.concat([titanic_train,titanic_test], axis = 0, sort = False)
titanic.info()
#Rule 1: There should be no missing values.



#It is evident that there are missing values in Age, Cabin and Embarked.

#Let's inspect the missing data by plotting using seaborn and count the missing values.
titanic.describe()
#Plotting Missing values using SEABORN

sb.heatmap(titanic.isnull(), cbar=True)
titanic.isnull().sum()
#With 354 missing values in the Age variable, 1374 in Cabin and 4 in Embarked, I have decided to cater to each feature individually.



#Age: 1). Mean imputation, 2). Feature Inference using Parch.

#Cabin: Will be dealt with later.

#Embarked: Drop (<5% of values)



#There are TWO OPTIONS for imputing age:

#Option 1: Mean Imputation. 

## titanic['Age'].fillna(titanic['Age'].mean(), inplace = True) 



#Option 2: Impute using other features.

#Impute the age using the Parch variable - The underlying concept is that each parch discrete value - number of parents and children -  will have a corresponding average age in the dataset.

#If the passenger had 0 parents and children (parch==0), their average age in the dataset will be imputed for missing age values for all members with no parents and children.

#To do that, the first thing is to check the average age across the parch variable. A boxolot is great for such discrete, categorical variables.



sb.boxplot(x='Parch',y='Age', data=titanic, palette='hls')



#It can be observed that, as the median age increases, the number of parents/children increase.

#Can impute based on avergae number of children/parent. For instance, a person's unknown age can be calculated 

#using average number of parch (parent/children) they have. Ex: WHen they have 0 kids, they're median age=30.

#If it's 1, they're age can be imputed as 22.
#Checking average age across parch.

parch_groups = titanic.groupby(titanic['Parch']) #Only returns numerical values.

parch_groups.mean()
#Function for Imputing Age using Parch.



def age_cal (col):

    Age = col[0]

    Parch = col[1]

    

    if pd.isnull(Age):

        if Parch == 0:

            return 32

        elif Parch ==1:

            return 24

        elif Parch ==2:

            return 17

        elif Parch ==3:

            return 33

        elif Parch ==4:

            return 45

        else:

            return 30

    

    else:

        return Age

#Applying newly created age_cal function

titanic['Age'] = titanic[['Age','Parch']].apply(age_cal, axis=1)

titanic.isnull().sum()



#No missing values in Age.
# Feature: SibSp, Parch are combined. 
#Since SibSp (Sibling/Spouse) and Parch are family variables. We can think of combining them but let's see if they're similar.

#Catplot is a categorical plot.



for i,col in enumerate (['SibSp','Parch']):

    plt.figure (i)

    sb.catplot (x=col, y='Survived', data = titanic, kind = 'point', aspect =2,)



#Larger bar idicates a smaller sample size (it's the error).

#Categorical Plot shows the more children or parents you have, the less likely you are to survive.
#Since both the plots show very similar characteristics, we can go ahead and combine them.

#Combine the two and drop (later) individual columns

titanic['Family_Count'] = titanic['Parch']+titanic['SibSp']
# Features: Name, PassengerId, Ticket and Multi Collinear Variables (SibSp and Parch) are dropped.



titanic = titanic.drop(['PassengerId','Name','Ticket','SibSp','Parch'], axis=1)

titanic.head()



#Since PassengerId, Name and Ticket are just used for identification, dropping features.

#Drop Parch and SibSp (Will lead to multi collinearity issues).
# Feature: Cabin



#Lets check Survival based on Cabin was missing or not. Grouped by whether cabin was missing or not.

#For those two groups, check survived column and take the mean.67% of people were missing a cabin.

titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean()
#Make a cabin indicator for encoding 0 to missing values and 1 for non missing.

titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(),0,1) #If null was found, put 0 else put 1.

titanic.head()
# Feature: Sex



#Hard coding categorical variables (for 1 being Male and 0 being Female using a Dictionary)



gender_num = {'male':1,'female':0}

titanic['Sex'] = titanic['Sex'].map (gender_num)

titanic.head()
# Once the features are engineered and data is model ready, it's time to fit it in the model.



#Dropping Embarked, Cabin from the titanic dataframe.

titanic.drop(['Embarked','Cabin'], axis=1,inplace=True)

titanic.head()
# #Rule 2 - Check for independence. 

#There should be no correlation between variables.



sb.heatmap(titanic.corr())
#Pclass and Fare show a strong correlation - may lead to overfitting in model.

#Dropping pclass SINCE it has a strong coorelation with Fare.



titanic.drop(['Pclass'], axis=1, inplace=True)

titanic.head()
#Drop the balance Na's and reset INDEX

titanic.dropna(inplace=True)

titanic.reset_index(inplace=True,drop=True)



print (titanic.info())  #Rule3 - Checking data size (>50 in each variable to have good results)
#Let's check the distribution of the features.



print('Distributions of features')

plt.figure(figsize=(26, 24))

for i, col in enumerate(list(titanic.columns)):

    plt.subplot(7, 4, i + 1)

    plt.hist(titanic[col])

    plt.title(col)
#Checking the distribution of survived vs not survived.

titanic['Survived'].value_counts()



#The data is clearly skewed towards not survived. Scaling should take care of this later.
#Split into TRAIN, TEST.



predictors = titanic.drop('Survived', axis = 1)

target = titanic['Survived']



#If you want to set aside data for validation as well, then uncomment the following code. This will split first into 60 (train) -40 and then 20 (test) and 20 (validation)

# #Set aside 40% of the data for test and validation. Keep 60% for training.

# X_train, X_test, y_train, y_test = train_test_split (predictors, target, test_size=.40, random_state = 200)

# #Split the 40% into two: test and validation sets.

# X_val, X_test, y_val, y_test = train_test_split (X_test, y_test, test_size=.50, random_state = 42)



#Splitting the data into train and test (80-20).

X_train, X_test, y_train, y_test = train_test_split (predictors, target, test_size=.20, random_state = 200)



#Applying Scaler.

scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train) #notice how the target feature (y) is untouched.

X_test = scaler.fit_transform(X_test)





print (X_train.shape)

print (y_train.shape)
#Good to see the data once before building the machine learning model.

X_train[0:100:20]
# Function to print the results.

def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
#Building the model



#Lets instantiate the model and use cross validation to find the best params.

LogModel = LogisticRegression()

parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2','l1']} #remember C is the regularization param, key to minimizing loss is optimizing this as one of the key params in log regression.

#smaller value of c is stronger regularization. (it is the inverse of regularization strength)

cv = GridSearchCV(LogModel, parameters, cv=5)

cv.fit(X_train, y_train.ravel()) #fit it on the train data to find best params.



print_results(cv)
#Since the best params identified in this case are 0.789 (+/-0.041) for {'C': 0.1, 'penalty': 'l2'}

#Using that to build the log model. 

LogReg = LogisticRegression(C=.1,penalty='l2')

#Fit it on the train data.

LogReg.fit(X_train,y_train)



#Predict using the model

y_pred = LogReg.predict(X_test)
#MODEL Evaluation can be done using a). Classification Report b).K-fold cross validation/confusion matrices.



##Classification report without cross-validation.

print (classification_report(y_test, y_pred))
#To check if the model is overfitting, let's check the score on train.

y_pred_train = LogReg.predict(X_train)

print (classification_report(y_train, y_pred_train))
#For just the precision score.

# #Lets check the precision score.

# precision_score(y_test, y_pred)



# #Lets check the precision score.

# precision_score(y_train, y_pred_train)
#Now there is not much differnce between the prediction on the train vs prediction on the test, the model hence does not seem to be overfitting.
#K-Fold Cross validation and confusion matrices

y_train_pred = cross_val_predict(LogReg, X_train, y_train, cv=5)

confusion_matrix(y_train, y_train_pred)
#Make a TEST PREDICTION

titanic[860:863] #example passenger
#Create a test passenger and predict the survival.



#Let's make a test passenger with ID:860, Sex: Male, Age:41,Fare:15, Family_Count:2, Cabin_ind:0

test_passenger_array = np.array([1, 41,15, 2, 0]).reshape(1,-1)

#This shows the prediction. (LogReg is a sigmoid function : 0 -1)

print(LogReg.predict(test_passenger_array))

#This gives the probability of survival.

print(LogReg.predict_proba(test_passenger_array))
#The passenger will not have survived (0) as seen above. We can say that with almost 99% confidence. Wow!

#And that is how you predict survival from the titanic disaster.
#This was a mini tutorial covering all the basics of a logistic regression project.



#Next, lets build an Artificial Neural Network model using the same data. 
# Initialising the Neural Network

model = Sequential()



#Layers

#Input Layer

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

#Hidden Layers

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

#Ouput Layer

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



#Binary cross_entropy is for classification loss function.

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#Fit the model

model.fit(X_train, y_train, batch_size = 32, epochs = 200, verbose=2)
model.summary()
#Getting the prediction using Artificial Neural Network.

y_pred = model.predict(X_test)

y_fin = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

#That's all for now folks.