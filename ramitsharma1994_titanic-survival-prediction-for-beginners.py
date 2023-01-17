import warnings
warnings.filterwarnings("ignore")
#importing libraries

import numpy as np    # very useful for multi dimensional arrays and for mathematical operations
import pandas as pd   # Used to load our datasets and present it in the form of a dataframes
import matplotlib.pyplot as plt   # Used for data visualization 
import seaborn as sns
import string
import re

#Scikit - learn libraries for preprocessing and data modeling

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer
from sklearn_pandas import CategoricalImputer   #Library outside of sklearn to impute missing values


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
# reading training and test files using pandas

train = pd.read_csv("../input/titanic/train.csv")

test  = pd.read_csv('../input/titanic/test.csv')

sub = pd.read_csv("../input/titanic/gender_submission.csv")

df_train = train.copy()
df_test = test.copy()
# train.head() will show only the top 5 rows in our train dataset 
train.head()
# Similarly, test.head() will show the top 5 rows in our test dataset
test.head()
print("Train dataset shape: ", train.shape) #this command tells us the shape of our dataset, here we have 891 rows and 12 columns
print("Test dataset shape: ", test.shape)   # in our test dataset we have 418 rows and 11 columns
#checking for null values in our train and test data
print("**********************Train Set*************************************")
print((train.isnull().sum()))
print("********************************************************************\n")

print("**********************Test Set**************************************")
print(test.isnull().sum())
print("********************************************************************")
# Let's first visualize how many people survived. Here, I have used seaborn for visualization. 
# What is seaborn? 
# Seaborn is a Python data visualization library based on matplotlib. 
#It provides a high-level interface for drawing attractive and informative statistical graphics.

plt.figure(figsize = (5, 5)) # setting the size of the figure

sns.set(style="darkgrid")    # setting the style of my vidualizations

sns.countplot(x = 'Survived', data = train, palette= ['Red', 'Blue'])    #Creating a barplot with count of survived or not survived

plt.title("Survived (0 vs 1)")   # setting the title of my plot

plt.show()
#Lets visualize the number of survivors bases on their gender. 

plt.figure(figsize = (5,5))

survived_sex = train[train['Survived']==1]['Sex'].value_counts().sort_index(ascending = False)   # series containing the count of Male and Female survived  
not_survived_sex = train[train['Survived']==0]['Sex'].value_counts()       # series containing the count of Male and Female not survived

dataframe1 = pd.DataFrame({'Survived':survived_sex, 'Not-Survived':not_survived_sex})    #Creating a dataframe from the above two series
dataframe1.index = dataframe1.index.str.capitalize()               

dataframe1.plot(kind='bar', stacked= 'True', color = ['Blue', 'Red'], alpha = 0.8)  #plotting a stacked barplot from the above dataframe
plt.ylabel('Count of people')
plt.xticks(rotation= "horizontal")
plt.show()
#plotting the scatterplot of Age vs Fare based on whether the person survived or not. Since, both age and Fare are important factors
#let's see the relationship between the Age and Fare among the survivors of titanic disaster

plt.figure(figsize=(15,8))  # setting the size of the figure

color = {0: 'Red', 1: 'Blue'}    #setting the color for survived and not survived

plt.scatter(train['Age'], train['Fare'], c =train['Survived'].apply(lambda x: color[x]), alpha = 0.8 ) #scatter plot between age and fair

plt.title('Age vs Fare')  #setting the title of our plot
plt.xlabel("Age")         #setting the xlabel
plt.ylabel("Fare")        #setting the ylable

plt.show()
# Lets see the distributions of our Age columns. And as mentioned above majority of people does belong to the age of 20-50. 

plt.figure(figsize = (10, 6))
sns.distplot(train['Age'], bins = 30, color = "blue")
plt.title("")
plt.legend()
plt.show()
# Lets see the distributions of our Age columns. And as mentioned above majority of people does belong to the age of 20-50. 

plt.figure(figsize = (10, 6))
sns.distplot(train['Fare'], bins = 50, color = "blue")
plt.title("")
plt.legend()
plt.show()
train['Fare'].describe()
# lets visualize the Pclass column. Pclass is nothing but a ticket class and it represents the socio-economic status (SES), 1st = Upper, 
# 2nd = Middle, 3rd = Lower

plt.figure(figsize = (5,5))

#we are using catplot here to combine a countplot() and a FacetGrid. This allows grouping within additional categorical variables. 
sns.catplot(x = 'Pclass', data = train, col="Survived", kind = 'count')
plt.show()
# lets visualize the Embarked column. 
plt.figure(figsize = (5,5))

sns.catplot(x = 'Embarked', data = train, col="Survived", kind = 'count')
plt.show()
train.head()
# Lets first convert categorical values to ordinal. And we have to make sure that whatever we are changing to training dataset
# we have to make the same changes in our test dataset


#mapping 0 to male and 1 to female
train['Sex'] = train['Sex'].map({"male": 0, "female": 1})  
test['Sex'] = test["Sex"].map({"male": 0, "female": 1})

#before converting the embarked values to 0, 1 and 2. Lets first impute the missing values.

cg = CategoricalImputer()
train['Embarked'] = cg.fit_transform(train['Embarked'].values.reshape(-1,1))
test['Embarked'] = cg.transform(test['Embarked'].values.reshape(-1,1))

#similarly, mapping 0 to S, 1 to C and 2 to Q
train['Embarked'] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test['Embarked'] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# For other categorical variables we will handle it later
train.head()
# Lets try to impute missing values using KNNImputer for Age, Embarked and Fare. 

knn_train = train.copy()
knn_test =  test.copy()

knn_imputer = KNNImputer(n_neighbors=4)

knn_train.iloc[:, 5] = knn_imputer.fit_transform(knn_train.iloc[:,5].values.reshape(-1,1))
knn_test.iloc[:, 4] = knn_imputer.transform(knn_test.iloc[:,4].values.reshape(-1,1))

knn_imputer = KNNImputer(n_neighbors=4)
knn_imputer.fit(knn_train.iloc[:,9].values.reshape(-1,1))
knn_test.iloc[:,8] = knn_imputer.transform(knn_test.iloc[:,8].values.reshape(-1,1))



# For Cabin we can use the first letter of the cabin. And for missing values in Cabin lets denote a letter N

knn_train['Cabin'] = knn_train['Cabin'].apply(lambda x: str(x)[0].upper())
knn_test['Cabin'] = knn_test['Cabin'].apply(lambda x: str(x)[0].upper())
knn_train.isnull().sum()
knn_test.isnull().sum()
# Since SibSP and Parch means siblings/spouse and parent/children, so we can combine these two columns and create one 

knn_train['Family'] = knn_train['SibSp'] + knn_train['Parch']
knn_test['Family'] = knn_test['SibSp'] + knn_test['Parch']
# Lets also use MinMax Scaler to standardize the values of fare and age. And the reasong I am using 
# minmax scaler is to get the values between 0 and 1, since all the other values in our dataset are also either 0 or 1.

mx = MinMaxScaler()

knn_train['Fare'] = mx.fit_transform(knn_train['Fare'].values.reshape(-1,1))
knn_test['Fare'] = mx.transform(knn_test['Fare'].values.reshape(-1,1))

mx = MinMaxScaler()

knn_train['Age'] = mx.fit_transform(knn_train['Age'].values.reshape(-1,1))
knn_test['Age'] = mx.transform(knn_test['Age'].values.reshape(-1,1))
# For tickets column also, lets take out the text part and for the number part, we will assign some letter to it.

#Function to get the text out of the "Ticket" column
def ticket_text(x): 
    x = str(x)
    
    if(x.isdigit()):  #isdigit() is used to check if there is any digit in the string or not
        return 'X'
    else: 
        x = x.strip(string.digits)    #stripping the number out of the text
        x = x.strip(" ")              # stripping the extra space out
        x = x.translate(str.maketrans("","", string.punctuation ))   # removing punctutaion, since there are lot of same tickets but have punctutaion in it
        x = x.replace(" ","")
        return x
    
knn_train['Ticket'] = knn_train['Ticket'].apply(lambda x: ticket_text(x))   #applying the function to transform the tickets columns
knn_test['Ticket'] = knn_test['Ticket'].apply(lambda x: ticket_text(x))     # transforming the test set
knn_train.head()
knn_test.head()
#now lets drop some columns that we dont want for our modeling
# dropping columns from both training and test set

knn_train = knn_train.drop(['PassengerId','Name', 'SibSp', 'Parch'], axis = 1)
knn_test = knn_test.drop(['PassengerId','Name', 'SibSp','Parch'], axis = 1)


knn_test['Ticket'].value_counts()
knn_train.shape
# now we need to create dummy variables. And for that I will first stack my train and test data. If I dont stack it, then there will be
# different number of columns due to tickets

stacked_data = pd.concat([knn_train, knn_test])
stacked_data.head()
stacked_data = stacked_data.reset_index(drop=True)
# Now lets create dummy variables for the ordinal columns 

stacked_data = pd.get_dummies(stacked_data, columns = ['Pclass', 'Ticket', 'Cabin', 'Embarked'])

# We also need to discard one column from each of the dummy variable created otherwise there will be a correlation problem.

stacked_data = stacked_data.drop(['Pclass_1', 'Ticket_A4', "Cabin_A", "Embarked_0"], axis = 1)

# Now lets get back our original train and test data

knn_train = stacked_data.iloc[:891, :]
knn_test = stacked_data.iloc[891:, :]
# checking the shape of training dataset
knn_train.shape
# checking the shape of test dataset
knn_test.shape
# removing the target variable from my training and test data set 
# this will be used in the train_test_split

X = knn_train.drop(['Survived'], axis =1 )
y = knn_train['Survived']

knn_test = knn_test.drop(['Survived'], axis =1)

#creating training and testing dataset for our model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3)
#1. Lets first start with KNN using gridsearch
#KNN using gridsearch

KNN_model = KNeighborsClassifier()
param_grid = {'n_neighbors': [1,3,5,7,9,11,13,15]}
cv = StratifiedKFold(n_splits=10, random_state=0)
grid = GridSearchCV(KNN_model, param_grid, cv = cv, scoring='accuracy',
                    return_train_score=True)
grid.fit(X_train, y_train)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))
# checking accuracy on our (30%) test set 
y_test_hat = grid.predict(X_test)
accuracy_score(y_test,y_test_hat)
#Initializing model again and this time will use our whole training dataset to learn the model
knn_best = KNeighborsClassifier(n_neighbors=9)
knn_best.fit(X, y)

#use kaggle test set for predictions
y_test_hat_1 = knn_best.predict(knn_test)



#Logistic Regression using gridsearch

lr  = LogisticRegression()
param_grid = {'max_iter': [5000, 10000], 'solver': ['sag', 'saga', 'liblinear'] }

cv = StratifiedKFold(n_splits=10)

grid1 = GridSearchCV(lr, param_grid, cv = cv, scoring='accuracy',
                    return_train_score=True)
grid1.fit(X_train, y_train)

print("Best Parameter: {}".format(grid1.best_params_))
print("Best Cross Vlidation Score: {}".format(grid1.best_score_))
# checking accuracy on our (30%) test set 
y_test_hat = grid1.predict(X_test)
accuracy_score(y_test,y_test_hat)
#Initializing model again and this time will use our whole training dataset to learn the model
lr_best = LogisticRegression(max_iter = 5000, solver = "sag")
lr_best.fit(X, y)

#use kaggle test set for predictions
y_test_hat_2 = lr_best.predict(knn_test)



#Random Forest

random_forest_model  = RandomForestClassifier()
param_grid = {'n_estimators': [100,200,300,500]}

cv = StratifiedKFold(n_splits=10, shuffle=True)

grid2 = GridSearchCV(random_forest_model, param_grid, cv = cv, scoring='accuracy',
                    return_train_score=True)
grid2.fit(X_train, y_train)

print("Best Parameter: {}".format(grid2.best_params_))
print("Best Cross Vlidation Score: {}".format(grid2.best_score_))
# checking accuracy on our (30%) test set
y_test_hat = grid2.predict(X_test)
accuracy_score(y_test,y_test_hat)
#Initializing best model  and this time will use our whole training dataset to learn the model

random_forest_best = RandomForestClassifier(max_depth = 6)
random_forest_best.fit(X, y)

#use kaggle test set for predictions
y_test_hat_3 = random_forest_best.predict(knn_test)


#creating dataframe for submission
random_forest_dataframe=pd.DataFrame(y_test_hat_3,columns=['Survived'])

#Gradient Boosting Classifier


grbt_model  = GradientBoostingClassifier()

param_grid = {'n_estimators': [100,200,300,500]}

cv = StratifiedKFold(n_splits=10, shuffle=True)

grid2 = GridSearchCV(grbt_model, param_grid, cv = cv, scoring='accuracy',
                    return_train_score=True)
grid2.fit(X_train, y_train)

print("Best Parameter: {}".format(grid2.best_params_))
print("Best Cross Vlidation Score: {}".format(grid2.best_score_))
# checking accuracy on our (30%) test set
y_test_hat = grid2.predict(X_test)
accuracy_score(y_test,y_test_hat)
#Initializing best model  and this time will use our whole training dataset to learn the model

grbt_best = GradientBoostingClassifier(n_estimators = 300)
grbt_best.fit(X, y)

#use kaggle test set for predictions
y_test_hat_4 = grbt_best.predict(knn_test)


#creating dataframe for submission
grbt_dataframe=pd.DataFrame(y_test_hat_4,columns=['Survived'])
#neaural net using gridsearch

mlp_model  = MLPClassifier()
param_grid = {'hidden_layer_sizes': [100,[2,100],[3,150]], 'activation':['logistic', 'tanh'], 'learning_rate':['constant','adaptive']}

cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

grid = GridSearchCV(mlp_model, param_grid, cv = cv, scoring='accuracy',
                    return_train_score=True)
grid.fit(X_train, y_train)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))
# checking accuracy on our (30%) test set
y_test_hat = grid.predict(X_test)
accuracy_score(y_test,y_test_hat)
#Initializing best model  and this time will use our whole training dataset to learn the model

mlp_best = MLPClassifier(activation='tanh', hidden_layer_sizes=100, learning_rate = 'adaptive')
mlp_best.fit(X, y)

#use kaggle test set for predictions
y_test_hat_2 = mlp_best.predict(knn_test)

#creating dataframe for submission
mlp_dataframe=pd.DataFrame(y_test_hat_2,columns=['Survived'])


#neaural net using gridsearch

xgboost_model  = XGBClassifier()

xgboost_model.fit(X_train, y_train)

cv = KFold(n_splits=10, shuffle=True)

kf_cv_scores = cross_val_score(xgboost_model, X_train, y_train, cv=cv )

print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# checking accuracy on our (30%) test set
y_test_hat = grid.predict(X_test)
accuracy_score(y_test,y_test_hat)

#Initializing best model  and this time will use our whole training dataset to learn the model

xgb_best = XGBClassifier()
xgb_best.fit(X, y)

#use kaggle test set for predictions
y_test_hat_2 = xgb_best.predict(knn_test)

#creating dataframe for submission
xgb_dataframe=pd.DataFrame(y_test_hat_2,columns=['Survived'])


X = df_train.drop(['Survived'], axis = 1)
y = df_train['Survived']

knn_train.head()
! pip install pycaret --user
from pycaret.classification import *
classifier1 = setup(data = knn_train,target = 'Survived', session_id = 345)
compare_models()
tuned_model = tune_model('lightgbm')
knn_train.head()
knn_test
y_pred = predict_model(tuned_model, data=knn_test)
sub['Survived'] = round(y_pred['Score']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head()
## To be Continued.....