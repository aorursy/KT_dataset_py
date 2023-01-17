#Import Required Libraries to work
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
%matplotlib inline
#Read train data set
df_titanic_train_dataset = pd.read_csv('../input/train.csv')
#Check the train dataset shape
df_titanic_train_dataset.shape
#Visualize first 5 records
df_titanic_train_dataset.head()
#Read test data set
df_titanic_test_dataset = pd.read_csv('../input/test.csv')
#Check the test dataset shape
df_titanic_test_dataset.shape
df_titanic_train_dataset.columns
#Visualize first 5 records
df_titanic_test_dataset.head()
#Check Stats
df_titanic_train_dataset.describe()
#Check missing values in columns and impute - 177 missing values in Age column
df_titanic_train_dataset.Age.isnull().sum()
#Reorder columns and push the survived column to end
df_titanic_train_dataset = df_titanic_train_dataset[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
#Split input features and target label
X = df_titanic_train_dataset.iloc[:,:-1].values
y = df_titanic_train_dataset.iloc[:,11].values
#Check X and y splits
#X
#y

# Age imputer
#==============================================================================
# Handle the missing values, we can see that in dataset there are some missing
# values, we will use strategy to impute mean of column values in these places
#==============================================================================

from sklearn.preprocessing import Imputer
# First create an Imputer
missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 
                               axis = 0)
# Set which columns imputer should perform
missingValueImputer = missingValueImputer.fit(X[:,4:5])
# update values of X with new values
X[:,4:5] = missingValueImputer.transform(X[:,4:5])
#Check if 177 missing values in Age column has been imputed with mean
X[:,4:5]
#Visualize Age distribution with a plot on the train dataset
x_axis = X[:,4:5]
style.use('ggplot')
plt.Figure(figsize=(10,10))
plt.hist(x_axis,bins=10,color='blue')
plt.xlabel('Age bins')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()
#Plot Age against survival and look for any pattern
x_axis = X[:,4:5]
y_axis = y
style.use('ggplot')
plt.Figure(figsize=(10,10))
plt.scatter(x_axis,y_axis,color='blue')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Plot of Age against Survival')
plt.show()

#Below plot tells us that age group approx. between 65 and 79 have never survived the tragedy! 
#Age feature must be considered for model training.
#Sex Feature must be label encoded as it is a categorical feature
#==============================================================================
# Encode the categorial data. So now instead of character values we will have
# corresponding numerical values
#==============================================================================

from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:,3] = X_labelencoder.fit_transform(X[:,3])
X
#Visualize factor plot using seaborn for sex,survived with hue as Pclass(3D) 
#Use original train dataset
# Set up a factorplot
sns.factorplot(x="Sex",y="Survived", hue="Pclass", kind="bar",data=df_titanic_train_dataset)

#Below conditional factor plot illustrate below points
#1. female passengers from 1st and 2nd Pclass survived the tragedy.
#2. Male passengers irrespective of Pclass  - survival rate is lesser than female passengers
#Pclass, Sex are features that must be considered for model training
#Feature Scaling 
#Lets scale Age column for model training

from sklearn.preprocessing import StandardScaler
stdsclr = StandardScaler()
X[:,4:5] = stdsclr.fit_transform(X[:,4:5])
#Select cols = ["Pclass","Sex","Age"] for model training
pclass_col = X[:,1]
sex_col = X[:,3]
age_col = X[:,4]
X = np.column_stack((pclass_col,sex_col,age_col))
X
#Split data from training set to train and test the model - size as 75% train-25% test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/4,random_state=0)

#Check shape
y_test.shape
#==============================================================================
# Fitting the Logistic Regression algorithm to the Training set
#==============================================================================

from sklearn.linear_model import LogisticRegression
logResgressionagent = LogisticRegression()
logResgressionagent.fit (X_train, y_train ) 
#Predict with 25% split test data
predict_survival = logResgressionagent.predict(X_test)
predict_survival
# Use score method to check accuracy of model 
score = logResgressionagent.score(X_test, y_test)
print(score)

#Achieved 78% accuracy - my model predicts with 78% accuracy - needs to evolve
#==============================================================================
# Fitting the Support vector machine algorithm to the Training set
#==============================================================================

from sklearn import svm
svmagent = svm.SVC()
svmagent.fit (X_train, y_train ) 
#Predict with 25% split test data
predict_survival_svm = svmagent.predict(X_test)
predict_survival_svm
# Use score method to check accuracy of model 
score_svm = svmagent.score(X_test, y_test)
print(score_svm)

#Achieved around 80% accuracy - my svm model predicts with 80% accuracy - needs to evolve
#==============================================================================
# Fitting the Naive Bayes algorithm to the Training set
#==============================================================================

from sklearn.naive_bayes import GaussianNB
nbagent = GaussianNB()
nbagent.fit (X_train, y_train ) 
#Predict with 25% split test data
predict_survival_nb = nbagent.predict(X_test)
predict_survival_nb
# Use score method to check accuracy of model 
score_nb = nbagent.score(X_test, y_test)
print(score_nb)

#Achieved around 80% accuracy - my nb model predicts with 78% accuracy - needs to evolve
#View Test set
df_titanic_test_dataset.head()
#Pick Pclass, Sex and Age columns only to feed in to model for prediction
df_titanic_test_dataset  = df_titanic_test_dataset[['Pclass','Sex','Age']]
df_titanic_test_dataset
X1 = df_titanic_test_dataset
X1.head()
#Sex Feature must be label encoded as it is a categorical feature
#==============================================================================
# Encode the categorial data. So now instead of character values we will have
# corresponding numerical values
#=============================================================================
X1.Sex= X_labelencoder.fit_transform(X1.Sex)
X1
# Age imputer
#==============================================================================
# Handle the missing values, we can see that in dataset there are some missing
# values, we will use strategy to impute mean of column values in these places
#==============================================================================
X1.Age =  X1.Age.fillna(X1.Age.mean())
#View test set
X1
#Feature Scaling 
#Lets scale Age column for model training
age_col = X1.iloc[:,:].values
stdsclr.fit(age_col[:,2:3])
age_col[:,2:3] = stdsclr.transform(age_col[:,2:3])
X1.Age = age_col[:,2]
#View Final test set
X1
#Let's Predict test set using svmagent as accuracy of this model is 80%
svmagent.predict(X1)
