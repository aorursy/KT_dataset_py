# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics
# Importing the dataset

data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# Printing the 1st 5 columns

data.head()
# Printing the dimenions of data

data.shape
# Viewing the column heading

data.columns
# Inspecting the target variable

data.Attrition.value_counts()
data.dtypes
# Identifying the unique number of values in the dataset

data.nunique()
# Checking if any NULL values are present in the dataset

data.isnull().sum()
# See rows with missing values

data[data.isnull().any(axis=1)]
# Viewing the data statistics

data.describe()
# Here the value for columns, Over18, StandardHours and EmployeeCount are same for all rows, we can eliminate these columns

data.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'],axis=1, inplace=True)
# Plotting a boxplot to study the distribution of features

fig,ax = plt.subplots(1,3, figsize=(20,5))               

plt.suptitle("Distribution of various factors", fontsize=20)

sns.boxplot(data['DailyRate'], ax = ax[0]) 

sns.boxplot(data['MonthlyIncome'], ax = ax[1]) 

sns.boxplot(data['DistanceFromHome'], ax = ax[2])  

plt.show()
# Finding out the correlation between the features

corr = data.corr()

corr.shape
# Plotting the heatmap of correlation between features

plt.figure(figsize=(20,20))

sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# Check for multicollinearity using correlation plot

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data[['DailyRate','HourlyRate','MonthlyIncome','MonthlyRate']].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Plotting countplots for the categorical variables

fig,ax = plt.subplots(2,3, figsize=(20,20))            

plt.suptitle("Distribution of various factors", fontsize=20)

sns.countplot(data['Attrition'], ax = ax[0,0]) 

sns.countplot(data['BusinessTravel'], ax = ax[0,1]) 

sns.countplot(data['Department'], ax = ax[0,2]) 

sns.countplot(data['EducationField'], ax = ax[1,0])

sns.countplot(data['Gender'], ax = ax[1,1])  

sns.countplot(data['OverTime'], ax = ax[1,2]) 

plt.xticks(rotation=20)

plt.subplots_adjust(bottom=0.4)

plt.show()
# Combine levels in a categorical variable by seeing their distribution

JobRoleCrossTab = pd.crosstab(data['JobRole'], data['Attrition'], margins=True)

JobRoleCrossTab
JobRoleCrossTab.div(JobRoleCrossTab["All"], axis=0)
# Combining job roles with high similarities together

data['JobRole'].replace(['Human Resources','Laboratory Technician'],value= 'HR-LT',inplace = True)

data['JobRole'].replace(['Research Scientist','Sales Executive'],value= 'RS-SE',inplace = True)

data['JobRole'].replace(['Healthcare Representative','Manufacturing Director'],value= 'HE-MD',inplace = True)
# Encoding Yes / No values in Attrition column to 1 / 0

data.Attrition.replace(["Yes","No"],[1,0],inplace=True)

data.head()
# One hot encoding for categorical variables

final_data = pd.get_dummies(data)

final_data.head().T
final_data.shape
# Spliting target variable and independent variables

X = final_data.drop(['Attrition'], axis = 1)

y = final_data['Attrition']
# Splitting the data into training set and testset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0, stratify=y)
y_train.value_counts()
# Checking distribtution of Target varaible in training set

y_train.value_counts()[1]/(y_train.value_counts()[0]+y_train.value_counts()[1])*100
y_test.value_counts()
# Checking distribtution of Target varaible in test set

y_test.value_counts()[1]/(y_test.value_counts()[0]+y_test.value_counts()[1])*100
# Logistic Regression



# Import library for LogisticRegression

from sklearn.linear_model import LogisticRegression



# Create a Logistic regression classifier

logreg = LogisticRegression()



# Train the model using the training sets 

logreg.fit(X_train, y_train)
# Prediction on test data

y_pred = logreg.predict(X_test)
# Calculating the accuracy, precision and the recall

acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Total Accuracy : ', acc_logreg )

print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )

print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )
# Create confusion matrix function to find out sensitivity and specificity

from sklearn.metrics import auc,confusion_matrix

def draw_cm(actual, predicted):

    cm = confusion_matrix( actual, predicted, [1,0]).T

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Yes","No"] , yticklabels = ["Yes","No"] )

    plt.ylabel('Predicted')

    plt.xlabel('Actual')

    plt.show()
# Confusion matrix 

draw_cm(y_test, y_pred)
# Gaussian Naive Bayes



# Import library of Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



# Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets 

model.fit(X_train,y_train)
# Prediction on test set

y_pred = model.predict(X_test)
# Calculating the accuracy, precision and the recall

acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Total Accuracy : ', acc_nb )

print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )

print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )
# Confusion matrix 

draw_cm(y_test, y_pred)
# Decision Tree Classifier



# Import Decision tree classifier

from sklearn.tree import DecisionTreeClassifier



# Create a Decision tree classifier model

clf = DecisionTreeClassifier(criterion = "gini" , min_samples_split = 100, min_samples_leaf = 10, max_depth = 50)



# Train the model using the training sets 

clf.fit(X_train, y_train)
# Model prediction on train data

y_pred = clf.predict(X_train)
# Finding the variable with more importance

feature_importance = pd.DataFrame([X_train.columns, clf.tree_.compute_feature_importances()])

feature_importance = feature_importance.T.sort_values(by = 1, ascending=False)[1:10]
sns.barplot(x=feature_importance[1], y=feature_importance[0])

# Add labels to the graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
# Prediction on test set

y_pred = clf.predict(X_test)
# Confusion matrix

draw_cm(y_test, y_pred)
# Calculating the accuracy, precision and the recall

acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Total Accuracy : ', acc_dt )

print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )

print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )
# Random Forest Classifier



# Import library of RandomForestClassifier model

from sklearn.ensemble import RandomForestClassifier



# Create a Random Forest Classifier

rf = RandomForestClassifier()



# Train the model using the training sets 

rf.fit(X_train,y_train)
# Finding the variable with more importance

feature_imp = pd.Series(rf.feature_importances_,index= X_train.columns).sort_values(ascending=False)

# Creating a bar plot

feature_imp=feature_imp[0:10,]

sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to the graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
# Prediction on test data

y_pred = rf.predict(X_test)
# Confusion metrix

draw_cm(y_test, y_pred)
# Calculating the accuracy, precision and the recall

acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )

print( 'Total Accuracy : ', acc_rf )

print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100 , 2 ) )

print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )
# SVM Classifier



# Creating scaled set to be used in model to improve the results

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Import Library of Support Vector Machine model

from sklearn import svm



# Create a Support Vector Classifier

svc = svm.SVC()



# Train the model using the training sets 

svc.fit(X_train,y_train)
# Prediction on test data

y_pred = svc.predict(X_test)
# Confusion Matrix

draw_cm(y_test, y_pred)
# Calculating the accuracy, precision and the recall

acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Total Accuracy : ', acc_svm )

print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )

print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines'],

    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm]})

models.sort_values(by='Score', ascending=False)