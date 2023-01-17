# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = [10,5]
# Read data

train_data = pd.read_csv('../input/train.csv')
# Data shape

print('train data:',train_data.shape)
# View first few rows

train_data.head(3)
# Data Info

train_data.info()
# Heatmap

sns.heatmap(train_data.isnull(),yticklabels = False, cbar = False,cmap = 'tab20c_r')

plt.title('Missing Data: Training Set')

plt.show()
plt.figure(figsize = (10,7))

sns.boxplot(x = 'Pclass', y = 'Age', data = train_data, palette= 'GnBu_d').set_title('Age by Passenger Class')

plt.show()
# Imputation function

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            

            return 29



        else:

            return 24



    else:

        return Age

    

# Apply the function to the Age column

train_data['Age']=train_data[['Age','Pclass']].apply(impute_age, axis =1 )    
# Remove Cabin feature

train_data.drop('Cabin', axis = 1, inplace = True)
# Remove rows with missing data

train_data.dropna(inplace = True)
# Data types

print(train_data.info())



# Identify non-null objects

print('\n')

print('Non-Null Objects to Be Converted to Category')

print(train_data.select_dtypes(['object']).columns)
# Remove unnecessary columns  

train_data.drop(['Name','Ticket'], axis = 1, inplace = True)



# Convert objects to category data type

objcat = ['Sex','Embarked']



for colname in objcat:

    train_data[colname] = train_data[colname].astype('category')
# Numeric summary

train_data.describe().transpose()
# Remove PassengerId

train_data.drop('PassengerId', inplace = True, axis = 1)
# Survival Count

print('Target Variable')

print(train_data.groupby(['Survived']).Survived.count())



# Target Variable Countplot

sns.set_style('darkgrid')

plt.figure(figsize = (10,5))

sns.countplot(train_data['Survived'], alpha =.80, palette= ['grey','lightgreen'])

plt.title('Survivors vs Non-Survivors')

plt.ylabel('# Passengers')

plt.show()

# Identify numeric features

print('Continuous Variables')

print(train_data[['Age','Fare']].describe().transpose())

print('--'*40)

print('Discrete Variables')

print(train_data.groupby('Pclass').Pclass.count())

print(train_data.groupby('SibSp').SibSp.count())

print(train_data.groupby('Parch').Parch.count())



# Subplots of Numeric Features

sns.set_style('darkgrid')

fig = plt.figure(figsize = (20,16))

fig.subplots_adjust(hspace = .30)



ax1 = fig.add_subplot(321)

ax1.hist(train_data['Pclass'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')

ax1.set_xlabel('Pclass', fontsize = 15)

ax1.set_ylabel('# Passengers',fontsize = 15)

ax1.set_title('Passenger Class',fontsize = 15)



ax2 = fig.add_subplot(323)

ax2.hist(train_data['Age'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')

ax2.set_xlabel('Age',fontsize = 15)

ax2.set_ylabel('# Passengers',fontsize = 15)

ax2.set_title('Age of Passengers',fontsize = 15)



ax3 = fig.add_subplot(325)

ax3.hist(train_data['SibSp'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')

ax3.set_xlabel('SibSp',fontsize = 15)

ax3.set_ylabel('# Passengers',fontsize = 15)

ax3.set_title('Passengers with Spouses or Siblings',fontsize = 15)



ax4 = fig.add_subplot(222)

ax4.hist(train_data['Parch'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')

ax4.set_xlabel('Parch',fontsize = 15)

ax4.set_ylabel('# Passengers',fontsize = 15)

ax4.set_title('Passengers with Children',fontsize = 15)



ax5 = fig.add_subplot(224)

ax5.hist(train_data['Fare'], bins = 20, alpha = .50,edgecolor= 'black',color ='teal')

ax5.set_xlabel('Fare',fontsize = 15)

ax5.set_ylabel('# Passengers',fontsize = 15)

ax5.set_title('Ticket Fare',fontsize = 15)



plt.show()
# Passenger class summary

print('Passenger Class Summary')



print('\n')

print(train_data.groupby(['Pclass','Survived']).Pclass.count().unstack())



# Passenger class visualization

pclass = train_data.groupby(['Pclass','Survived']).Pclass.count().unstack()

p1 = pclass.plot(kind = 'bar', stacked = True, 

                   title = 'Passengers by Class: Survivors vs Non-Survivors', 

                   color = ['grey','lightgreen'], alpha = .70)

p1.set_xlabel('Pclass')

p1.set_ylabel('# Passengers')

p1.legend(['Did Not Survive','Survived'])

plt.show()
# SibSp Summary

print('Passengers with Siblings or Spouse')

print('\n')

print(train_data.groupby(['SibSp','Survived']).SibSp.count().unstack())



sibsp = train_data.groupby(['SibSp','Survived']).SibSp.count().unstack()

p2 = sibsp.plot(kind = 'bar', stacked = True,

                   color = ['grey','lightgreen'], alpha = .70)

p2.set_title('Passengers with Siblings or Spouse: Survivors vs Non-Survivors')

p2.set_xlabel('Sibsp')

p2.set_ylabel('# Passengers')

p2.legend(['Did Not Survive','Survived'])

plt.show()
print(train_data.groupby(['Parch','Survived']).Parch.count().unstack())



parch = train_data.groupby(['Parch','Survived']).Parch.count().unstack()

p3 = parch.plot(kind = 'bar', stacked = True,

                   color = ['grey','lightgreen'], alpha = .70)

p3.set_title('Passengers with Children: Survivors vs Non-Survivors')

p3.set_xlabel('Parch')

p3.set_ylabel('# Passengers')

p3.legend(['Did Not Survive','Survived'])

plt.show()
# titanic.hist(bins=10,figsize=(9,7),grid=False)

# Statistical summary of continuous variables 

print('Statistical Summary of Age and Fare')

print('\n')

print('Did Not Survive')

print(train_data[train_data['Survived']==0][['Age','Fare']].describe().transpose())

print('--'*40)

print('Survived')

print(train_data[train_data['Survived']==1][['Age','Fare']].describe().transpose())

# Subplots of Numeric Features

sns.set_style('darkgrid')

fig = plt.figure(figsize = (16,10))

fig.subplots_adjust(hspace = .30)



ax1 = fig.add_subplot(221)

ax1.hist(train_data[train_data['Survived'] ==0].Age, bins = 25, label ='Did Not Survive', alpha = .50,edgecolor= 'black',color ='grey')

ax1.hist(train_data[train_data['Survived']==1].Age, bins = 25, label = 'Survive', alpha = .50, edgecolor = 'black',color = 'lightgreen')

ax1.set_title('Passenger Age: Survivors vs Non-Survivors')

ax1.set_xlabel('Age')

ax1.set_ylabel('# Passengers')

ax1.legend(loc = 'upper right')



ax2 = fig.add_subplot(223)

ax2.hist(train_data[train_data['Survived']==0].Fare, bins = 25, label = 'Did Not Survive', alpha = .50, edgecolor ='black', color = 'grey')

ax2.hist(train_data[train_data['Survived']==1].Fare, bins = 25, label = 'Survive', alpha = .50, edgecolor = 'black',color ='lightgreen')

ax2.set_title('Ticket Fare: Suvivors vs Non-Survivors')

ax2.set_xlabel('Fare')

ax2.set_ylabel('# Passenger')

ax2.legend(loc = 'upper right')



ax3 = fig.add_subplot(122)

ax3.scatter(x = train_data[train_data['Survived']==0].Age, y = train_data[train_data['Survived']==0].Fare,

                        alpha = .50,edgecolor= 'black',  c = 'grey', s= 75, label = 'Did Not Survive')

ax3.scatter(x = train_data[train_data['Survived']==1].Age, y = train_data[train_data['Survived']==1].Fare,

                        alpha = .50,edgecolors= 'black',  c = 'lightgreen', s= 75, label = 'Survived')

ax3.set_xlabel('Age')

ax3.set_ylabel('Fare')

ax3.set_title('Age of Passengers vs Fare')

ax3.legend()



plt.show()
# Identify categorical features

train_data.select_dtypes(['category']).columns
# Suplots of categorical features v price

sns.set_style('darkgrid')

f, axes = plt.subplots(1,2, figsize = (15,5))



# Plot [0]

sns.countplot(x = 'Sex', data = train_data, palette = 'GnBu_d', ax = axes[0])

axes[0].set_xlabel('Sex')

axes[0].set_ylabel('# Passengers')

axes[0].set_title('Gender of Passengers')



# Plot [1]

sns.countplot(x = 'Embarked', data = train_data, palette = 'GnBu_d',ax = axes[1])

axes[1].set_xlabel('Embarked')

axes[1].set_ylabel('# Passengers')

axes[1].set_title('Embarked')



plt.show()
# Suplots of categorical features v price

sns.set_style('darkgrid')

f, axes = plt.subplots(1,2, figsize = (20,7))



gender = train_data.groupby(['Sex','Survived']).Sex.count().unstack()

p1 = gender.plot(kind = 'bar', stacked = True, 

                   title = 'Gender: Survivers vs Non-Survivors', 

                   color = ['grey','lightgreen'], alpha = .70, ax = axes[0])

p1.set_xlabel('Sex')

p1.set_ylabel('# Passengers')

p1.legend(['Did Not Survive','Survived'])





embarked = train_data.groupby(['Embarked','Survived']).Embarked.count().unstack()

p2 = embarked.plot(kind = 'bar', stacked = True, 

                    title = 'Embarked: Survivers vs Non-Survivors', 

                    color = ['grey','lightgreen'], alpha = .70, ax = axes[1])

p2.set_xlabel('Embarked')

p2.set_ylabel('# Passengers')

p2.legend(['Did Not Survive','Survived'])



plt.show()
# Shape of train data

train_data.shape
# Identify categorical features

train_data.select_dtypes(['category']).columns
# Convert categorical variables into 'dummy' or indicator variables

sex = pd.get_dummies(train_data['Sex'], drop_first = True) # drop_first prevents multi-collinearity

embarked = pd.get_dummies(train_data['Embarked'], drop_first = True)
# Add new dummy columns to data frame

train_data = pd.concat([train_data, sex, embarked], axis = 1)

train_data.head(2)
# Drop unecessary columns

train_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)



# Shape of train data

print('train_data shape',train_data.shape)



# Confirm changes

train_data.head()
# Split data to be used in the models

# Create matrix of features

x = train_data.drop('Survived', axis = 1) # grabs everything else but 'Survived'



# Create target variable

y = train_data['Survived'] # y is the column we're trying to predict



# Use x and y variables to split the training data into train and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 101)
# Fit 

# Import model

from sklearn.linear_model import LogisticRegression



# Create instance of model

lreg = LogisticRegression()



# Pass training data into model

lreg.fit(x_train, y_train)
# Predict

y_pred_lreg = lreg.predict(x_test)
# Score It

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



# Confusion Matrix

print('Logistic Regression')

print('\n')

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_lreg))

print('--'*40)



# Classification Report

print('Classification Report')

print(classification_report(y_test,y_pred_lreg))



# Accuracy

print('--'*40)

logreg_accuracy = round(accuracy_score(y_test, y_pred_lreg) * 100,2)

print('Accuracy', logreg_accuracy,'%')
# Fit

# Import model

from sklearn.svm import SVC



# Instantiate the model

svc = SVC()



# Fit the model on training data

svc.fit(x_train, y_train)
# Predict

y_pred_svc = svc.predict(x_test)
# Score It

print('Support Vector Classifier')

print('\n')

# Confusion matrix

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_svc))

print('--'*40)



# Classification report

print('Classification Report')

print(classification_report(y_test, y_pred_svc))



# Accuracy

print('--'*40)

svc_accuracy = round(accuracy_score(y_test, y_pred_svc)*100,2)

print('Accuracy', svc_accuracy,'%')
# Create parameter grid

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
# Fit

# Import

from sklearn.model_selection import GridSearchCV



# Instantiate grid object

grid = GridSearchCV(SVC(),param_grid, refit = True, verbose = 1)#verbose is the text output describing the process



# Fit to training data

grid.fit(x_train,y_train)
# Call best_params attribute

print(grid.best_params_)

print('\n')

# Call best_estimators attribute

print(grid.best_estimator_)
# Predict using best parameters

y_pred_grid = grid.predict(x_test)
# Score It

# Confusion Matrix

print('SVC with GridSearchCV')

print('\n')

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_grid))

print('--'*40)

# Classification Report

print('Classification Report')

print(classification_report(y_test, y_pred_grid))



# Accuracy

print('--'*40)

svc_grid_accuracy = round(accuracy_score(y_test, y_pred_grid)*100,2)

print('Accuracy',svc_grid_accuracy,'%')
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)

x_test_sc = sc.transform(x_test)
# Fit

# Import model

from sklearn.svm import SVC



# Instantiate model object

ksvc= SVC(kernel = 'rbf', random_state = 0)



# Fit on training data

ksvc.fit(x_train_sc, y_train)
# Predict

y_pred_ksvc = ksvc.predict(x_test_sc)
# Score it

print('Kernel SVC')

# Confusion Matrix

print('\n')

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_ksvc))



# Classification Report

print('--'*40)

print('Classification Report')

print(classification_report(y_test, y_pred_ksvc))



# Accuracy

print('--'*40)

ksvc_accuracy = round(accuracy_score(y_test,y_pred_ksvc)*100,1)

print('Accuracy',ksvc_accuracy,'%')
# Standardize the Variables



# Import StandardScaler

from sklearn.preprocessing import StandardScaler



# Create instance of standard scaler

scaler = StandardScaler()



# Fit scaler object to feature columns

scaler.fit(train_data.drop('Survived', axis = 1)) # Everything but target variable 



# Use scaler object to do a transform columns

scaled_features = scaler.transform(train_data.drop('Survived', axis = 1)) # performs the standardization by centering and scaling

scaled_features
# Use scaled features variable to re-create a features dataframe

df_feat = pd.DataFrame(scaled_features, columns = train_data.columns[:-1])
# Split

# Import

from sklearn.model_selection import train_test_split



# Create matrix of features

x = df_feat



# Create target variable

y = train_data['Survived']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 101)
# Fit

# Import model

from sklearn.neighbors import KNeighborsClassifier



# Create instance of model

knn = KNeighborsClassifier(n_neighbors = 1)



# Fit to training data

knn.fit(x_train,y_train)
# Predict

y_pred_knn = knn.predict(x_test)
# Score it

print('K-Nearest Neighbors (KNN)')

print('k = 1')

print('\n')

# Confusion Matrix

print('Confusion Matrix')

print(confusion_matrix(y_test,y_pred_knn))



# Classification Report

print('--'*40)

print('Classification Report')

print(classification_report(y_test, y_pred_knn))



# Accuracy

print('--'*40)

knn_accuracy = round(accuracy_score(y_test, y_pred_knn)*100,1)

print('Accuracy',knn_accuracy,'%')
# Function

error_rate = []



for i in range (1,40):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train, y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))



# Plot error rate

plt.figure(figsize = (10,6))

plt.plot(range(1,40), error_rate, color = 'blue', linestyle = '--', marker = 'o', 

        markerfacecolor = 'green', markersize = 10)



plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.show()
# Fit new KNN

# Create model object

knn = KNeighborsClassifier(n_neighbors = 13)



# Fit new KNN on training data

knn.fit(x_train, y_train)
# Predict new KNN

y_pred_knn_op = knn.predict(x_test)
# Score it with new KNN

print('K-Nearest Neighbors(KNN)')

print('k = 13')



# Confusion Matrix

print('\n')

print(confusion_matrix(y_test, y_pred_knn_op))



# Classification Report

print('--'*40)

print('Classfication Report',classification_report(y_test, y_pred_knn_op))



# Accuracy

print('--'*40)

knn_op_accuracy =round(accuracy_score(y_test, y_pred_knn_op)*100,2)

print('Accuracy',knn_op_accuracy,'%')
# Fit

# Import model

from sklearn.tree import DecisionTreeClassifier



# Create model object

dtree = DecisionTreeClassifier()



# Fit to training sets

dtree.fit(x_train,y_train)
# Predict

y_pred_dtree = dtree.predict(x_test)
# Score It

print('Decision Tree')

# Confusion Matrix

print('\n')

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_dtree))



# Classification Report

print('--'*40)

print('Classification Report',classification_report(y_test, y_pred_dtree))



# Accuracy

print('--'*40)

dtree_accuracy = round(accuracy_score(y_test, y_pred_dtree)*100,2)

print('Accuracy',dtree_accuracy,'%')
# Fit

# Import model object

from sklearn.ensemble import RandomForestClassifier



# Create model object

rfc = RandomForestClassifier(n_estimators = 200)



# Fit model to training data

rfc.fit(x_train,y_train)
# Predict

y_pred_rfc = rfc.predict(x_test)
# Score It

print('Random Forest')

# Confusion matrix

print('\n')

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_rfc))



# Classification report

print('--'*40)

print('Classification Report')

print(classification_report(y_test, y_pred_rfc))



# Accuracy

print('--'*40)

rf_accuracy = round(accuracy_score(y_test, y_pred_rfc)*100,2)

print('Accuracy', rf_accuracy,'%')
models = pd.DataFrame({

     'Model': ['Logistic Regression', 'Linear SVC', 'Kernel SVC', 

               'K-Nearest Neighbors', 'Decision Tree', 'Random Forest'],

    'Score': [logreg_accuracy, svc_grid_accuracy, ksvc_accuracy, 

               knn_op_accuracy, dtree_accuracy, rf_accuracy]})

models.sort_values(by='Score', ascending=False)
# Load test data

test_data = pd.read_csv('../input/test.csv')



# Test data info

test_data.info()



# Test data shape

print('shape',test_data.shape)
# Heatmap

sns.heatmap(test_data.isnull(),yticklabels = False, cbar = False,cmap = 'tab20c_r')

plt.title('Missing Data Test Set')

plt.show()
# Missing Data

# Impute Age

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            

            return 29



        else:

            return 24



    else:

        return Age

    

# Apply the function to the Age column

test_data['Age']=test_data[['Age','Pclass']].apply(impute_age, axis =1 )    



# Drop cabin feature

test_data = test_data.drop(['Cabin'], axis = 1)



# Impute Fare

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



# Confirm changes

test_data.info()
# Remove unecessary columns

test_data = test_data.drop(['Name','Ticket'],axis = 1)

test_data.columns
# Identify categorical variables

test_data.select_dtypes(['object']).columns
# Convert categorical variables into 'dummy' or indicator variables

testsex = pd.get_dummies(test_data['Sex'], drop_first = True) # drop_first prevents multi-collinearity

testembarked = pd.get_dummies(test_data['Embarked'], drop_first = True)



# Add new dummy columns to data frame

test_data = pd.concat([test_data, testsex, testembarked], axis = 1)



# Drop unecessary columns

test_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
# Test data shape

print(test_data.shape)



# Confirm changes

test_data.head()
# Split

x_train2 = train_data.drop("Survived", axis=1)

y_train2 = train_data["Survived"]

x_test2  = test_data.drop("PassengerId", axis=1).copy()

print('x_train shape', x_train2.shape)

print('y_train shape',y_train2.shape)

print('x_test shape', x_test2.shape)
# Fit new KNN

# Create model object

knn2 = KNeighborsClassifier(n_neighbors = 13)



# Fit new KNN on training data

knn2.fit(x_train2, y_train2)
# Predict 

y_pred_knn_op2 = knn2.predict(x_test2)
# Create contest submission

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": y_pred_knn_op2

    })



submission.to_csv('mySubmission.csv', index=False)