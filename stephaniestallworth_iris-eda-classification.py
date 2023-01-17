# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [10,5]
# Import data

dataset = pd.read_csv('../input/Iris.csv')
# Data shape

print('dataset shape:',dataset.shape)



# View first few rows

print('\n')

print(dataset.head())



# Data types

print('\n')

print(dataset.info())
# Convert to categorical variable

dataset['Species'] = dataset['Species'].astype('category')



# Drop Id

dataset.drop('Id', axis =1, inplace = True)



# Confirm changes

print(dataset.info())
# Species count

print('Target Variable')

print(dataset.groupby(['Species']).Species.count())



# Species countplot

sns.set_style('darkgrid')

sns.countplot(dataset['Species'],alpha = .95, palette = 'inferno')

plt.title('Iris Species')

plt.ylabel ('Count')

plt.show()
print('Statistical Summary')

print(dataset.describe().transpose())



# Subplots of Numeric Variables

sns.set_style('darkgrid')

fig = plt.figure(figsize = (16,10))



ax1 = fig.add_subplot(221)

ax1.hist(dataset['SepalLengthCm'], bins = 20,color = 'teal',edgecolor= 'black',alpha = .70)

ax1.set_xlabel('SepalLengthCm')

ax1.set_ylabel('Count')

ax1.set_title('SepalLength (cm)')



ax2 = fig.add_subplot(222)

ax2.hist(dataset['SepalWidthCm'], bins = 20,color = 'teal',edgecolor= 'black',alpha = .70)

ax2.set_xlabel('SepalWidthCm')

ax2.set_ylabel('Count')

ax2.set_title('Sepal Width (cm)')



ax3 = fig.add_subplot(223)

ax3.hist(dataset['PetalLengthCm'], bins = 20,color = 'teal',edgecolor= 'black',alpha = .70)

ax3.set_xlabel('PetalLengthCm')

ax3.set_ylabel('Count')

ax3.set_title('Petal Length (cm)')



ax4 = fig.add_subplot(224)

ax4.hist(dataset['PetalWidthCm'], bins = 20, color = 'teal',edgecolor= 'black', alpha = .70)  

ax4.set_xlabel('PetalWidthCm')

ax4.set_ylabel('Count')

ax4.set_title('Petal Width (cm)')



plt.show()
# Statistical summary of continuous variables 

print('Statistical Summary by Species')

print('\n')

print('Setosa')

print(dataset[dataset['Species']=='Iris-setosa'].describe().transpose())

print('--'*40)

print('Versicolor')

print(dataset[dataset['Species']== 'Iris-versicolor'].describe().transpose())

print('--'*40)

print('Virginica')

print(dataset[dataset['Species']== 'Iris-virginica'].describe().transpose())



# Subplots of Numeric Features

sns.set_style('darkgrid')

fig = plt.figure(figsize = (16,10))

fig.subplots_adjust(hspace = .30)



ax1 = fig.add_subplot(221)

ax1.hist(dataset[dataset['Species'] =='Iris-setosa'].SepalLengthCm, bins = 12, label ='Setosa', alpha = .80,edgecolor= 'black',color ='green')

ax1.hist(dataset[dataset['Species']=='Iris-versicolor'].SepalLengthCm, bins = 12, label = 'Versicolor', alpha = .80, edgecolor = 'black',color = 'blue')

ax1.hist(dataset[dataset['Species']=='Iris-virginica'].SepalLengthCm, bins = 12, label = 'Verginica', alpha = .80, edgecolor = 'black',color = 'orange')

ax1.set_title('Sepal Length by Species')

ax1.set_xlabel('Sepal Length (cm)')

ax1.set_ylabel('Count')

ax1.legend(loc = 'upper right')



ax2 = fig.add_subplot(222)

ax2.hist(dataset[dataset['Species'] =='Iris-setosa'].SepalWidthCm, bins = 12, label ='Setosa', alpha = .80,edgecolor= 'black',color ='green')

ax2.hist(dataset[dataset['Species']=='Iris-versicolor'].SepalWidthCm, bins = 12, label = 'Versicolor', alpha = .80, edgecolor = 'black',color = 'blue')

ax2.hist(dataset[dataset['Species']=='Iris-virginica'].SepalWidthCm, bins = 12, label = 'Verginica', alpha = .80, edgecolor = 'black',color = 'orange')

ax2.set_title('Sepal Width by Species')

ax2.set_xlabel('Sepal Width (cm)')

ax2.set_ylabel('Count')

ax2.legend(loc = 'upper right')



ax3 = fig.add_subplot(223)

ax3.hist(dataset[dataset['Species'] =='Iris-setosa'].PetalLengthCm, bins = 12, label ='Setosa', alpha = .80,edgecolor= 'black',color ='green')

ax3.hist(dataset[dataset['Species']=='Iris-versicolor'].PetalLengthCm, bins = 12, label = 'Versicolor', alpha = .80, edgecolor = 'black',color = 'blue')

ax3.hist(dataset[dataset['Species']=='Iris-virginica'].PetalLengthCm, bins = 12, label = 'Verginica', alpha = .80, edgecolor = 'black',color = 'orange')

ax3.set_title('Petal Length by Species')

ax3.set_xlabel('Petal Length (cm)')

ax3.set_ylabel('Count')

ax3.legend(loc = 'upper right')



ax4 = fig.add_subplot(224)

ax4.hist(dataset[dataset['Species'] =='Iris-setosa'].PetalWidthCm, bins = 12, label ='Setosa', alpha = .80,edgecolor= 'black',color ='green')

ax4.hist(dataset[dataset['Species']=='Iris-versicolor'].PetalWidthCm, bins = 12, label = 'Versicolor', alpha = .80, edgecolor = 'black',color = 'blue')

ax4.hist(dataset[dataset['Species']=='Iris-virginica'].PetalWidthCm, bins = 12, label = 'Verginica', alpha = .80, edgecolor = 'black',color = 'orange')

ax4.set_title('Petal Width by Species')

ax4.set_xlabel('Petal Width (cm)')

ax4.set_ylabel('Count')

ax4.legend(loc = 'upper right')



plt.show()
plt.figure(figsize=(12,7))

sns.heatmap(dataset.corr(),cmap = 'coolwarm',linewidth = 1,annot= True, annot_kws={"size": 15})

plt.title('Correlation Coefficients')

plt.show()
# Pairplot

sns.pairplot(dataset, hue = 'Species', palette = 'inferno', diag_kws={'edgecolor':'w'})

plt.show()
# Split

from sklearn.model_selection import train_test_split



# Create matrix of features

x = dataset.drop('Species', axis = 1)



# Create independent variable array

y = dataset['Species']



# Split data in test and train sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
# Fit 

# Import model

from sklearn.linear_model import LogisticRegression



# Create instance of model

lreg = LogisticRegression()



# Fit model with training data

lreg = lreg.fit(x_train, y_train)
# Predict

y_pred_lreg = lreg.predict(x_test)
# Score It

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



# Confusion Matrix

print('Logistic Regression')

print(lreg)

print('--'*40)

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_lreg))

print('--'*40)



# Classification Report

print('Classification Report')

print(classification_report(y_test,y_pred_lreg))



# Average Accuracy Using Cross Validation

from sklearn.model_selection import cross_val_score

lreg_accuracy_avg= cross_val_score(estimator = lreg, X = x_train, y = y_train, cv = 10).mean()

print('--'*40)

print('Average Accuracy', round(lreg_accuracy_avg *100,2),'%')
# Import model

from sklearn.svm import SVC



# Create instance of model

svc = SVC()



# Import GridSearch

from sklearn.model_selection import GridSearchCV



# Create dictionary of parameters

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, # investigate linear model option

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]} #non-linear model option and investigate several sub-options of penalty parameters and gamma

              ]



# Create GridSearchCV object and pass in model

svc_grid = GridSearchCV(estimator = svc, # machine learning model

                           param_grid = parameters, 

                           scoring = 'accuracy', # scoring metric we're going to use to decide what the best parameters are (could be accuracy, precision, recall)

                           cv = 10, # so 10 fold cross validation will be applied through grid search

                           n_jobs = -1) # for large data sets

                          

# Fit grid search object to training set

svc_grid = svc_grid.fit(x_train, y_train)
# Use best model and predict

y_pred_svc_grid = svc_grid.predict(x_test)
# Score It

from sklearn.metrics import confusion_matrix, classification_report

print('SVC with Grid Search')

print(svc_grid.best_estimator_)

print('--'*40)

# Confusion Matrix

print('Confusion Matrix')

print(confusion_matrix(y_test,y_pred_svc_grid))

print('--'*40)



# Classification Report

print('Classification Report')

print(classification_report(y_test, y_pred_svc_grid))

print('--'*40)



# Average accuracy

svc_grid_accuracy_avg = svc_grid.best_score_

print('Average Accuracy',round(svc_grid_accuracy_avg*100,2),'%')
# Preprocessing - Standardize features

# Import StandardScaler

from sklearn.preprocessing import StandardScaler



# Create instance of standard scaler

scaler = StandardScaler()



# Tranform features

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.fit_transform(x_test)
# Determine the best k parameter



# Import model

from sklearn.neighbors import KNeighborsClassifier



# Function

error_rate = []



for i in range (1,40):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train_scaled, y_train)

    pred_i = knn.predict(x_test_scaled)

    error_rate.append(np.mean(pred_i != y_test))



# Plot error rate

plt.figure(figsize = (10,6))

plt.plot(range(1,40), error_rate, color = 'blue', linestyle = '--', marker = 'o', 

        markerfacecolor = 'green', markersize = 10)



plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.show()
# Fit



# Import model

from sklearn.neighbors import KNeighborsClassifier



# Create instance of model with the best k value

knn = KNeighborsClassifier(n_neighbors = 12)



# Fit model to training data

knn = knn.fit(x_train_scaled,y_train)
# Predict

y_pred_knn = knn.predict(x_test_scaled)
# Score It

print('KNN')

print(knn)

print('--'*40)

from sklearn.metrics import confusion_matrix, classification_report



# Confusion Matrix

print('Confusion Matrix')

print(confusion_matrix(y_test,y_pred_knn))

print('--'*40)



# Classification Report

print('Classification Report')

print(classification_report(y_test, y_pred_knn))

print('--'*40)



# Average Accuracy with cross validation

from sklearn.model_selection import cross_val_score

knn_accuracy_avg= cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10).mean()

print('Average Accuracy', round(knn_accuracy_avg *100,2),'%')
# Fit

# Import Model

from sklearn.tree import DecisionTreeClassifier



# Create instance of model

dtree = DecisionTreeClassifier()



# Fit model to training data

dtree = dtree.fit(x_train, y_train)
# Predict

y_pred_dtree = dtree.predict(x_test)
# Score It

from sklearn.metrics import confusion_matrix, classification_report

print('Decision Tree')

print(dtree)

print('--'*40)



# Confusion Matrix

print('Confusion Matrix')

print(confusion_matrix(y_test,y_pred_dtree))

print('--'*40)

      

# Classification Report

print('Classification Report')

print(classification_report(y_test,y_pred_dtree))

      

# Average Accuracy using cross validation

from sklearn.model_selection import cross_val_score

      

dtree_accuracy_avg = cross_val_score(estimator = dtree, X = x_train, y = y_train, cv = 10).mean()

print('Average Accuracy',round(dtree_accuracy_avg*100,2),'%')
# Fit



# Import model

from sklearn.ensemble import RandomForestClassifier



# Create model object

rforest = RandomForestClassifier()



# Fit model to training data

rforest = rforest.fit(x_train, y_train)
# Predict

y_pred_rforest = rforest.predict(x_test)
# Score It

from sklearn.metrics import confusion_matrix, classification_report

print('Random Forest')

print(rforest)

print('--'*40)



# Confusion Matrix

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred_rforest))

print('--'*40)



# Classification Report

print('Classification Report')

print(classification_report(y_test, y_pred_rforest))



# Average Accuracy using cross validation

from sklearn.model_selection import cross_val_score

rforest_accuracy_avg = cross_val_score(estimator = rforest, X = x_train, y = y_train, cv = 10).mean()

print('Average Accuracy',round(rforest_accuracy_avg *100,2),'%')
models = pd.DataFrame({

     'Avg Accuracy': [lreg_accuracy_avg, svc_grid_accuracy_avg, knn_accuracy_avg, dtree_accuracy_avg, rforest_accuracy_avg]},

      index = list(['Logistic Regression', 'SVC with GridSearch',  'K-Nearest Neighbors', 'Decision Tree', 'Random Forest']))

print(models.sort_values(by='Avg Accuracy', ascending=False))