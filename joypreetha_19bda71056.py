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
train=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv") #Loading train dataset

test=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv") #Loading test dataset
train.isna().sum() #Checking for missing values in train dataset
test.isna().sum() #Checking for missing values in test dataset
train.shape #Number of Dimension of train data
test.shape #Number of Dimension of test data
train.columns #Column names
train.describe() #Summary statistics of train dataset
test.describe() #Summary statistics of test dataset
ind_data = train.copy() #Making a copy of the train dataset
ind_data.drop(["flag"], axis = 1, inplace = True) #Dropping the dependent variable
ind_data #Display of the independent variables
y = train["flag"] #assigning dependent variable to y

y
y.value_counts() #Checking count of unique values
y.value_counts().plot.bar() #Plotting a bar plot of the count of unique values
for i in ind_data.columns: #Making boxplots of the independent variables

    ind_data.boxplot()
import matplotlib.pyplot as plt #Is a comprehensive library for creating static, animated, and interactive visualizations.

import seaborn as sns #Is a data visualization library which provides a high-level interface for drawing attractive plots.
sns.pairplot(train ,hue ='flag', vars =['timeindex', 'flag', 'currentBack', 'motorTempBack', 'positionBack',

       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',

       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',

       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',

       'velocityFront']) #To plot pairwise relationship on the dataset
Corrdf=train.corr(method="pearson") #Pearson Correlation matrix on the dataset
#Only keeping the lower triangle of the corr plot.

mask=np.zeros_like(Corrdf)  

mask[np.triu_indices_from(mask)] = True
#Heatmap of the corr plot.

sns.heatmap(Corrdf, cmap="RdYlGn_r", vmax=1.0,mask=mask ,vmin = -1.0, linewidths= 2.5 )
from sklearn.preprocessing import StandardScaler #To Standardize features by removing the mean and scaling to unit variance.



X = StandardScaler().fit_transform(ind_data) #Applying standardization on the dataset.
X #Output of Standardized features.
# Import train_test_split function.

from sklearn.model_selection import train_test_split
#Split dataset into training set and test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10945325) # 70% training and 30% test
print(X_train.shape) #Dimension of the the split train dataset.

print(X_test.shape) #Dimension of the the split test dataset.
#Import svm model.

from sklearn import svm
#Creating a svm Classifier.

clf = svm.SVC()
#Train the model using the training sets.

clf.fit(X_train, y_train)
#Predict the response for test dataset.

y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation.

from sklearn import metrics
# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Model Precision: what percentage of positive tuples are labeled as such?

print("Precision:",metrics.precision_score(y_test, y_pred))



#Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, y_pred))
#Model F1 score: harmonic mean of Precision and Recall.

print("F1 score:",metrics.f1_score(y_test, y_pred))
#Classification report for the model.

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
param_grid ={'C':[0.01,0.1,1,10],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']} #Considering few parameters value.
from sklearn.model_selection import GridSearchCV #Exhaustively searches over specified parameter values for an estimator.
grid=GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=4) #Fitting a SVC model for tuning hyperparameters and assigning class weights to account for the class imbalance.
grid.fit(X_train,y_train) #Searching for the best parameters using Grid Search.
grid.best_params_ #The Best Parameter values.
grid_predictions=grid.predict(X_test) #Predictiing using the new parameter values on the test data.
confusion_matrix(y_test,grid_predictions) #Confusion Matrix for the test data.
metrics.f1_score(y_test,grid_predictions) #F1 score for the improved model.
print(classification_report(y_test,grid_predictions)) #Classification report for the improved model.
from sklearn.model_selection import cross_val_score #Cross validation for evaluating estimator performance. 
clf1 = svm.SVC(C=10, kernel='rbf',gamma=0.1) #RBF kernel and improved parameters fit for an svm.
scores = cross_val_score(clf1, X, y, cv=5) #Implementing 5-fold cross validation on f1 scoring.
scores #f1 scores
sample=pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv") #Reading sample submission data.
Test_x = StandardScaler().fit_transform(test) #Implementing standardization on final test Data
#Training the model using the training sets

clf1.fit(X, y)
prediction = clf1.predict(Test_x) #Prediction for the test data
sample.head(10) #Display of 1st 10 rows of sample submission dataset.
sample["flag"] = prediction #replacing the Flag values with the predicted ones.
sample.head(10) ##Display of 1st 10 rows of the updated sample submission dataset
sample["flag"].value_counts() #Distribution of the Predicted variable.
sample.to_csv("Submit_8.csv", index=False) #Writing the sample submission to a final file