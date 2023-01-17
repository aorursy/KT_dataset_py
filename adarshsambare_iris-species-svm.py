# Importing required libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
## Reading the data
iris = pd.read_csv('../input/iris/Iris.csv')
## Checking the data
iris.head(2)
# as you can see id is unnecessary column
# Dropping the id column.
iris.drop('Id', axis =1, inplace =True)
## Checking Null values 
fig, ax = plt.subplots(figsize=(12,12)) # increase the figure size
sns.heatmap(iris.isnull()) 
# NO null values
## Pair plot on whole data set
sns.pairplot(iris,hue = 'Species')
# As our data set is small in number we can refer to pair plot
## KDE plot for Iris-setosa
setosa = iris[iris['Species']=='Iris-setosa']
sns.kdeplot(setosa['SepalWidthCm'],setosa['SepalWidthCm'],
           cmap = 'plasma',shade = True, shade_lowest = False)
from sklearn.model_selection import train_test_split
## y is Species
## x is everthing except Species columns

x = iris.drop('Species',axis =1)
y = iris['Species']

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.3,random_state = 101)

## Importing SVC 
from sklearn.svm import SVC
svc_model = SVC()
## Fitting the model to training set
svc_model.fit(x_train,y_train)
## Predicting with our model on test data 
predict =  svc_model.predict(x_test)
## Confusion matrix and classification report
from sklearn.metrics import classification_report,confusion_matrix
## Printing the results
print(classification_report(predict,y_test))
print('\n')
print(confusion_matrix(predict,y_test))
## accuracy = 98%
## Grid Search is one of the technique used
from sklearn.model_selection import GridSearchCV
## Giving the parameter grid lower and upper values 
param_grid = {'C':[0.1,1,10,100,100],
             'gamma':[1,0.1,0.01,0.001,0.0001]}
## Creating a grid model with SVC Model
grid = GridSearchCV(SVC(),param_grid,verbose=3)
## Fitting the grid model on our training data
grid.fit(x_train,y_train)
# fitting large data to grid may take some time
## The best fit parameter is given by:
print(grid.best_params_)
print('\n')
print(grid.best_estimator_)
## Checking the best score possible on Grid SVC model
grid.best_score_
# though yo can see 100% accuracy if you browse through grid model
# here it is showing 96.2% only
## Manual fine tunning the model
svc_model_2 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3,
                  kernel='rbf', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
## Fitting the model to training data
svc_model_2.fit(x_train,y_train)
## predicting on our test data
svc_predict_2 = svc_model_2.predict(x_test)
## The classificatiion report and confusion matrix 
print(classification_report(svc_predict_2,y_test))
print('/n')
print(confusion_matrix(svc_predict_2,y_test))