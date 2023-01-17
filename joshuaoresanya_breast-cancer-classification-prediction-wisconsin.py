import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.datasets import load_breast_cancer
#Then we take an instance of the load_breat_cancer 

cancer = load_breast_cancer()
#Let's now view the cancer dataset:

cancer
#To check the dictionaries we have within Cancer, run cancer.keys() method:

cancer.keys()
#Let's view contents of some dictionaries:

print (cancer['DESCR'])
print (cancer['data'])
print (cancer['feature_names'])
#To print file path or location of the dataset:

print (cancer['filename'])
#To check the target or representation (in binary or boolean), where zero (0) = malignant and one (1) = benign.

print (cancer['target'])
#To check the target_names representation, that is classification of zeros and ones in the previous line of code:

print (cancer['target_names'])
#To check the shape of the data:

cancer['data'].shape
#Creating a dataframe is essential to present all the data tabularly instead of as arrays like we have been seeing previously

df_cancer = pd.DataFrame (np.c_[cancer['data'], cancer['target']], columns = np.append (cancer ['feature_names'],['target']))
#Let's view the first 5 rows of the dataframe using .head() method

df_cancer.head()
#Using Pairplot and specifying the specific columns to be displayed.

sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
#To specify the target class, include the HUE option and from the graph, 0 (blue) indicates malignant and 1 indicates benign (orange):

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
#Using the target column to plot its total count per classification against each other, showing as 0 (blue) indicates malignant and 1 indicates benign

sns.countplot(df_cancer['target'])

#From the graph, you can see that malignant has a little of 200 count and benign has a little over 350 count.
#Using Scatterplot and specifying the specific columns to be displayed.

sns.scatterplot (x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
#To have an overall view of the correlation of the data using heatmap, we do:

plt.figure (figsize = (30, 15))

sns.heatmap(df_cancer.corr(), annot = True)
#Define our X (inpput) value 

#Drop the target column while showing everything else within df_cancer

X = df_cancer.drop(['target'], axis = 1)

X.head()

# Do not forget that upper case X was used.
#Define our y (output) values (target)

y = df_cancer['target']

#Display first 5 rows of y

y.head()
#Let's split our data into split and test. Then after training, apply the test data set which the model hasnt seen before.

from sklearn.model_selection import train_test_split

# Method used to store training set or data:

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.33, random_state = 42)

#Note: Random_state of 42 is the default and test_size means 33% is assigned to test while the rest is assigned to trian.
#Let's view our X_train

X_train.head(10)
#Let's view out y_train

y_train.head(10)
# Using Support Vector Machines and we are gonna use the metrics inside it which are classification report and confusion matrix:

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
# Let's extentiate an object out of the SVC, in this case, we name it svc_model

svc_model = SVC()

# We use the fit method to train the data and then we are done. The next step would be to evaluate the model for efficiency 

svc_model.fit(X_train, y_train)
#Do not forget that we now want to evaluate the data using test data.

# We are naming it y_predict because we are now trying to see the result of our prediction using the test data.

y_predict = svc_model.predict(X_test)
# View y_predict. It returns values of ones as it is not trained yet.

y_predict
# Hence we plot a confusion matrix which is a one stop shop showing all the metrics showing all correctly and missed classified samples

cm = confusion_matrix (y_test, y_predict)
# Let's view the confusion matrix using heatmap on seaborn

sns.heatmap(cm)

# Note that just running CM does not show any values or counts within the heatmap, until you include annotation as in the cell below.
# Let's view the confusion matrix using heatmap on seaborn

sns.heatmap(cm, annot=True)

# Note:The annot=True shows the values within the heatmap and makes it more readable.
# First improvement using norminalization from 0-1

# Normalize X_train, then optain the range.

min_train = X_train.min()

range_train = (X_train-min_train).max()

X_train_scaled = (X_train - min_train)/range_train
# Let's plot the train data to verify we scaled it correctly. The 'mean area' is not yet scaled hence we can still improve

sns.scatterplot(x= X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
sns.scatterplot(x= X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

#Notice that the mean area which is our x axis is now also normalized.
# Let's perform the same norminalization on the X_test. 

# Note that it is the same code we used above, except that we replaced train with test.

min_test = X_test.min()

range_test = (X_test-min_test).max()

X_test_scaled = (X_test - min_test)/range_test
# Let's fit the model again after norminalization:

svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict (X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap (cm, annot = True)
# Let's print our classification report:

print (classification_report (y_test, y_predict))

#Note that we have not even tuned the parameters of the support vector machine (svm) yet and we have an accuracy of 97%.
#Note that sklearn already has embedded within it a way of searching for the best c and gamma parameters.<br>

#So let's use that option to optimize our C and gamma parameters:

param_grid = {'C': [0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
### Importing GridSearchCV

from sklearn.model_selection import GridSearchCV
# Let's apply the GridSearchCV:

grid = GridSearchCV (SVC(), param_grid, refit = True, verbose = 4)
# Let's use our grid to fit the training data:

grid.fit(X_train_scaled, y_train)
# To get our best values:

grid.best_params_
# Let's use our grid object to plot our confusion matrix prediction 

grid_predictions = grid.predict(X_test_scaled)
# Let's use our confusion matrix again:

cm = confusion_matrix (y_test, grid_predictions)
# Let's now view the result on a heatmap:

sns.heatmap (cm, annot = True)

# We are expecting to see less than 5 False values to show improvement from what we had before.

# Unfortunately, we got values of 19 for type II error and 1 for type I error.
# Let's plot our prediction result on grid_predictions instead of y_predict

print (classification_report(y_test, grid_predictions))

# Note that the accuracy reduced to 89% instead of increase from 97%. 

#So we would go with out earlier accuracy of 97%.