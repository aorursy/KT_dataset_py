# for performing mathematical operations

import numpy as np 



# for data processing, CSV file I/O 

import pandas as pd 



# for plotting and visualozing data

import matplotlib.pyplot as plt 
# read the data from the csv file into a dataframe

labels = ['Wood Slab', 'Slab Height', 'Slab Width', 'Slab Shade']

dataset = pd.read_csv('../input/raw-materials-wood-slabs/Raw_Materials_Wood_Slabs.csv', names=labels, header=None)
# checking first five rows of our dataset

dataset.head()
# extracting information from the dataset for the predictor and target variables

dataset.info()
# finding out all types of wood slabs that exist as our target and their respective count

print("\nDifferent Types of Wood")

dataset['Wood Slab'].value_counts()
# finding out the null values 

dataset.isnull().sum()
# eliminating all the null values 

dataset = dataset.dropna()
# looking for null values

dataset.isnull().sum()
# visualize the relationship between the features and the target using scatterplots

fig, axs = plt.subplots(1, 3, figsize=(20, 8))

dataset.plot(kind='scatter',  x='Wood Slab', y='Slab Height', ax=axs[0], title="Height of Wood Slabs")

dataset.plot(kind='scatter',  x='Wood Slab', y='Slab Width', ax=axs[1], title="Width of Wood Slabs")

dataset.plot(kind='scatter',  x='Wood Slab', y='Slab Shade', ax=axs[2], title="Shade of Wood")
# visualize the flow of the target using line-plot

plt.figure(figsize=(16, 5))

plt.plot(dataset['Wood Slab'], color='Red', label="Wood Type", linewidth=3)

plt.grid()

plt.legend()
# import the KNeighborsClassifier module

from sklearn.neighbors import KNeighborsClassifier
# instantiating KNeighborsClassifier

knn = KNeighborsClassifier()
# for splitting data into training and testing data

from sklearn.model_selection import train_test_split
# defining target variables 

target = dataset['Wood Slab']



# defining predictor variables 

features = dataset.drop(['Wood Slab'], axis=1)



# assigning the splitting of data into respective variables

X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.4, random_state=42, stratify = target)
print("X_train shape: %s" % repr(X_train.shape))

print("y_train shape: %s" % repr(y_train.shape))

print("X_test shape: %s" % repr(X_test.shape))

print("y_test shape: %s" % repr(y_test.shape))
# to display the HTML representation of an object.

from IPython.display import display_html



X_train_data = X_train.describe().style.set_table_attributes("style='display:inline'").set_caption('Summary of Training Data')

X_test_data = X_test.describe().style.set_table_attributes("style='display:inline'").set_caption('Summary of Testing Data')



# to display the summary of both training and testing data, side by side for comparison 

display_html(X_train_data._repr_html_()+"\t" +X_test_data._repr_html_(), raw=True)
# for exhaustive search over specified parameter values for an estimator

from sklearn.model_selection import GridSearchCV
# assigning the dictionary of variables whose optimium value is to be retrieved

param_grid = {'n_neighbors' : np.arange(1,50)}
# performing Grid Search CV on knn-model, using 5-cross folds for validation of each criteria

knn_cv = GridSearchCV(knn, param_grid, cv=5)
# training the model with the training data and best parameter

knn_cv.fit(X_train,y_train)
# finding out the best parameter chosen to train the model

print("The best paramter we have is: {}" .format(knn_cv.best_params_))



# finding out the best score the chosen parameter achieved

print("The best score we have achieved is: {}" .format(knn_cv.best_score_))
# predicting the values using the testing data set

y_pred = knn_cv.predict(X_test)
# example of ebony slab , given height, width and shade

prediction1=knn_cv.predict([[5.3, 3.6, 0.1]])

prediction1
# example of oak slab , given height, width and shade

prediction2=knn_cv.predict([[3.7, 8.2, 0.9]])

prediction2
# the score() method allows us to calculate the mean accuracy for the test data

knn_cv.score(X_test,y_test)
# for performance metrics

from sklearn.metrics import classification_report, confusion_matrix
# call the classification_report and print the report

print(classification_report(y_test, y_pred))
# call the confusion_matrix and print the matrix

print(confusion_matrix(y_test, y_pred))