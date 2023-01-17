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
import matplotlib.pyplot as plt

import seaborn as sns
#import the contents of the csv file into a pandas dataframe

df = pd.read_csv("/kaggle/input/iris/Iris.csv")

#check the iris dataframe to see if contents are loaded or not

df.head(5)
# check the number of rows and columns of the dataset

df.shape

# from the output below we can see that the dataframe has 150 rows(observations) and 6 columns(features)
# check for numeric and categorical columns 

df.info()

# only species column is non numeric , all other columns are numeric , so we don't have to change them
df.describe()

# count gives us the number of observations 

# we can see that sepal length and width max and min values are greater than petal length and width values
# let's explore our target variable a little more

df['Species'].value_counts()
# same thing as above but done by using groupby function

df.groupby('Species')['Id'].count()
#let's check the distribution of 'species' categories in a pie chart

df_species_series = df.groupby('Species')['Id'].count()

df_species = pd.DataFrame(df_species_series)
df_species.columns
df_species
fig1,ax1 = plt.subplots(figsize=(10,6))

ax1.pie(df_species['Id'],labels=df_species.index,autopct='%1.1f%%')

ax1.set_title("Distribution of Species in Pie Chart")

plt.tight_layout()
sns.countplot(df['Species'])
df.columns
#Let's check if there are any null values

df.isnull().sum()

#no null values
#plotting null values in heat map

sns.heatmap(df.isnull())
#let's check the correlation among different columns of the Iris DataFrame

df.corr()
sns.heatmap(df.corr(),cmap = 'viridis')
sns.pairplot(df,hue = 'Species')
sns.distplot(df['PetalLengthCm'])
sns.rugplot(df['PetalLengthCm'])
sns.kdeplot(df['PetalLengthCm'])
sns.distplot(df['SepalLengthCm'],rug=True)
sns.distplot(df['PetalWidthCm'],rug = True)
sns.distplot(df['SepalWidthCm'],rug = True)
sns.scatterplot(x = 'PetalLengthCm',y = 'PetalWidthCm', data = df, hue = 'Species')
sns.barplot(x = 'Species', y = 'PetalLengthCm', data = df)
sns.barplot(x = 'Species', y = 'PetalWidthCm', data = df)
sns.barplot(x = 'Species', y = 'SepalLengthCm', data = df)
sns.barplot(x = 'Species', y = 'SepalWidthCm', data = df)
sns.boxplot(x = 'Species', y = 'PetalLengthCm', data = df)
sns.violinplot(x = 'Species', y = 'PetalLengthCm', data = df)
sns.swarmplot(x = 'Species', y = 'PetalLengthCm', data = df)
sns.stripplot(x = 'Species', y = 'PetalLengthCm', data = df)
#divide the dataset into training and testing set

from sklearn.model_selection import train_test_split

renamed_values = {'Iris-setosa':1,'Iris-virginica':2,'Iris-versicolor':3}

X = df.drop('Species',axis = 1)

y = df['Species'].replace(renamed_values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
#let's check X and y datasets shape

X_train.shape
X_test.shape
y_train.shape
y_test.shape
y_train
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train,y_train)
# Now that we have created the model and fit our training set in it , let;s check the predictons

predictions = linear_model.predict(X_test)

predictions
# let's check if the size of the y_test dataset(actual values) and the predictions dataset (predicted values) are of the same length or not

len(predictions)
len(y_test)
from sklearn import metrics

print("Mean absolute error : " + str(metrics.mean_absolute_error(y_test,predictions)))

print('\n')

print("Mean squared error : " + str(metrics.mean_squared_error(y_test,predictions)))

print('\n')

print("Square root of mean squared error : " + str(np.sqrt(metrics.mean_squared_error(y_test,predictions))))
df_residuals = pd.DataFrame({'Actual':y_test,'Predicted':predictions})

df_residuals.head(5)
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()

logistic_model.fit(X_train,y_train)

logistic_prediction = logistic_model.predict(X_test)
df_logistic_prediction = pd.DataFrame({'Actual' : y_test,'Predicted' : logistic_prediction})

df_logistic_prediction.head(5)
# plotting the actual Vs predicted values

df_logistic_prediction.plot(kind = 'bar',figsize = (19,10))
# check the accuracy of the logistic model

print('Accuracy of the logistic model : ' + str(metrics.accuracy_score(y_test,logistic_prediction)))
print(metrics.classification_report(y_test,logistic_prediction))
print(metrics.confusion_matrix(y_test,logistic_prediction))
from sklearn.svm import SVC

SVM_model = SVC()

SVM_model.fit(X_train,y_train)

SVM_predictions = SVM_model.predict(X_test)
print('SVM classification report')

print(metrics.classification_report(y_test,SVM_predictions))
print('SVM Confusion Matrix')

print(metrics.confusion_matrix(y_test,SVM_predictions))
print('SVM model accuracy : ' + str(metrics.accuracy_score(y_test,SVM_predictions)))
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[1,0.1,0.01,0.001],'gamma':[0.1,0.01,0.001,0.0001]}

grid_model = GridSearchCV(SVC(),param_grid,verbose = 10,refit=True)

grid_model.fit(X_train,y_train)
grid_model.best_estimator_
grid_model.best_params_
grid_predictions = grid_model.predict(X_test)
print('GridSearch SVM Accuracy : ' + str(metrics.accuracy_score(y_test,grid_predictions)))

print("GridSearch SVM classification report : " + metrics.classification_report(y_test,grid_predictions))

print("GridSearch SVM confusion matrix : " + str(metrics.confusion_matrix(y_test,grid_predictions)))
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_train,y_train)

decision_tree_predictions = decision_tree_model.predict(X_test)

print("Decision tree accuracy : " + str(metrics.accuracy_score(y_test,decision_tree_predictions)) + "\n")

print("Decision tree classification report : " + metrics.classification_report(y_test,decision_tree_predictions))

print("Decision tree confusion matrix : " + str(metrics.confusion_matrix(y_test,decision_tree_predictions)))
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors=1)

KNN_model.fit(X_train,y_train)

KNN_predictions = KNN_model.predict(X_test)

print("KNN model accuracy : " + str(metrics.accuracy_score(y_test,KNN_predictions)))

print("KNN classification report : " + metrics.classification_report(y_test,KNN_predictions))

print("KNN confusion matrix : " + "\n" + str(metrics.confusion_matrix(y_test,KNN_predictions)))