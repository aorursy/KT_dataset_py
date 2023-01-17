# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

df.head()
df.info()
df.columns
df.drop(columns=['Unnamed: 32','id'],inplace=True)  #dropping the columns that are not required at all

df.columns
df.shape
df['diagnosis'].unique()
# Describing the head of the loaded dataframe 

df.describe()
X = df.iloc[:,1:]

y = df.iloc[:,0]
X.shape
y.shape
# Benign Vs Malignant

value_count = df['diagnosis'].value_counts()  

plt.figure()



# Plotting the Count for the value counts in the diagnosis column

value_count.plot(kind = "bar", color = "blue", rot=0)

plt.ylabel("Counts")

plt.title("A bar chart showing the count of Benign Vs Malignant Labels")

plt.grid(True)

plt.show()
# Displaying a violin plot for all the features 

labels = y 

input_features = X



# Normalizing the dataframe 

data = (input_features - input_features.mean()) / (input_features.std())

input_features = pd.concat([y, data.iloc[:,:]], axis = 1)



data = pd.melt(input_features, id_vars = "diagnosis", var_name = "features", 

              value_name = "value")



# Plotting the first Ten feature

plt.figure(figsize = (18, 7))

sns.violinplot(x = "features", y = "value", hue = "diagnosis", data = data, split = True, 

              inner = "quart")

plt.xticks(rotation = 90)

plt.grid(True)

plt.show()
# Displaying a Correlation matrix of all the features of the breast cancer dataset.

f,ax = plt.subplots(figsize=(20, 18))

sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Setting the figure of the Graph 

plt.figure(figsize=(18, 7))



# Plotting a scatter plot of diagnosis against radius_mean

plt.scatter(df['diagnosis'], df['radius_mean'])



# Displaying the graph 

plt.show()
# Plotting a boxplot of the "Area_mean","area_se" and "area_worst"

boxplot = df.boxplot(column = ["area_mean","area_se","area_worst"], by="diagnosis", 

                    layout = (3, 1), figsize=(19, 9))

# Showing the boxplot

plt.show()
# Plotting a boxplot of the means of different features

boxplot = df.boxplot(column = ["area_mean","perimeter_mean","compactness_mean"], by="diagnosis", 

                    layout = (3, 1), figsize=(19, 9))

# Showing the boxplot

plt.show()
# Plotting a boxplot of the means of different features

boxplot = df.boxplot(column = ["concave points_mean","concavity_mean","smoothness_mean"], by="diagnosis", 

                    layout = (3, 1), figsize=(19, 9))

# Showing the boxplot

plt.show()
# Plotting a boxplot of the means of different features

boxplot = df.boxplot(column = ["fractal_dimension_mean","radius_mean","symmetry_mean","texture_mean"], by="diagnosis", 

                    layout = (4, 1), figsize=(19, 9))

# Showing the boxplot

plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
# Creating scaled set to be used in model to improve the results

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Import Library of Support Vector Machine model

from sklearn import svm

from sklearn.model_selection import GridSearchCV



# Create a Support Vector Classifier

svc = svm.SVC()



# Hyperparameter Optimization

parameters = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

]



# Run the grid search

grid_obj = GridSearchCV(svc, parameters)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the svc to the best combination of parameters

svc = grid_obj.best_estimator_



# Train the model using the training sets 

svc.fit(X_train,y_train)

# Prediction on test data

y_pred = svc.predict(X_test)
# Calculating the accuracy

from sklearn.metrics import accuracy_score

acc_svm = round( accuracy_score(y_test, y_pred) * 100)

print( 'Accuracy of SVM model : ', acc_svm )