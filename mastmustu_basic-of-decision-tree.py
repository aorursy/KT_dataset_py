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
# read the data 

df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
#top 5 record of the data 

df.head()
# columns

df.columns
# describe  the data 



df.describe()



# only works on numerical cols

# statistics about data 

# data distributions
# Information about the data in a glance



df.info()
# Dimension  i.e. rows and columns



df.shape
#Unique values is Species 

df['species'].unique()
# count of the different species



df['species'].value_counts()
# Features V/s Target 



features  = list(df.columns)[:-1]

print(features)

target = list(df.columns)[-1:][0]

print(target)



# Features  --- > Predictor 

# Target  --> Predicted 
# Data Visualisation Step





import seaborn as sns

sns.pairplot(df,hue ='species')



# hue - color based on a column name  - in most case it will be Target Columns

# Separates features and corresponding labels/target 

# by dropping Species - we get data frame with all features 



X = df.drop(['species'], axis=1)  #  X will hold all features

y = df['species'] # y will hold target/labels



print(X.shape) #dimensions of input data

print(y.shape) #dimensions of output data





# is this binary classification or not  ?


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.5, random_state = 0) 

print(X_train.shape)

print(X_test.shape)
# Training using Decision Tree Classifier 



from sklearn.tree import DecisionTreeClassifier  

classifier1 = DecisionTreeClassifier(criterion='gini')  

classifier1.fit(X_train, y_train) 



# Check Criteria  ?

print(classifier1)
# Using information gain 



classifier2 = DecisionTreeClassifier(criterion='entropy')  

classifier2.fit(X_train, y_train) 
# predict using both the classifier 







y_pred_1 = classifier1.predict(X_test)  

print(y_pred_1)



y_pred_2 = classifier2.predict(X_test)  

print(y_pred_2)
# compute accuracy 



from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc_1 = accuracy_score(y_test,y_pred_1)

print("Accuracy for Gini model {} %".format(acc_1*100))





acc_2 = accuracy_score(y_test,y_pred_2)

print("Accuracy for Entropy model {} %".format(acc_2*100))
# confusion matrix for Gini Model 





from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred_1))

print(classification_report(y_test, y_pred_1)) 
# important features 



print(classifier1.feature_importances_)



#SepalLenCm   SepalWidCm    PetalLenCm     PetalWidCm

# Plot a tree 





from sklearn import tree

from sklearn.tree import export_graphviz



tree.export_graphviz(classifier2,out_file='tree.dot',feature_names = ['SepalLenCm','SepalWidCm','PetalLenCm', 'PetalWidCm'],

class_names = 'Species',rounded = True, proportion = False, precision = 2, filled = True)  



!dot -Tpng tree.dot -o tree.png

from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.neighbors import KNeighborsClassifier





classifier3 = KNeighborsClassifier(n_neighbors= 7)  

classifier3.fit(X_train, y_train) 
y_pred_3 = classifier3.predict(X_test)  

print(y_pred_3)

acc_3 = accuracy_score(y_test,y_pred_3)

print("Accuracy for Entropy model {} %".format(acc_3*100))

# confusion matrix for Gini Model 





from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred_3))

print(classification_report(y_test, y_pred_3)) 