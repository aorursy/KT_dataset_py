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
#import libraries for KNN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#import libraries for pandas profiling
# Detailing for pandas profiling: https://github.com/pandas-profiling/pandas-profiling
from pathlib import Path
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
#The dataset provides the patients’ information
df = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
df.info(verbose = True)
df.describe().T

p = df.hist(figsize=(20,20))
#scatter matrix
from pandas.plotting import scatter_matrix
p=scatter_matrix(df,figsize=(25, 25))
# Pair plot
p=sns.pairplot(df, hue = 'class')
#Profiling: we can see our dataset details,simply.
profile = ProfileReport(df, title="biomechanical-features-of-orthopedic-patients/column_3C_weka")
# The Notebook Widgets Interface
profile.to_widgets()

#Correlation-heatmap
plt.figure(figsize=(12,10)) 
p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn') 
df.head()
y = df["class"].values
X = df.drop("class",axis=1).values
#Train Test Split data==> 80% of data set for Train, 20% of data set for Test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#Setup arrays to store train and test accuracies
neighbors = np.arange(1,11)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i,k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k, metric= "manhattan")
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train,y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test,y_test)
    
# Generate plot
plt.title("Biomechanical features of orthopedic patients")
plt.plot(neighbors,train_accuracy,label="Train Accuracy")
plt.plot(neighbors,test_accuracy,label="Test Accuracy")
plt.legend()
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.show()
# See details for best prediction
knn = KNeighborsClassifier(n_neighbors=8, metric= "manhattan")
#fitting
knn.fit(X_train,y_train)
#prediction

print("Prediction of features (test set): ",knn.predict(X_test))
print("Actual label variables (test set): ",y_test)
#ACCURACY
print("Train Accuracy Score: ",knn.score(X_train,y_train))
print("Test Accuracy Score",knn.score(X_test, y_test))
#import confusion matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_predict = knn.predict(X_test)
confusion_matrix(y_test,y_predict)
pd.crosstab(y_test,y_predict,rownames=["True"],colnames=["Predicted"],margins=True)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))