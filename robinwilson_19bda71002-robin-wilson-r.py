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
#importing necessary libraries
import pandas as pd # used for DataFrame, series operations
import numpy as np # used for n dimension array
import matplotlib.pyplot as plt #plotting the graph
%matplotlib inline
import seaborn as sns # visualizing the data
#loading train and test data
train_data=pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv") # train data
test_data=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv") # test data
#displaying first 5 rows of train data
train_data.head()
#information of train data
train_data.info()
# shape of train data
train_data.shape
#viewing columns in train data
train_data.columns

#statistical description of a data
train_data.describe()
#checking for null values
train_data.isnull().sum()
#finding missing percentage of null values
missing=pd.DataFrame({'total':train_data.isnull().sum(),'percentage':(train_data.isnull().sum()/11227)*100})
missing
#plotting heatmap to chech any missing values
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)
#finding total null values
train_data.isnull().sum().sum()
#plotting a boxplot for outlier detection
columns=train_data.columns
plt.figure(figsize=(20,5))
train_data[columns].boxplot()
plt.title("All numerical variables")
plt.show()
# checking for correlation
train_data.corr()

# Now its time to build a model so I need to import necessary packages and split the data
from sklearn.neighbors import KNeighborsClassifier # importing Knn classifier
from sklearn.model_selection import train_test_split # splitting the data using train_test_split
# divided or data taken as dependent and independent
train_x=train_data.drop(columns=['flag'],axis=1) # independent variables
train_y=train_data["flag"] # dependent variable
# shape of the train_x and train_y
train_x.shape, train_y.shape
#splitted the data as train and test
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2231,random_state=1)
#shape of splitted data
x_train.shape, x_test.shape ,y_train.shape,y_test.shape
# fitting KNN model
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_x,train_y)
# predicted the model
predict=classifier.predict(x_test)
print(predict)
# imported some metrics for score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
# the result is
result=confusion_matrix(y_test,predict)
print("confusion matrix is:",result)
result1=classification_report(y_test,predict)
print("report is:",result1)
result2=accuracy_score(y_test,predict)
print("accuracy:", result2)
#accuracy
accuracy=accuracy_score(y_test,predict)
print("accuracy:",accuracy)
#f1_score
score=f1_score(y_test,predict)
print("F1_score:",score)
# converting to dataframe of train data
data=pd.DataFrame(predict,columns=["flag"])
data.head()
# predicted for test data
predict2=classifier.predict(test_data)
print(predict2)
# printing classification report for test data
print("Report is:",classification_report(y_test,predict2))
# converting test as dataframe
data2=pd.DataFrame(predict2,columns=['flag'])
data2.head()
# data2.to_csv("final.csv",header=['flag'],index=False)
# loading sample data
sample=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
sample.head()
#finally added sample data with predicted value
# sample["predict"]=data2
# sample.head()
# sample=sample.drop(columns='Sl.No')
#converting to csv file
submission=pd.DataFrame({
    'Sl.No':sample['Sl.No'],
    'flag':predict2
})

submission.to_csv("final8.csv",index=False)
# #final shape of sample
# sample.shape
submission=pd.DataFrame({
    'Sl.No':sample['Sl.No'],
    'flag':predict2
})

submission.to_csv("final3.csv",index=False)
# # converting to csv file
# sample.to_csv('submit final1.csv',index=False)
#splitting the data
x=train_data[[ 'currentBack', 'motorTempBack', 'positionBack',
       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',
       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',
       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',
       'velocityFront']]
y=train_data[['flag']]
#splitting test data
test=test_data[['currentBack', 'motorTempBack', 'positionBack',
       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',
       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',
       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',
       'velocityFront']]
#importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
d_train,d_test,n_train,n_test=train_test_split(x,y,test_size=0.2231,random_state=1)# splitting the data
#fitting a model
classi=DecisionTreeClassifier()
classi.fit(d_train,n_train)
#predicting
d_pred=classi.predict(d_test)
print(d_pred)
#prediction for test data
dpred1=classi.predict(test)
dpred1
#importing metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
confmat=confusion_matrix(n_test,d_pred)
recall=recall_score(n_test,d_pred)
precision=precision_score(n_test,d_pred)
accuracy=accuracy_score(n_test,d_pred)
f1=f1_score(n_test,d_pred)
print("confusion matrix:",confmat)
print("precision is:",precision)
print("recall is:",recall)
print("f1_score is:", f1)
submission=pd.DataFrame({
    'Sl.No':sample['Sl.No'],
    'flag':dpred1
})

submission.to_csv("final9.csv",index=False)
#from sklearn.ensemble import RandomForestClassifier          #library to run random forest model
