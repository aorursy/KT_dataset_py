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
#importing packages
import pandas as pd #datapreprocessing , csv file uploading
import numpy as np #linear algebra
import seaborn as sns #for plots
import matplotlib.pyplot as plt #for plot
%matplotlib inline
#loading train and test datas
train = pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
test = pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#displaying first 5 rows of train data
train.head()
#displaying the size
train.shape
#assigning the dependent variable to target variable
target = train["flag"]
#checking number of each values
train["flag"].value_counts()
#displaying first 5 rows of test data
test.head()
#checking missing values
train.isnull().sum()
#train data information
train.info()
#scatter plot showing the distribution
plt.scatter(train["flag"],train["currentBack"])
#boxplot for showing outliers
sns.boxplot(train['motorTempBack'])
#calculating quantiles
Q1 = np.array(train["motorTempBack"].quantile(0.25))
Q3 = np.array(train["motorTempBack"].quantile(0.75))
#Inter quartile range
IQR = Q3 - Q1
#finding the lower whisker
min_limit = Q1 - (1.5*IQR)
print("miniimum=",min_limit)
#removing extreme values to na
train.loc[train['motorTempBack'] < 31,'motorTempBack']=np.nan
#checking number of na values
train['motorTempBack'].isnull().sum()
#filling na values
na_values = np.array(train["motorTempBack"].quantile(0.05))
na_values
train['motorTempBack'].fillna(na_values ,inplace=True)
#applying log  transformation
train["positionBack"] = np.log(train["positionBack"])
#applying log  transformation
train["refPositionBack"] = np.log(train["refPositionBack"])
#applying log  transformation
train["refVelocityBack"] = np.log(train["refVelocityBack"])
#applying log  transformation
train["trackingDeviationBack"] = np.log(train["trackingDeviationBack"])
#applying log  transformation
train["velocityBack"] = np.log(train["velocityBack"])
#scatter plot shwoing the distribution
plt.scatter(train["flag"],train["currentFront"])
##boxplot for showing outliers
sns.boxplot(train["motorTempFront"])
#calculating quantiles
Q1 = np.array(train["motorTempFront"].quantile(0.25))
Q3 = np.array(train["motorTempFront"].quantile(0.75))
#Inter quartile range
IQR = Q3 - Q1
#finding the lower whisker
min_limit = Q1 - (1.5*IQR)
print("minimum=",min_limit)
#removing extreme values to na
train.loc[train['motorTempFront'] <34 ,'motorTempFront']=np.nan
#checking number of na values
train['motorTempFront'].isnull().sum()
#filling na values
na_values = np.array(train["motorTempFront"].quantile(0.05))
na_values
train['motorTempFront'].fillna(na_values ,inplace=True)
#applying log  transformation
train["positionFront"] = np.log(train["positionFront"])
#applying log  transformation
train["refPositionFront"] = np.log(train["refPositionFront"])
#applying log  transformation
train["refVelocityFront"] = np.log(train["refVelocityFront"])
#applying log  transformation
train["trackingDeviationFront"] = np.log(train["trackingDeviationFront"])
#applying log  transformation
train["velocityFront"] = np.log(train["velocityFront"])
#Initializing X and y
X = train.drop(["timeindex","flag"],axis = 1)
y = target
#spliting X and y into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#fitting and predicting the model
from sklearn.ensemble import RandomForestClassifier

cl = RandomForestClassifier(n_estimators=70, random_state=0)
cl.fit(X_train, y_train)
y_pred = cl.predict(X_test)
#checking the accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred))
#droping the same columns which were dropped from the train data
test = test.drop(["timeindex"],axis = 1)
#applying log transformation
test["positionBack"] = np.log(test["positionBack"])
test["refPositionBack"] = np.log(test["refPositionBack"])
test["refVelocityBack"] = np.log(test["refVelocityBack"])
test["trackingDeviationBack"] = np.log(test["trackingDeviationBack"])
test["velocityBack"] = np.log(test["velocityBack"])
test["positionFront"] = np.log(test["positionFront"])
test["refPositionFront"] = np.log(test["refPositionFront"])
test["refVelocityFront"] = np.log(test["refVelocityFront"])
test["trackingDeviationFront"] = np.log(test["trackingDeviationFront"])
test["velocityFront"] = np.log(test["velocityFront"])
#predicting the test data
pred = cl.predict(test)
#loading sample submission.csv
sample = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
#assigning predicted values of test data to sample submission data 
sample["flag"] = pred
#displaying last five rows
sample.head()
#converting the sample submission file to csv
sample.to_csv("submit_11.csv",index=False)