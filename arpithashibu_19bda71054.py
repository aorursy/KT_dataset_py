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
#importing the necessary library for preprocessing

from sklearn import preprocessing

#used for train test split

from sklearn.model_selection import train_test_split,cross_validate

#importing libraries for visualization

import seaborn as sns

from sklearn.impute import SimpleImputer # used for handling missing data

#loading the train and test dataset

data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")

Test=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")

sample=pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
#to view the first 5 rows i.e, to get an overview

data.head()
#to get the basic info about the data types of various features

data.info()
#to get the description of the data

data.describe()
#to check whether there are any missing values

data.count().isnull()
#using pairplot to find how each features are depended

#sns.pairplot(data)
#sns.pairplot(data,hue="flag")
"""scaler=StandardScaler()

data1=scaler.fit_transform(data)

data1=pd.DataFrame(data1)

data1"""
#to draw a boxplot to detect ouliers

sns.boxplot(data["velocityFront"])
#to get all column names

data.columns
#finding the q1,q3,min,max range for each column

q1_positionBack=np.quantile(data['positionBack'],0.25)

q3_positionBack=np.quantile(data['positionBack'],0.75)

min_positionBack=q1_positionBack-(1.5*(q3_positionBack - q1_positionBack))

max_positionBack=q3_positionBack+(1.5*(q3_positionBack - q1_positionBack))
#replacing with nan for the values that lie less than min value

data.loc[data['positionBack']<min_positionBack,'positionBack']=np.nan

data['positionBack'].fillna(np.array(data['positionBack'].quantile(0.05)),inplace=True)
#replacing with nan for the values that lie beyond than max value

data.loc[data['positionBack']>max_positionBack,'positionBack']=np.nan

data['positionBack'].fillna(np.array(data['positionBack'].quantile(0.95)),inplace=True)
#to get the boxplot

sns.boxplot(data["positionBack"])
#finding the q1,q3,min,max range for each column

q1_refPositionFront=np.quantile(data['refPositionFront'],0.25)

q3_refPositionFront=np.quantile(data['refPositionFront'],0.75)

min_refPositionFront=q1_refPositionFront-(1.5*(q3_refPositionFront - q1_refPositionFront))

max_refPositionFront=q3_refPositionFront+(1.5*(q3_refPositionFront - q1_refPositionFront))
#replacing with nan for the values that lie less than min value

data.loc[data['refPositionFront']<min_refPositionFront,'refPositionFront']=np.nan

data['refPositionFront'].fillna(np.array(data['refPositionFront'].quantile(0.05)),inplace=True)

#data['refPositionFront']
##replacing with nan for the values that lie beyond than max value

data.loc[data['refPositionFront']>max_refPositionFront,'refPositionFront']=np.nan

data['refPositionFront'].fillna(np.array(data['refPositionFront'].quantile(0.95)),inplace=True)

data['refPositionFront']
sns.boxplot(data["refPositionFront"])
q1_refVelocityBack=np.quantile(data['refVelocityBack'],0.25)

q3_refVelocityBack=np.quantile(data['refVelocityBack'],0.75)

min_refVelocityBack=q1_refVelocityBack-(1.5*(q3_refVelocityBack - q1_refVelocityBack))

max_refVelocityBack=q3_refVelocityBack+(1.5*(q3_refVelocityBack - q1_refVelocityBack))
data.loc[data['refVelocityBack']<min_refVelocityBack,"refVelocityBack"]=np.nan

data['refVelocityBack'].fillna(np.array(data['refVelocityBack'].quantile(0.05)),inplace=True)

data['refVelocityBack']
data.loc[data['refVelocityBack']>max_refVelocityBack,"refVelocityBack"]=np.nan

data['refVelocityBack'].fillna(np.array(data['refVelocityBack'].quantile(0.95)),inplace=True)

data['refVelocityBack']
sns.boxplot(data["refVelocityBack"])
q1_currentFront=np.quantile(data['currentFront'],0.25)

q3_currentFront=np.quantile(data['currentFront'],0.75)

min_currentFront=q1_currentFront-(1.5*(q3_currentFront - q1_currentFront))

max_currentFront=q3_currentFront+(1.5*(q3_currentFront - q1_currentFront))
data.loc[data['currentFront']<min_currentFront,'currentFront']=np.nan

data['currentFront'].fillna(np.array(data['currentFront'].quantile(0.05)),inplace=True)

data['currentFront']
data.loc[data['currentFront']>max_currentFront,'currentFront']=np.nan

data['currentFront'].fillna(np.array(data['currentFront'].quantile(0.95)),inplace=True)

data['currentFront']
sns.boxplot(data["currentFront"])
#to draw a boxplot to detect ouliers

sns.boxplot(data["positionBack"])
q1_currentBack=np.quantile(data['currentBack'],0.25)

q3_currentBack=np.quantile(data['currentBack'],0.75)

min_currentBack=q1_currentBack-(1.5*(q3_currentBack - q1_currentBack))

max_currentBack=q3_currentBack+(1.5*(q3_currentBack - q1_currentBack))
data.loc[data['currentBack']<min_currentBack,'currentBack']=np.nan

data['currentBack'].fillna(np.array(data['currentBack'].quantile(0.05)),inplace=True)
data.loc[data['currentBack']>max_currentBack,'currentBack']=np.nan

data['currentBack'].fillna(np.array(data['currentBack'].quantile(0.95)),inplace=True)
sns.boxplot(data['currentBack'])
data.corr()
sns.heatmap(data.corr())
"""data['velocityBack']=np.log(data['velocityBack'])

data['motorTempBack']=np.log(data['motorTempBack'])

data['refVelocityBack']=np.log(data['refVelocityBack'])

data['motorTempFront']=np.log(data['motorTempFront'])

data['refVelocityFront']=np.log(data['refVelocityFront'])

data['velocityFront']=np.log(data['velocityFront'])

data['trackingDeviationFront']=np.log(data['trackingDeviationFront'])"""
"""from sklearn.preprocessing import Normalizer

scaler=Normalizer().fit(data)

Test=scaler.transform(data)"""

#splitting the dataset into indept variables

X=data.drop(["flag","timeindex","positionBack","refPositionBack","positionFront","refPositionFront"],axis=1)

#X.head()

#X=data.drop(["flag","timeindex"],axis=1)

#filtering the dependent variable

Y=data[["flag"]]

#Y.head()
#to get the first 5 values

X.head()
#for train, test splitting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#importing the necessary library for classification

#from sklearn.neighbors import KNeighborsClassifier
#fixing the no of neighbourers and assigning to a variable

#knn = KNeighborsClassifier(n_neighbors=5) 
#training our model on traning dataset

#knn.fit(X_train, Y_train) 
# Calculate the accuracy of the model 

#print(knn.score(X_test, Y_test)) 
#Test.head()
#dropping the timeindex column from test dataset

Test=Test.drop(["timeindex","positionBack","refPositionBack","positionFront","refPositionFront"],axis=1)

#Test=Test.drop(["timeindex"],axis=1)
#applying log transformation on the test dataset

"""Test['velocityBack']=np.log(Test['velocityBack'])

Test['motorTempBack']=np.log(Test['motorTempBack'])

Test['refVelocityBack']=np.log(Test['refVelocityBack'])

Test['motorTempFront']=np.log(Test['motorTempFront'])

Test['refVelocityFront']=np.log(Test['refVelocityFront'])

Test['velocityFront']=np.log(Test['velocityFront'])

Test['trackingDeviationFront']=np.log(Test['trackingDeviationFront'])"""
#predicting the value for test dataset

#pred=knn.predict(Test)
#inserting the predicted value to the sample file

#sample['flag']=pred
#sample.head()
#writing it to a csv format

#sample.to_csv('Sample Submission3.csv',index=False)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=2, random_state=0)



# Train the Classifier to take the training features and learn how they relate

# to the training y (the species)

clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test)) 
#predicting the value for test dataset

pred=clf.predict(Test)
#inserting the predicted value to the sample file

sample['flag']=pred
sample.head()
#writing it to a csv format

sample.to_csv('Sample Submission9.csv',index=False)
