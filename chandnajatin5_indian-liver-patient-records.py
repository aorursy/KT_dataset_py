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
#Opening the Dataset

LiverDataset = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
#Checking the datatypes of all the columns
LiverDataset.dtypes

#checking the null values
a=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio', 'Dataset']
LiverDataset.isnull().sum()
#There are 4 null values in the column Albumin_and_Globulin_Ratio
#checking the shape of the dataset
LiverDataset.shape
#checking the output Imbalaning
print("Liver Patients",LiverDataset['Dataset'][LiverDataset['Dataset']==1].count())
print("Non Liver Patients",LiverDataset['Dataset'][LiverDataset['Dataset']==2].count())
#Plotting the imbalance
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(LiverDataset['Dataset'])
plt.show()
LiverDatset_duplicate = LiverDataset[LiverDataset.duplicated(keep=False)]
#Dropping all the duplicates present in the dataset
LiverDataset = LiverDataset.drop_duplicates()
#Checking the Duplicates
LiverDataset[LiverDataset.duplicated()]
#Extracting the sum of the Null Values
LiverDataset.isnull().sum()

#Visualising the Null values
sns.heatmap(LiverDataset.isnull(),yticklabels=False,cbar=False)
#Trying to Explore Age Group to enter null values
sns.distplot(LiverDataset['Age'])
#Visualising the categorical data with the Output
sns.countplot(x='Dataset',hue='Gender',data=LiverDataset)
#Visualizing the distribution of the columns present in the 
sns.distplot(LiverDataset['Albumin_and_Globulin_Ratio'])
#Visualizing the coumn containing null values with the categorical data
plt.figure(figsize=(12,7))
sns.boxplot(x='Gender',y='Albumin_and_Globulin_Ratio',data=LiverDataset)
LiverDataset.columns
#Replacing the Female Null values with 0.9 & Males null value with 1.0
#Impute the Missing Values
def impute_null(cols):
    Gender = cols[0]
    Albumin_and_Globulin_Ratio = cols[1]
    if pd.isnull(Albumin_and_Globulin_Ratio):
        if Gender == 'Male':
            return 0.9
        else:
            return 1.0
    else:
        return Albumin_and_Globulin_Ratio
        
    
    
    
    
#Executing the above function and imputing the above null values
LiverDataset['Albumin_and_Globulin_Ratio']=LiverDataset[['Gender','Albumin_and_Globulin_Ratio']].apply(impute_null,axis=1)
#All the null values are being removed
LiverDataset.isnull().sum()
#Determining the categorical data
LiverDataset.info()
#Applying the get dummies method for the Gender column
Gender = pd.get_dummies(LiverDataset['Gender'])

#Dropping the Gender & Dataset column from the main dataset
LiverDataset.drop(['Gender'],axis=1,inplace=True)
LiverDataset
X=LiverDataset.drop(['Dataset'],axis=1)
Y=LiverDataset['Dataset']
#Concatenating the two Dataframes to add the get dummies value in the gender column
LiverDataset = pd.concat([LiverDataset,Gender],axis=1)
LiverDataset
#Performing the Feature scaling using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x=X.values
LiverDataset_scaled = scaler.fit_transform(X)
LiverDataset_scaled
LiverDataset = pd.DataFrame(LiverDataset_scaled)
x=LiverDataset
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(sampling_strategy=1)
X.shape,Y.shape
X,Y = os.fit_sample(X,Y)
X.shape,Y.shape
Y
#Dividing the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1000)
from sklearn.linear_model import LogisticRegression
# Create logistic regression object

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
log_predicted= logreg.predict(X_test)
log_predicted
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
print('Accuracy: \n', accuracy_score(Y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(Y_test,log_predicted))
print('Classification Report: \n', classification_report(Y_test,log_predicted))
sns.heatmap(confusion_matrix(Y_test,log_predicted),annot=True,fmt='d')
from sklearn.tree import DecisionTreeClassifier
# Create decision tree object

dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
log_predicted=dt.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
print('Accuracy: \n', accuracy_score(Y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(Y_test,log_predicted))
print('Classification Report: \n', classification_report(Y_test,log_predicted))
sns.heatmap(confusion_matrix(Y_test,log_predicted),annot=True,fmt='d')
from sklearn.ensemble import RandomForestClassifier
#creating random forest object
random_forest = RandomForestClassifier()
random_forest.fit(X_train,Y_train)
log_predicted=random_forest.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
print('Accuracy: \n', accuracy_score(Y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(Y_test,log_predicted))
print('Classification Report: \n', classification_report(Y_test,log_predicted))
