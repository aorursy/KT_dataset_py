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
'''

Approach:

.

    Data:

        Train data has 11227 rows and 16 columns(including the dependent variable) . Test data  2505 rows and 15 columns.

        

    Preprocessing: 

        The data is read and checked for missing values. Then, the outliers in the data is found. Those outliers less than

    10th percentile is replaced by 10th percentile. And those outliers greater than 90th percentile are replaced by 90th 

    percentile. Then, logarithmic transformation is applied to standardize the data. The data is then split into test and

    train sets, each of them having features and labels.

    

    Fitting the model:

        Random Forest classifier is used which was trained using the training set and validated using the test set. The 

    perfomance measures are also given. Random Forest classifier is an ensemble tree-based algorithms. It is based on several descision trees that takes 

    randomly selected subset of the training set. The votes from these trees are aggregated to predict the final class.

    

    Model f1 score: 0.84

        

    Predictions for the test set:

        Then the data for testing was read and preprocessed similar to the trained set. It was then used to predict its labels

    which are saved to Sample Submission 1.csv.

    

'''
#Reading the training set



data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
#prints the first 5 rows of the training set



print(data.shape)

data.head()
#checks for missing values



def check_missing(data):

    print(data.isnull().sum())

check_missing(data)
#splitting into Feature set and labels



X=data.drop('flag',axis=1)

y=data['flag']
#plots histogram for all the features that shows the distribution of the data



def check_distribution(X):

    pd.DataFrame.hist(X, figsize=(15,15))

check_distribution(X)
#boxplot for each of the features to show the outliers



def plot_boxplot(data):

    for i in range(data.shape[1]):

        if data.columns[i]!="flag":

            data.boxplot(column=data.columns[i], by="flag", figsize=(5,5))

plot_boxplot(data)
#Outliers below 10th percentile are replaced by the 10th percentile

#Outliers above 90th percentile are replaced by the 90th percentile



def manage_outliers(X):

    for i in X.columns:

        low_lim=X[i].quantile(.10)

        upper_lim=X[i].quantile(.90)

        X[i]=np.where(X[i]<low_lim, low_lim, X[i])

        X[i]=np.where(X[i]>upper_lim, upper_lim, X[i])

    return X

X=manage_outliers(X)
#Using log transformation to convert the data to a normal distribution



def log_transform(X):

    for i in X.columns:

        X[i]=X[i].map(lambda i: np.log(i) if i > 0 else 0)

    return X

X=log_transform(X)
#checking the distribution after the log transform



check_distribution(X)
#final form of the feature set and labels



print("Feature set\n",X)

print("Labels\n",y)
#The dataset for training is split again into test and train



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=10)
#Random Forest Classifier is used where the n_estimators is 40. It indicates the number of trees to model.



from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=60, criterion='entropy',random_state=0)

model.fit(X_train,Y_train)
from sklearn.metrics import classification_report



#predictions are made using test set

y_actual,y_pred=Y_test,model.predict(X_test)



#Accuracy of the model w.r.t trained set

print(model.score(X_train,Y_train))



#Accuracy of the model w.r.t test set

print(model.score(X_test,Y_test))



#lists the precision, recall and f1 score

print(classification_report(y_actual,y_pred))
#confusion matrix



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_actual,y_pred)

cm
#A better representation of the confusion matrix



import seaborn as sb

import matplotlib.pyplot as plt

plt.figure(figsize=(3,3))

sb.heatmap(cm,annot=True)

plt.xlabel("Predicted")

plt.ylabel("Actual")
#Reading the training set



test_data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#prints first 5 rows of the test set



print(test_data.shape)

test_data.head()
#checking the distribution of data



check_distribution(test_data)
#checks for missing values



check_missing(test_data)
#Outliers are replaced similar to the trained set



manage_outliers(test_data)
#normalizing the data using logarithmic transformation



log_transform(test_data)
#checking the distribution of data after transofrmation



check_distribution(test_data)
#final form of the feature set 



print("Feature set\n",test_data)
#predictions of labels are made



predict=model.predict(test_data)
#submitting predictions



submission=pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv",index_col=False)

submission['flag']=predict
#final submission

submission
#final submission is made in Sample_Submission_RF.csv



submission.to_csv("Sample_Submission_RF.csv",index=False)