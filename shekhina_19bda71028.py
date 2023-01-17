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
#Reading the train, test and sample csv files 

train = pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')

test = pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')

sample = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
#Displaying the first 5 records of the train dataset

train.head()
#Displaying the descriptive statistics of the training set

train.describe()
#Separating the train data into its features (X) and labels (y)

X = train.drop('flag', axis=1)

y = train['flag']
#Displaying the number of missing values in each column of X.

X.isnull().sum()
#Displaying the number of missing values in each column of the test data.

test.isnull().sum()
#Displaying the information of the feature set X, such as column data types, non-null values and memory usage.

X.info()
#Importing the seaborn library for data visualization

import seaborn as sns
#Distribution plot of timeindex

sns.distplot(X['timeindex'])
#Distribution plot of currentBack

sns.distplot(X['currentBack'])
#Distribution plot of motorTempBack

sns.distplot(X['motorTempBack'])
#Distribution plot of positionBack

sns.distplot(X['positionBack'])
#Distribution plot of refPositionBack

sns.distplot(X['refPositionBack'])
#Distribution plot of refVelocityBack

sns.distplot(X['refVelocityBack'])
#Distribution plot of currentFront

sns.distplot(X['currentFront'])
#Distribution plot of motorTempFront

sns.distplot(X['motorTempFront'])
#Distribution plot of positionFront

sns.distplot(X['positionFront'])
#Distribution plot of refPositionFront

sns.distplot(X['refPositionFront'])
cor_matrix = train.corr()    #To obtain the correlation matrix

sns.heatmap(cor_matrix)      #To obtain the heatmap of the correlaation matrix
#Correlation between each feature and the target variable

corr = X.corrwith(y)

cor_df = pd.DataFrame({'Col_names': corr.index, 'Correlation': abs(corr.values)}).sort_values(by = 'Correlation', ascending = False)

cor_df
#Finding the positive-valued columns to apply log transormations

pos_col = X[X>0]

pos = pos_col.loc[:,pos_col.columns[pos_col.isnull().sum()==0]]
#Finding the skewness of each feature

from scipy.stats import skew 

skewness = pos.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
#Finding features with skewness greater than 0.5

skewed_feature = skewness[abs(skewness) > 0.5].index



#Applying log transformations to the skewed features

trans_feature = np.log1p(X.loc[:, skewed_feature])

X.loc[:, trans_feature.columns] = trans_feature
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)           #Standardizing train features

X_scaled_test = StandardScaler().fit_transform(test)   #Standardizing test features
#Splitting into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 4, stratify = y)
#Importing the matplotlib library to plot graphs

import matplotlib.pyplot as plt
#Bar plot showing class imbalance

plt.bar(y_train.value_counts().index, y_train.value_counts().values)
#Correcting class imbalance by over-sampling using SMOTE

import tensorflow

from imblearn import under_sampling, over_sampling

from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_smot, y_smot = SMOTE().fit_sample(X_train, y_train)
#Bar plot showing correction of class imbalance

plt.bar(y_smot.value_counts().index, y_smot.value_counts().values)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



#Fitting Logistic Regression model

log = LogisticRegression().fit(X_smot, y_smot)
#Predicting class labels using the trained logit regression model

y_hat1 = log.predict(X_test)
#Finding the F1-score of the logit regression model

f1 = f1_score(y_test, y_hat1)

f1
from sklearn.ensemble import RandomForestClassifier



#Fitting Random Forest Classifier

rf = RandomForestClassifier().fit(X_smot ,y_smot)
#Predicting class labels using the trained Random Forest model

y_hat2 = rf.predict(X_test)
#Finding the F1-score of the Random Forest model

f2 = f1_score(y_test, y_hat2)

f2
#Finding Feature Importance

FI = rf.feature_importances_

df = pd.DataFrame({'Col_names': X.columns.tolist(), 'FI': FI})

df
#Bar plot showing the importance of each feature

plt.barh(df.Col_names, df.FI)
from sklearn.svm import SVC



#Fitting Support Vector Machine

svc = SVC().fit(X_smot, y_smot)
#Predicting class labels using the trained SVM model

y_hat3 = svc.predict(X_test)
#Finding the F1-score of the SVM model

f3 = f1_score(y_test, y_hat3)

f3
from sklearn.neighbors import KNeighborsClassifier



#Fitting K-Nearest Neighbor Classifier

knn = KNeighborsClassifier().fit(X_smot, y_smot)
#Predicting class labels using the trained KNN model

y_hat4 = knn.predict(X_test)
#Finding the F1-score of the KNN model

f4 = f1_score(y_test, y_hat4)

f4
#Finding the class labels for the test dataset using Random Forest model

pred = rf.predict(X_scaled_test)

pred