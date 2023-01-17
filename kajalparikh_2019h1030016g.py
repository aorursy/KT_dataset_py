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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import resample



%matplotlib inline
df = pd.read_csv("../input/minor-project-2020/train.csv")
#Displaying five rows of the original train dataset. 

df.head()
#Displaying the count of target value '0' and '1' 

df.target.value_counts()
#Separating target value and other features into Y and X

Y = df[['target']]

df.drop(['target'], axis=1, inplace=True)

X = df



from sklearn.model_selection import train_test_split



#Splitting X and Y into train and test dataset with seed value(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



#Concatenating X_train and y_train for further downsampling

train_data = pd.concat([X_train, y_train], axis=1)



negative = train_data[train_data.target==0]

positive = train_data[train_data.target==1]



#Displaying number of records in majority and minority classes

print('Number of records in majority class:',len(negative))

print('Number of records in minority class:',len(positive))
#Performing undersampling to balance the train dataset

negative_downsampled = resample(negative,

 replace=True, #sample with replacement

 n_samples=len(positive), #number of records in minority class

 random_state=42) # reproducible results



#Combining minority and downsampled majority

downsampled = pd.concat([positive, negative_downsampled])



#Checking new counts of target value 

downsampled.target.value_counts()
#Calculating some statistical values for each columns

downsampled.describe()
#Checking for null or 0 values for each column

downsampled.isna().sum()
#Again separating y_train and X_train from the downsampled training dataset.

y_train = downsampled[['target']]

downsampled.drop(['target'], axis=1, inplace=True)

X_train = downsampled
from sklearn.preprocessing import StandardScaler



X_train.info()



#Using scalar for normalising data 

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
#Displaying length of the downsampled train dataset and splitted test dataset

print(len(X_train),len(X_test))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



#Taking parameters to be passed to GridSearchCV

parameters = { "C": [0.1,1,100], "penalty": ["l1", "l2"], "solver": ["liblinear"]}



lr_cv = LogisticRegression(random_state=42)



#Performing Grid Search

clf = GridSearchCV(lr_cv, parameters, verbose=1)



#Fitting model to determine best parameters

best_model = clf.fit(scaled_X_train, y_train.values.reshape(-1,))



#Displaying best parameters calculated

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])
#Linear Regression with best parameters calculated earlier

lr = LogisticRegression(C=100,penalty="l1",solver="liblinear", random_state=42)

lr.fit(scaled_X_train, y_train)



#Printing score

print("score",lr.score(X_test,y_test))
#Predicting probabilities using this model

y_pred = lr.predict_proba(scaled_X_test)[:,1]
#Printing accuracy obtained for test dataset

print("Accuracy is : {}".format(lr.score(scaled_X_test, y_test)))
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



#Calculating ROC 

FPR, TPR, _ = roc_curve(y_test, y_pred)



#Calculating AUC score

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



#Plotting ROC Curve (FPR vs TPR)

plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC', fontsize= 18)

plt.show()
#Reading original test dataset

df_test = pd.read_csv("../input/minor-project-2020/test.csv")
X_test = df_test

scaled_X_test = scalar.transform(X_test)
#Predicting probabilities of test dataset using model developed 

y_pred = lr.predict_proba(scaled_X_test)[:,1]

y_pred = pd.Series(y_pred)
#Creating dataframe with id and target as features

data = pd.DataFrame({

                  "id": df_test['id'], 

                  "target": y_pred})
import csv  

    

#Converting dataframe to .csv file which consist of final test predictions

data.to_csv('test_predictions_gridsearch_prob.csv', index = False)
#Reading final test predictions

df_result = pd.read_csv("test_predictions_gridsearch_prob.csv")
#Displaying first 20 predictions

df_result.head(20)