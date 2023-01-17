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
import numpy as np # used for handling numbers

import pandas as pd # used for handling the dataset

from sklearn.model_selection import train_test_split # used for splitting training and testing data

from sklearn.preprocessing import StandardScaler # used for feature scaling

import seaborn as sns # Seaborn visualization library

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv') # to import the train dataset into a variable

test = pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv') #to import the test dataset into a variable

sample = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv') #to import the sample submission into a variable
data.head() # get top rows of the data
data.describe() # description of dataset
data.dtypes #data types of all columns
data.apply(lambda x: len(x.unique())) # get unique values in each column
import pandas_profiling  

pandas_profiling.ProfileReport(data) # to get a profiling report
# to get a correlation plot of the dataset using seaborn library and get the values 

corr = data.corr()

mask = np.zeros_like(corr, dtype=bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,12))

sns.heatmap(

    corr, mask=mask,

    vmin=-1, vmax=1, center=0,

    cmap='coolwarm',

    annot=True,

    square=True

)
X = data.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values #subset the independent features

y = data.iloc[:, 1].values # subset te target variable
## Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train
#scaling the dataset , not used since random forest

#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)

#X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier( n_estimators=20,criterion = 'entropy',max_depth=None, max_features=10,

                                    min_samples_split=8,random_state = 0)

classifier.fit(X_train, y_train)# Predicting the Test set results
# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score

f1_score(y_test, y_pred, labels=None)
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()
test_pred = classifier.predict(test) #predicting for test data
test_pred.shape
sample['flag']=test_pred #replacing flag values
sample.tail()
#convertinf sample back to csv file

sample.to_csv('submit_07.csv',index=False)