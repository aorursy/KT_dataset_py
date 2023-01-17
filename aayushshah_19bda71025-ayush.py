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
import pandas as pd#for reading csv
import numpy as np#for numerical operations
import matplotlib.pyplot as plt#for plotting
import seaborn as sns#for interactive plotting
%matplotlib inline
#for inline executions
train=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")#reading train data
test=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")#reading test data
train.head()#printing first 5 values of train dataset
test.head()#printing first 5 values of test dataset 
train.shape#printing the number of rows and columns for train
test.shape#printing the number of rows and columns for test
train.info()
train.describe()
train.isnull().sum()

#Checking for class imbalance
print(train['flag'].value_counts())
sns.countplot(x='flag',data=train)
plt.show()
count_anomaly = len(train[train['flag']==0])
count_normal = len(train[train['flag']==1])
pct_anomaly = count_anomaly/(count_anomaly+count_normal)
print("percentage of anomaly:", pct_anomaly*100)
pct_normal = count_normal/(count_anomaly+count_normal)
print("percentage of normal:", pct_normal*100)
print("There is no class imbalance")
sns.pairplot(train,palette='bwr')
import seaborn as sns
sns.boxplot(data=train,orient='v')
train.hist(figsize=(20,20))
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X = train.drop('flag',axis=1)
y = train['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
train.info
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
predictions= dtree.predict(test)
predictions
y_test
print(classification_report(y_train[:2505],predictions))

def correlation(data,columns):
    #get the columns 
    simp = train[columns]
    #do correlation on the columns
    correlation= simp.corr()
    #correlation = pd.DataFrame(data=correlation)
    #Heat map
    sns.heatmap(correlation, annot=True)
 #Different columns names can be given
col = ['timeindex','currentBack']
correlation(train,col)
y_test.shape
