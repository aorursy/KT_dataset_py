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
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
#Importing the Dataset
train_data = pd.read_csv('../input/bda-2019-ml-test/Train_Mask.csv')
test_data = pd.read_csv('../input/bda-2019-ml-test/Test_Mask_Dataset.csv')
#Viewing the first few columns of the train dataset to see what the dataset looks like.
train_data.head()
#Viewing the number of rows and columns of the train_data and test_data
print ('The train data has {0} rows and {1} columns'.format(train_data.shape[0],train_data.shape[1]))

print ('The test data has {0} rows and {1} columns'.format(test_data.shape[0],test_data.shape[1])) 

#Viewing the columns of the train_data
print(list(train_data.columns))
#Decriptive statistics of train_data
train_data.describe()
#information of training dataset
train_data.info()
#checking the distribution of the Target variable 
sns.distplot(train_data['flag'])
#plotting the distribution of the variables
train_data.hist(bins=50, figsize=(15,15))
plt.show()
#checking the missing value in different columns
train_data.isnull().sum()

#checking the types of data
numeric_features = train_data.select_dtypes(include=[np.number])
numeric_features.dtypes
#corelation between the columns with flag
corr = numeric_features.corr()
print (corr['flag'].sort_values(ascending=False)[:5], '\n')
print (corr['flag'].sort_values(ascending=False)[-5:])
# Create a dataset
df = pd.DataFrame(np.random.random((5,16)), columns=[['timeindex', 'flag', 'currentBack', 'motorTempBack', 'positionBack', 'refPositionBack', 'refVelocityBack', 'trackingDeviationBack', 'velocityBack', 'currentFront', 'motorTempFront', 'positionFront', 'refPositionFront', 'refVelocityFront', 'trackingDeviationFront', 'velocityFront']
])
 
# Default heatmap: just a visualization of this square matrix
p1 = sns.heatmap(df)

#counting number of times flag is 1(chain is tensed) and flag is 0(chain is loose)
train_data['flag'].value_counts()
#visualizing the number of times the flag is 0 and 1
sns.countplot(x='flag', data=train_data, palette='hls')
plt.show()
#dividing data into attributes and labels
X = train_data.drop('flag', axis=1)
y = train_data['flag']
#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
#Predictions and Evaluation of Decision Tree
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
##random forest
#Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
#Predictions and Evaluation
predictions = rfc.predict(X_test)
#classification report from the results.
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
#Confusion Matrix for the predictions.
print(confusion_matrix(y_test,predictions))
#Viewing the first few columns of the test dataset to see what the dataset looks like.
test_data.head()
#using the model we have built to make predictions on the test data set
#crate variable predictions1
#clf.predict() method will return a list of predictions given a set of predictors
prediction1 = rfc.predict(test_data)
# submission of the result
Sample_Submission = pd.read_csv('../input/bda-2019-ml-test/Sample Submission.csv')
Sample_Submission['flag'] = prediction1
Sample_Submission.to_csv('../input/bda-2019-ml-test/Sample Submission.csv',index=False)
Sample_Submission['flag'].value_counts()
