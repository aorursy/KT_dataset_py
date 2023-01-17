## Uploading my kaggle.json (required for accessing Kaggle APIs)
from google.colab import files
files.upload()
## Install Kaggle API
!pip install -q kaggle
## Moving the json to appropriate place
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download mohansacharya/graduate-admissions
!echo "========================================================="
!ls
!unzip graduate-admissions.zip
!echo "========================================================="
!ls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 60)
%matplotlib inline
FILE_NAME = "../input/Admission_Predict_Ver1.1.csv"

raw_data = pd.read_csv(FILE_NAME)
raw_data.head()
## Are any null values persent ?
raw_data.isnull().values.any()
## So no NaNs apparently
## Let's just quickly rename the dataframe columns to make easy references
## Notice the blankspace after the end of 'Chance of Admit' column name
raw_data.rename(columns = {
    'Serial No.' : 'srn',
    'GRE Score'  : 'gre',
    'TOEFL Score': 'toefl',
    'University Rating' : 'unirating',
    'SOP'        : 'sop',
    'LOR '        : 'lor',
    'CGPA'       : 'cgpa',
    'Research'   : 'research',
    'Chance of Admit ': 'chance'
}, inplace=True)
raw_data.describe()
fig, ax = plt.subplots(ncols = 2)
sns.regplot(x='chance', y='cgpa', data=raw_data, ax=ax[0])
sns.regplot(x='chance', y='unirating', data=raw_data, ax=ax[1])
fig, ax = plt.subplots(ncols = 2)
sns.regplot(x='chance', y='gre', data=raw_data, ax=ax[0])
sns.regplot(x='chance', y='toefl', data=raw_data, ax=ax[1])
fig, ax = plt.subplots(ncols = 3)
sns.regplot(x='chance', y='sop', data=raw_data, ax=ax[0])
sns.regplot(x='chance', y='lor', data=raw_data, ax=ax[1])
sns.regplot(x='chance', y='research', data=raw_data, ax=ax[2])
THRESH = 0.6
# I think we can also drop srn as it is not doing absolutely anything
raw_data.drop('srn', axis=1, inplace=True)
raw_data['chance'] = np.where(raw_data['chance'] > THRESH, 1, 0)
raw_data.head()
raw_data.describe()
X = raw_data.drop(columns='chance')
Y = raw_data['chance'].values.reshape(raw_data.shape[0], 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


print("Training set ...")
print("X_train.shape = {}, Y_train.shape =  is {}".format(X_train.shape, 
                                                          Y_train.shape))

print("Test set ...")
print("X_test.shape = {}, Y_test.shape =  is {}".format(X_test.shape, 
                                                          Y_test.shape))
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=3000000, 
                         tol=1e-8)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))
from sklearn.svm import LinearSVC
clf = LinearSVC(verbose=1, max_iter=3000000, tol=1e-8, C=1.25)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Mean Absolute Error = ", mae(Y_test, Y_pred))
