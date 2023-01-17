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

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix





import warnings

warnings.filterwarnings("ignore",)
train = pd.read_csv('/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Train.csv')

test = pd.read_csv('/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Test.csv')

print(train.shape)

train.head()
print(test.shape)

test.head()
sns.countplot(train['labels'])
train.isnull().sum()
train.describe()
train.skew().sort_values()
#  list of discrete variables

discrete_vars = [var for var in train.columns if len(train[var].unique())<20 and var not in ['labels']]



print('Number of discrete variables: ', len(discrete_vars))
# list of continuous variables

cont_vars = [var for var in train.columns if var not in discrete_vars+['labels']]



print('Number of continuous variables: ', len(cont_vars))
# let's visualise the discrete variables

train[discrete_vars].head()
# let's visualise the continuos variables

train[cont_vars].head()
# Let's go ahead and analyse the distributions of these variables

def analyse_continous(df, var):

    df = df.copy()

    df[var].hist(bins=20)

    plt.ylabel('Number of Features')

    plt.xlabel(var)

    plt.title(var)

    plt.show()

    

for var in cont_vars:

    analyse_continous(train, var)
# Let's go ahead and analyse the distributions of these variables

def analyse_transformed_continous(df, var):

    df = df.copy()

    

#     # log does not take negative values, so let's be careful and skip those variables

#     if 0 in train[var].unique():

#         pass

#     else:

        # log transform the variable

    df[var] = np.log(df[var])

    df[var].hist(bins=20)

    plt.ylabel('Number of houses')

    plt.xlabel(var)

    plt.title(var)

    plt.show()

    

for var in cont_vars:

    analyse_transformed_continous(train, var)
from imblearn.over_sampling import SMOTE

# First i will try for Smote if it was not good mean we will apply for the undersampling

over = SMOTE()
X = train.drop('labels', axis=1)  # Keep all features except 'Labels'

y = train['labels']  # Just keep 'Labels'
X, y = over.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
from sklearn.preprocessing import MinMaxScaler



scaler= MinMaxScaler()



scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, random_state=7)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)



print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



# Assess model performance

lr = LogisticRegression()

lr.fit(X_train, y_train)

strat_kfold = StratifiedKFold(10, random_state=7)

score = cross_val_score(lr, X_train, y_train, scoring='accuracy', cv=10)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(score), np.std(score)))

# RUN and Predict the Random Forest Model



submission = clf.predict(test)



submission = pd.DataFrame(submission)



submission.to_csv('submission.csv')
