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
#import libraries
import numpy as np
import sklearn as sk
import pandas as pd
import os
#load data
df = pd.read_csv('/kaggle/input/alzheimers-clinical-data/clinical-data-for Alzheimers.csv')
df.head()

#cleaning data
df['mmse'] = df['mmse'].str.replace(r'?', '72')
df['memory'] = df['memory'].str.replace(r'?','1' )

df = df.dropna()
df.isnull().sum().sum()

#split data to variables and target
X = df
Y = df['dx1']
del df

#Remove unnecessary columns (features), remove first 9 columns and 'Dx codes for submission'
remove_columns = list(X.columns)[0:1]
remove_columns.append('dx1')

print('Removing columns:', remove_columns)

X = X.drop(remove_columns, axis=1)

features = list(X.columns)
X.head(5)


#classify variables into numerical ans catergorical variables
numerical_vars = ['ageAtEntry', 'mmse', 'cdr','memory']
categorical_vars = list(set(features) - set(numerical_vars))

print('Categorical variable distributions:\n')

for var in categorical_vars:
    print('\nDistribution of', var)
    
    print(X[var].value_counts())
#visualize  data
from matplotlib import pyplot as plt
%matplotlib inline

print('Numerical Variable Distributions:\n')

for var in numerical_vars:
    plt.hist(X[var], bins=10)
    plt.title(var + ' Distribution')
    plt.show()
    
    # descriptive stats
    print(X[var].describe())
    print(X[var].value_counts())
plt.bar(Y.value_counts().index, Y.value_counts())
plt.show()
#Pre-processing
#for each categorical var, convert to 1-hot encoding
for var in categorical_vars:
    print('Converting', var, 'to 1-hot encoding')
    
    #get 1-hot and replace original column with the >= 2 categories as columns
    one_hot_df = pd.get_dummies(X[var])
    X = pd.concat([X, one_hot_df], axis=1)
    X = X.drop(var, axis=1)
    
X.head(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print('X_train:', X_train.shape, '\ty_train:', y_train.shape)
print('X_test:', X_test.shape, '\ty_test:', y_test.shape)
num_test = X_test.shape[0]
#buliding model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import FitFailedWarning
import warnings
y_testlog_clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000000, multi_class='multinomial')
print('Validation Accuracy = ', format(cross_val_score(log_clf, X_train, y_train, cv=5).mean(), '.2%'))
rf_clf = RandomForestClassifier(n_estimators=200)
print('Validation Accuracy = ', format(cross_val_score(rf_clf, X_train, y_train, cv=5).mean(), '.2%'))
knn_clf = KNeighborsClassifier(n_neighbors=10)
print('Validation Accuracy = ', format(cross_val_score(knn_clf, X_train, y_train, cv=5).mean(), '.2%'))
mlp_clf = MLPClassifier(hidden_layer_sizes=(15, 10), alpha=3, learning_rate='adaptive', max_iter=100000)
print('Validation Accuracy = ', format(cross_val_score(mlp_clf, X_train, y_train, cv=5).mean(), '.2%'))
SVC_clf=SVC()
print('Validation Accuracy = ', format(cross_val_score(SVC_clf, X_train, y_train, cv=5).mean(), '.2%'))

NB_clf=GaussianNB()
print('Validation Accuracy = ', format(cross_val_score(NB_clf, X_train, y_train, cv=5).mean(), '.2%'))

from sklearn.tree import DecisionTreeClassifier
DT_clf= DecisionTreeClassifier()
print('Validation Accuracy = ', format(cross_val_score(DT_clf, X_train, y_train,cv=5).mean(), '.2%'))

# Evaluate feature importances given by Random Forest

rf_clf.fit(X_train, y_train)

feature_importances = pd.DataFrame(rf_clf.feature_importances_, index=X_train.columns, 
                                   columns=['Importance']).sort_values('Importance', ascending=False)
print(feature_importances[:10])