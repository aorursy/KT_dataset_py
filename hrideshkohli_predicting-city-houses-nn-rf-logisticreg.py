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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent.csv')

df.head()
#checking the shape

df.shape
#checking null values

df.isnull().sum()
# as we can see 1st column is vague and needs to be dropped

df.drop(df.columns[0], axis=1, inplace=True)
df.head()
# floor contains "-" which needs to be removed

df['floor'].replace("-", 0, inplace=True)
# R$ to be removed from columns hoa, rent amount, property tax, fire insurance and total

for col in ['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']:

    df[col]= df[col].replace('R\$', "", regex=True).replace(",", "", regex=True)
df.head()
# Now let's check the unique values in each of the columns

df.nunique()
# As it can be seen above, we removed R$ but need to change the data type to INT
df[['floor', 'hoa','rent amount', 'property tax', 'fire insurance', 'total']]=df[['floor','hoa','rent amount', 'property tax', 'fire insurance', 'total']].astype(int)
# Let's remove "Sem info"

df.replace("Sem info", 0, inplace=True)

df.replace("Incluso", 0, inplace=True)
df.info()
# Now Let's change the column values to dummy values

df=pd.get_dummies(df, drop_first=True)
df.head()
# Let's split the data into X and y now



X=df.drop('city', axis=1)

y=df['city']
# Importing important libraries

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, f1_score, accuracy_score

from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1)
# Now let's apply transformations

sc=MinMaxScaler()

X_train_tx=sc.fit_transform(X_train)

X_test_tx=sc.transform(X_test)
X_train_tx
log_clf=LogisticRegression()

rf_clf=RandomForestClassifier(n_estimators=100, random_state=1)

svm_clf=SVC()

mlp_clf=MLPClassifier(max_iter=400, hidden_layer_sizes=(12, 12), activation='relu', solver='adam')
log_clf.fit(X_train_tx, y_train)

rf_clf.fit(X_train_tx, y_train)

svm_clf.fit(X_train_tx, y_train)

mlp_clf.fit(X_train_tx, y_train)
# Predictions

y_pred_log=log_clf.predict(X_test_tx)

y_pred_rf=rf_clf.predict(X_test_tx)

y_pred_svm=svm_clf.predict(X_test_tx)

y_pred_nn_mlp=mlp_clf.predict(X_test_tx)
# Accuracy of all the models

print("The accuracy of Logistic Regression is", log_clf.score(X_test_tx, y_test))

print("The accuracy of Random Forest is", rf_clf.score(X_test_tx, y_test))

print("The accuracy of SVM is", svm_clf.score(X_test_tx, y_test))

print("The accuracy of Neural Network MLP Classifier is", mlp_clf.score(X_test_tx, y_test))
# F1_score of all the models

print("The F1 Score of Logistic Regression is", f1_score(y_pred_log, y_test))

print("The F1 Score of Random Forest is", f1_score(y_pred_rf, y_test))

print("The F1 Score of SVM is", f1_score(y_pred_svm, y_test))

print("The F1 Score of Neural Network MLP Classifier is", f1_score(y_pred_nn_mlp, y_test))