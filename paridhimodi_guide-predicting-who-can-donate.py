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



df = pd.read_csv('../input/donorsprediction/Raw_Data_for_train_test.csv')



df.head()
df.columns[df.isnull().any()]
# Fill numeric rows with the median

for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            # Fill missing numeric values with median since it's more robust than the mean

            df[label] = content.fillna(content.median())

            

df.columns[df.isnull().any()]
df.info()
# Turn categorical variables into numbers

for label, content in df.items():

    # Check columns which aren't numeric

    if not pd.api.types.is_numeric_dtype(content):

        # print the columns that are objectt type 

        print(label)

        df[label] = pd.Categorical(content).codes+1
df.head()
df = df.drop('TARGET_D', axis=1)

df.head()
# input features

x = df.drop('TARGET_B', axis=1)



# Target variable

y = df['TARGET_B']



x.head()
y.head()
# Import standard scaler

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



# apply scaler

x = ss.fit_transform(x)



x
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)





from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# define and configure the model

model = KNeighborsClassifier()



# fit the model

model.fit(xtrain, ytrain)



# evaluate the model

preds = model.predict(xtest)

accuracy_score(ytest, preds)
from sklearn.ensemble import RandomForestClassifier



# define and configure the model

model = RandomForestClassifier()



# fit the model

model.fit(xtrain, ytrain)



# evaluate the model

preds = model.predict(xtest)

accuracy_score(ytest, preds)
from xgboost import XGBClassifier



# define and configure the model

model = XGBClassifier()



# fit the model

model.fit(xtrain, ytrain)



# evaluate the model

preds = model.predict(xtest)

accuracy_score(ytest, preds)
import numpy as np

from sklearn.model_selection import RandomizedSearchCV



# different randomforestregressor hyperperameters

rf_grid = {'n_estimators' : np.arange(10, 100, 10),

           'max_depth': [None, 3, 5, 10],

           'min_samples_split' : np.arange(2, 20, 2),

           'min_samples_leaf': np.arange(1, 20, 2),

            'max_features' : [0.5, 1, 'sqrt', 'auto']}



# instentiate randomizedsearchcv model

rs_model= RandomizedSearchCV(RandomForestClassifier(n_jobs = -1, 

                                                  random_state=42),

                                                  param_distributions = rf_grid,

                                                  n_iter = 90,

                                                  cv=5,

                                                  verbose=True)



rs_model.fit(xtrain, ytrain)
rs_model.best_params_
ideal_model = RandomForestClassifier(n_estimators= 70,

                                     min_samples_split = 8,

                                     min_samples_leaf = 1,

                                     max_features = 'auto',

                                     max_depth = 10)



# fit the model

ideal_model.fit(xtrain, ytrain)



# evaluate the model

preds = ideal_model.predict(xtest)

accuracy_score(ytest, preds)
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

probs = ideal_model.predict_proba(xtest)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(ytest, preds)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
test_df = pd.read_csv('../input/donorsprediction/Predict_donor.csv')

test_df.head()
# Fill numeric rows with the median

for label, content in test_df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            # Fill missing numeric values with median since it's more robust than the mean

            test_df[label] = content.fillna(content.median())

# Turn categorical variables into numbers

for label, content in test_df.items():

    # Check columns which aren't numeric

    if not pd.api.types.is_numeric_dtype(content):

        # print the columns that are object type 

        print(label)

        test_df[label] = pd.Categorical(content).codes+1
Target = ideal_model.predict(test_df)

Target
PREDICTED_df = pd.DataFrame()

PREDICTED_df['TARGET_B'] = Target

PREDICTED_df['CONTROL_NUMBER'] = test_df['CONTROL_NUMBER']

PREDICTED_df.head()