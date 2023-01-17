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
data = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

data.head()

# Here is our data set
data.isnull().sum()

# This is beautiful, there is no missing value
import seaborn as sns

import matplotlib.pyplot as plt

plt.subplots(figsize=(15,10))

sns.heatmap(data.corr(),annot=False,linewidths=1);
sns.boxplot(data['thalach'],palette='plasma_r');

# I want to believe that this outlier will not affect us much.
sns.boxplot(data['slope'],palette='Set3');

# Just fine
sns.boxplot(data['cp'],palette='terrain');

# Done
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,GridSearchCV

from sklearn.metrics import accuracy_score



# Here are the libraries we need
y = data['target']

x = data.drop(['target'],axis=1)

# Y is our target variable

# X contains arguments
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

# Train and test separation.
from sklearn.preprocessing import StandardScaler

# Here is the necessary library.
Scaler = StandardScaler() # We create a scaling object.

Scaler.fit(x_train) # We fit this to x_train.

x_train_scaled = Scaler.transform(x_train)

x_test_scaled = Scaler.transform(x_test)

# We have now standardized the sets to be used.
from sklearn.neural_network import MLPClassifier # Our artificial neural network library.

MLPC = MLPClassifier().fit(x_train_scaled,y_train) # We use x_train_scaled.

y_prediction = MLPC.predict(x_test_scaled) # We test with standardized x test.

print(y_prediction[:10]) # Estimated y values
NewData = pd.DataFrame({'Real_Y_Values':y_test,

                       'Prediction_Y_Values':y_prediction})

NewData.head()

# Actual and predicted y values.
accuracy_score(y_test,y_prediction)

# Pretty good test success but not over.
MLPC_params = {

    'alpha':[0.1,0.01,0.02,0.005,0.0001,0.00001],

    'hidden_layer_sizes':[(10,10,10),

                         (100,100,100),

                         (100,100),

                         (3,5),

                         (5,3)],

    'solver':['lbfgs','adam','sgd'],

    'activation':['relu','logistic']

}

MLPC = MLPClassifier() # Clean models

MLPC_cv_model = GridSearchCV(MLPC,MLPC_params,cv=10,n_jobs=-1,verbose=2)

MLPC_cv_model.fit(x_train_scaled,y_train)

# If you don't know this method, gridsearch cv python --> investigate
print('Best parameter for you <3:' + str(MLPC_cv_model.best_params_))
MLPC_tuned = MLPClassifier(activation=MLPC_cv_model.best_params_['activation'],

                            alpha=MLPC_cv_model.best_params_['alpha'],

                            hidden_layer_sizes=MLPC_cv_model.best_params_['hidden_layer_sizes'],

                            solver=MLPC_cv_model.best_params_['solver'])

MLPC_tuned.fit(x_train_scaled,y_train)

y_pred_tuned = MLPC_tuned.predict(x_test_scaled)

accuracy_score(y_test,y_pred_tuned)