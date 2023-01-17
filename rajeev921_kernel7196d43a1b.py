# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier # modelling

from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.model_selection import train_test_split



# Load the data

DATA_FOLDER = '/kaggle/input/learn-together'

df_train = pd.read_csv(f'{DATA_FOLDER}/train.csv', index_col='Id')

df_test = pd.read_csv(f'{DATA_FOLDER}/test.csv', index_col='Id')



#print(df_train)



X = df_train.iloc[:, :-1]  # All the dataframe 

y = df_train.iloc[:, -1]   # Last col of data frame

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_test = df_test



#df_train.head().T



#df_train.describe().T
from sklearn.ensemble import RandomForestClassifier



# Train the model on the test dataset

rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(X_train, y_train)



#Evaluate the model with the validation dataset

y_pred = rf.predict(X_val)

mae = mean_absolute_error(y_pred, y_val)

acc = accuracy_score(y_pred, y_val)

print(f'Mean Absolute Error = {mae}')

print(f'Accuracy Score = {acc}')