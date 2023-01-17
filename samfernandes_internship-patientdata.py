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
patientData = '/kaggle/input/patients-info/patient_data.csv'
AllPatientInfo = pd.read_csv(patientData)
AllPatientInfo.describe()
y = AllPatientInfo.isCovid

patient_features = ['temperature','pO2_saturation', 'leukocyte_count', 'neutrophil_count', 'lymphocyte_count']

X = AllPatientInfo[patient_features]


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))

# Imputation removed column names; put them back
imputed_train_X.columns = train_X.columns
imputed_val_X.columns = val_X.columns

print("MAE from Imputation:")
print(score_dataset(imputed_train_X, imputed_val_X, train_y, val_y))