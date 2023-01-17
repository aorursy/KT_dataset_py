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
import seaborn as sns
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe().T
df2 = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NAN)
df2
df2.isnull().sum()
sns.pairplot(df2)
df2['Glucose'].fillna(df2['Glucose'].mean(),inplace=True)
df2['BloodPressure'].fillna(df2['BloodPressure'].mean(),inplace=True)
df2['SkinThickness'].fillna(df2['SkinThickness'].median(),inplace=True)
df2['Insulin'].fillna(df2['Insulin'].median(),inplace=True)
df2['BMI'].fillna(df2['BMI'].median(),inplace=True)
df2.isnull().sum()
df3=pd.concat([df2,df[['Pregnancies','DiabetesPedigreeFunction','Age','Outcome']]],axis=1)
df3
sns.pairplot(df3,hue='Outcome')
!pip install pycaret
from pycaret.classification import *
clf1 = setup(data = df3, target = 'Outcome')
compare_models()
xgboost = create_model('xgboost')
interpret_model(xgboost)
interpret_model(xgboost, plot = 'correlation')
plot_model(xgboost)
plot_model(xgboost,'confusion_matrix')
final = finalize_model(xgboost)
