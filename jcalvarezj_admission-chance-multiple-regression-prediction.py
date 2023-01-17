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
admission_df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

admission_df.head()
admission_df.columns
admission_df = admission_df.rename(columns = {'LOR ': 'LOR', 'Chance of Admit ': 'Chance of Admit'}).drop(['Serial No.'], axis = 1)

admission_df.columns
admission_df.isna().any()
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



X = admission_df.drop(['Chance of Admit'], axis = 1)

y = admission_df[['Chance of Admit']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



linear_regression = sm.OLS(y_train, sm.add_constant(X_train)).fit()



y_predict = linear_regression.predict(sm.add_constant(X_test))



print('R2: ', r2_score(y_test, y_predict))
linear_regression.params
def calculate_prediction(gre, toefl, uni_rating, sop, lor, cgpa, research):

    X_test = [gre, toefl, uni_rating, sop, lor, cgpa, research]

    

    result = linear_regression.params[0]

    

    for i, x in enumerate(X_test):

        result += linear_regression.params[i+1] * x

    

    return result





prediction = calculate_prediction(280, 100, 3, 3, 5, 9, 0)



print(f'This candidate\'s chance of being admitted is of {prediction*100:.2f} %')