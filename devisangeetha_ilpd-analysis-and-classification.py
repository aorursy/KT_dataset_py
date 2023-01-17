

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
patient_data=pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
patient_data.head()
patient_data.shape
patient_data.info()
patient_data.isnull().sum()
sns.distplot(patient_data['Age'])