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
plt.style.use('seaborn')
# loading data
data = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
data.head()
data.info()
data.isna().sum()
# data distribution
data.select_dtypes(include= [np.int64, np.float64]).hist(figsize = (15,10))
plt.tight_layout()
plt.show()
# pairplot
sns.pairplot(data, hue = 'Gender')
plt.show()
# crate a new column whether they have disease or not
data['diagnosed'] = data['Dataset'].replace([1,2], ['yes', 'no'])
data.head()
# drop the column Dataset from data
data.drop(columns= ['Dataset'], axis = 1, inplace= True)
data.head()
# ploting diagnosed 
sns.countplot(x = 'diagnosed', data = data)
plt.show()
# ploting gender
sns.countplot(x = 'Gender', data = data)
plt.show()
pd.crosstab(data['Gender'], data['diagnosed']).plot(kind = 'bar', figsize = (10,5))
plt.show()
# average age of male and female diagnosed with it.
data.groupby(['Gender', 'diagnosed'])['Age'].mean()
# average age of male and female diagnosed with it.
data.groupby(['Gender', 'diagnosed'])['Age'].mean().plot(kind = 'bar')
plt.show()
# gender & diagnosed wise min values
data.groupby(['Gender', 'diagnosed']).agg(
    min_Total_Bilirubin = ('Total_Bilirubin', min),
    min_Direct_Bilirubin = ('Direct_Bilirubin', min),
    min_Alkaline_Phosphotase = ('Alkaline_Phosphotase', min),
    min_Alamine_Aminotransferase = ('Alamine_Aminotransferase', min),
    min_Total_Protiens = ('Total_Protiens', min),
    min_Albumin = ('Albumin', min),
    min_Albumin_and_Globulin_Ratio = ('Albumin_and_Globulin_Ratio', min)
    )

data.groupby(['Gender', 'diagnosed']).agg(
    min_Total_Bilirubin = ('Total_Bilirubin', min),
    min_Direct_Bilirubin = ('Direct_Bilirubin', min),
    min_Alkaline_Phosphotase = ('Alkaline_Phosphotase', min),
    min_Alamine_Aminotransferase = ('Alamine_Aminotransferase', min),
    min_Total_Protiens = ('Total_Protiens', min),
    min_Albumin = ('Albumin', min),
    min_Albumin_and_Globulin_Ratio = ('Albumin_and_Globulin_Ratio', min)
    ).plot(kind = 'bar', figsize = (15,10))
plt.show()
# gender & diagnosed wise min values
data.groupby(['Gender', 'diagnosed']).agg(
    max_Total_Bilirubin = ('Total_Bilirubin', max),
    max_Direct_Bilirubin = ('Direct_Bilirubin', max),
    max_Alkaline_Phosphotase = ('Alkaline_Phosphotase', max),
    max_Alamine_Aminotransferase = ('Alamine_Aminotransferase', max),
    max_Total_Protiens = ('Total_Protiens', max),
    max_Albumin = ('Albumin', max),
    max_Albumin_and_Globulin_Ratio = ('Albumin_and_Globulin_Ratio', max)
    )
data.groupby(['Gender', 'diagnosed']).agg(
    max_Total_Bilirubin = ('Total_Bilirubin', max),
    max_Direct_Bilirubin = ('Direct_Bilirubin', max),
    max_Alkaline_Phosphotase = ('Alkaline_Phosphotase', max),
    max_Alamine_Aminotransferase = ('Alamine_Aminotransferase', max),
    max_Total_Protiens = ('Total_Protiens', max),
    max_Albumin = ('Albumin', max),
    max_Albumin_and_Globulin_Ratio = ('Albumin_and_Globulin_Ratio', max)
    ).plot(kind = 'bar', figsize =(15,10))
plt.show()
data.columns
sns.pairplot(data.drop('Gender', axis = 1), hue = 'diagnosed')
plt.show()
