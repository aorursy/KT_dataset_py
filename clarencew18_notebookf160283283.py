# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#loading the data into memory

df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
#inspecting ranges for the numeric variables. This will form the basis for normalizing them later

df.describe()
#inspecting the levels of the categorical variable sales

df['sales'].value_counts()
#inspecting the levels of the categorical variable salary

df['salary'].value_counts()
from sklearn.preprocessing import OneHotEncoder, Normalizer, LabelEncoder

from sklearn.cross_validation import train_test_split
labels_ohe = OneHotEncoder()

sales_enc = LabelEncoder()

salary_enc = LabelEncoder()

sales_ohe = OneHotEncoder()

salary_ohe = OneHotEncoder()
#performing one hot encoding on the labels/target

y = labels_ohe.fit_transform(df['left'].reshape(-1,1)).toarray()



#performing label encoding for the multiple classes for sales and salary

#we then perform one hot encoding on the encoded variables

#reshape(-1,1) means a columnar vector

#the toarray() method returns a matrix

sales_bin = sales_ohe.fit_transform(

                        sales_enc.fit_transform(df['sales']).reshape(-1,1)).toarray()

salary_bin = salary_ohe.fit_transform(

                        salary_enc.fit_transform(df['salary']).reshape(-1,1)).toarray()
#checking if all went according to plan

y[:5]
sales_bin[:5]
def scaler(x):

    norm = Normalizer()

    return norm.fit_transform(x)
norm_num_fields = df[['number_project','average_montly_hours','time_spend_company']].apply(

                                            func=lambda x: scaler(x.reshape(-1,1)),axis=1)