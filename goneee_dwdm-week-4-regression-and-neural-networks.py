# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path='/kaggle/input/dwdm-week-6/hr_data.csv'

data = pd.read_csv(data_path)

data.head(7)
data.shape # prints a tuple with the (number of rows, number of columns)
data.columns
from pandasql import sqldf
df_results = sqldf("select count(sales) as no_employees, sales as dept  " +

                   "from data group by sales")

df_results
import matplotlib # import plotting library
matplotlib.style.available #print 
matplotlib.style.use('fivethirtyeight')

# https://bit.ly/dwdm-resources <-- link to course resoures
df_results.plot(kind='barh',x='dept',

                title="Number of employees in each department")

# seaborn , plotly
import seaborn as sns
#help(sns.barplot)

# import matplotlib.pyplot as plt

# import seaborn as sns

sns.barplot(x='dept',y='no_employees',data=df_results)

plt.title("Number of employees in each dept")

plt.ylabel("Employee Count")

def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
label_encoders = create_label_encoder_dict(data)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
# Apply each encoder to the data set to obtain transformed values

data2 = data.copy() # create copy of initial data set

for column in data2.columns:

    if column in label_encoders:

        data2[column] = label_encoders[column].transform(data2[column])



print("Transformed data set")

print("="*32)

data2
print("Columns before ",data2.columns)

data2_columns = data.columns.tolist()

data2_columns[8]='dept' #change "sales" entry in list to "dept"

#data2_columns = [ 'dept' if col =='sales' else col  for col in data2.columns.tolist() ]

print("Columns after",data2_columns)

data2.columns = data2_columns #update new column names

data2.head(3)


