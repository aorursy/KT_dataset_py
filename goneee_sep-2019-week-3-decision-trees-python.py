# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)]

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_path='/kaggle/input/dwdm-week-3/Creditcardprom.csv'

data = pd.read_csv(data_path)

data
data.drop([1,3],axis=0) # casewise deletion

data.drop(['Magazine Promo'],axis=1) # listwise deletion
data.columns # viewing all columns
# extracting only sex, age, income range, watch promo, life insurance

data2 = data[['Income Range','Sex','Age','Life Ins Promo','Watch Promo']]

data2
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
data3 = data.copy()
from sklearn.preprocessing import LabelEncoder

income_range_encoder = LabelEncoder()

income_range_encoder.fit(data3['Income Range'])

income_range_encoder
income_range_encoder.transform(['40-50,000'])
data3['Income Range'][0]
for column in data3.columns:

    print("*"*32)

    print(data3[column].describe())
data3['Has_Life_Insurance'] = data3['Life Ins Promo'].apply(lambda val : 1 if val == 'Yes' else 0)

data3['No_Life_Insurance'] = data3['Life Ins Promo'].apply(lambda val : 0 if val == 'Yes' else 1)

data3['Life_Insurance_Encoded'] = data3['Life Ins Promo'].apply(lambda val : 1 if val == 'Yes' else 0)

data3
data3['Income Range'].nunique()
label_encoders = create_label_encoder_dict(data2)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
label_encoders['Income Range'].classes_
column='Income Range'

pd.DataFrame(  )
[*range(0,len(label_encoders[column].classes_))]
x = [*range(1,6)]

y = [ num**2 for num in x]

print(x,y)
temp = pd.DataFrame({

    'x':x,

    'y':y

})

temp
print("Columns before",temp.columns)

temp.columns = ['x_col','y_col']

print("Columns after",temp.columns)

print(temp)
temp.index = ["row %d" % num for num in x]

print(temp)
# Apply each encoder to the data set to obtain transformed values

data3 = data2.copy() # create copy of initial data set

for column in data3.columns:

    if column in label_encoders:

        data3[column] = label_encoders[column].transform(data3[column])



print("Transformed data set")

print("="*32)

data3
# Let us separate our dependent/output(Y) and independent/input

# variables (X)

X_data = data3[['Income Range','Sex','Age','Watch Promo']]

Y_data = data3['Life Ins Promo']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data,test_size=0.3)
15*0.7
from pandasql import sqldf

#pysqldf = lambda q: sqldf(q, globals())
data3
data3['LifeInsPromo']=data['Life Ins Promo']

data3
sqldf('select Sex, Age from data3 where LifeInsPromo = "Yes"')
data3['LifeInsPromo'].value_counts()