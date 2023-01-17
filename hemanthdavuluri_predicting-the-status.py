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
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
!pip install missingno
import missingno as msno
msno.matrix(data)
msno.bar(data) 
temp=data[['status','salary']]
temp[temp['status']=="Not Placed"]
data['salary'].fillna(0,inplace=True)
data.isnull().sum()
data.head()
!pip install sweetviz
import sweetviz as sv
#feature_config = sv.FeatureConfig(skip="", force_text=["Age"])
my_report = sv.compare_intra(data, data["gender"] == "M", ["M", "F"])
my_report.show_html()
data.head()
print(data['ssc_b'].unique())
print(data['hsc_b'].unique())
print(data['hsc_s'].unique())
print(data['degree_t'].unique())
print(data['workex'].unique())
print(data['specialisation'].unique())
cat = data.select_dtypes(include=['object']).copy()
cat.head()
print('Gender column')
print(cat['gender'].value_counts())
print('ssc_b column')
print(cat['ssc_b'].value_counts())
print('hsc_b column')
print(cat['hsc_b'].value_counts())
print('hsc_s column')
print(cat['hsc_s'].value_counts())
print('degree_t column')
print(cat['degree_t'].value_counts())
print('workex column')
print(cat['workex'].value_counts())
print('specialisation column')
print(cat['specialisation'].value_counts())
print('status column')
print(cat['status'].value_counts())

data1 = data
#Gender Column
data1['gender']=data1['gender'].replace('M',0)
data1['gender']=data1['gender'].replace('F',1)
#SSC_B column
data1['ssc_b']=data1['ssc_b'].replace('Central',0)
data1['ssc_b']=data1['ssc_b'].replace('Others',1)
#HSC_B column
data1['hsc_b']=data1['hsc_b'].replace('Central',0)
data1['hsc_b']=data1['hsc_b'].replace('Others',1)
#HSC_S column
data1['hsc_s']=data1['hsc_s'].replace('Commerce',0)
data1['hsc_s']=data1['hsc_s'].replace('Science',1)
data1['hsc_s']=data1['hsc_s'].replace('Arts',1)
#degree_t column
data1['degree_t']=data1['degree_t'].replace('Comm&Mgmt',0)
data1['degree_t']=data1['degree_t'].replace('Sci&Tech',1)
data1['degree_t']=data1['degree_t'].replace('Others',1)                                   
#workex column
data1['workex']=data1['workex'].replace('Yes',0)
data1['workex']=data1['workex'].replace('No',1)
#specialisation column
data1['specialisation']=data1['specialisation'].replace('Mkt&Fin',0)
data1['specialisation']=data1['specialisation'].replace('Mkt&HR',1)
#stats column
data1['status']=data1['status'].replace('Placed',0)
data1['status']=data1['status'].replace('Not Placed',1)
data1.head()
data1=data1.drop(columns=['salary','sl_no'])
data1.corr()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = data1.drop(columns=['status'])
y = data1.status

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)