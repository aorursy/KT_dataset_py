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
import seaborn as sns
from pandas import DataFrame
!pip install sklearn
from sklearn import svm

from sklearn.model_selection import train_test_split
file='car5.csv'
data=pd.read_csv(file)
data.dtypes
def str2int(x):
  return int(x.replace(',',''))
data['displacement']=data['displacement'].apply(str2int)
data['empty_vehicle_weight']=data['empty_vehicle_weight'].apply(str2int)
data[['displacement','empty_vehicle_weight']].dtypes
#int형태로 변환된 것을 확인할 수 있습니다.
pd.unique(data.typeofoil)
pd.unique(data.typeofcar)
#array(['휘발유', '경유', 'LPG'], dtype=object)
#array(['하이브리드', '내연기관'], dtype=object)
data['typeofoil']=data['typeofoil'].map({'휘발유':0,'경유':1,'LPG':'2'})
data['typeofcar']=data['typeofcar'].map({'하이브리드':0,'내연기관':1})
data.drop(['model'],axis=1,inplace=True)
sns.countplot(data['grade'])
#데이터를 가공한 최종 Dataset
data.to_csv('car5_data.csv')
train, test=train_test_split(data,test_size=0.2,random_state=2019)

x_train=train.drop(['grade'],axis=1)
y_train=train.grade

x_test=test.drop(['grade'],axis=1)
y_test=test.grade

print(len(train),len(test))
train=DataFrame(train)
test=DataFrame(test)
train.to_csv('car5_train.csv')
test.to_csv('car5_test.csv')