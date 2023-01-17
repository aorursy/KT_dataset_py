# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df.head()
df.head()
drop_col = ['area_type','availability','society','balcony']
df = df.drop(drop_col,axis=1)
df.head()
df.isnull().sum()
df=df.dropna()
df.isnull().sum()
df.shape
df['bhk'] = df['size'].apply(lambda x : int(x.split()[0]))
df = df.drop(['size'],axis=1)
df
loc_count=df['location'].value_counts()
low_loc = loc_count[loc_count <= 10]
low_loc
df['location'] = df['location'].apply(lambda y : 'Others' if y in low_loc else y)
df['location'].value_counts()
df.head()
def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True
df[~( df['total_sqft'].apply(isFloat) )].head(10)
def convertSqftToNum(x):
    values = x.split('-')
    if len(values)==2:
        return (float(values[0])+float(values[1]))/2
    try:
        return float(x)
    except:
        return None
df['total_sqft'] = df['total_sqft'].apply(convertSqftToNum)
df.head(10)
df = df.copy()
df['price_per_sqft'] = (df['price']*100000)/df['total_sqft']
df.head()
df = df[~(df['total_sqft']/df['bhk']<300)]
df.shape
drop_row = df.loc[df['bhk']>20]
len(drop_row)
df = df[(df['bath'] <= df['bhk'])]
df
df.describe()
mean_pps = df['price_per_sqft'].mean()
std_pps = df['price_per_sqft'].std()
print(mean_pps,std_pps)
df = df[(df['price_per_sqft']<=mean_pps+std_pps-1000)]
df
df = df[(df['price_per_sqft'] >= mean_pps-std_pps+2500)]
df
x = df[['location','total_sqft','bhk','bath']].values
y = df.iloc[:,-3].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = ct.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)
plt.scatter(df['total_sqft'],df['price'],c='r')
plt.xlabel('Total Sqft')
plt.ylabel('Price')
plt.title('Total Sqft VS Price')
plt.show()
import pickle
with open('HousePrice Prediction Model.pkl','wb') as file:
  pickle.dump(reg,file)
location_list = df['location'].values
location_list = list(set(location_list))
location_list.sort()
location_list
x = pd.DataFrame(x.toarray())
def loc_search(p):
  for i in range(len(location_list)):
    if location_list[i]==p:
      re=i
      return re
  return re
def predict_house_price(loc_name,total_sqft,bhk,bath):
  p_list=np.zeros(len(x.columns))
  p_list[-1]=bath
  p_list[-2]=bhk
  p_list[-3]=total_sqft
  p_list[loc_search(loc_name)]=1.0
  p_list = p_list.reshape(1,len(p_list))
  return(p_list)
predict_house_price('TC Palaya',1440,5,2)
type(x)
loc_search(' Devarachikkanahalli')
reg.predict(predict_house_price(' Devarachikkanahalli',1400,5,2))
reg.predict(x_test)
print(y_test)
with open('HousePrice Prediction Model.pkl','rb') as file:
  model = pickle.load(file)
model.predict(predict_house_price('Yeshwanthpur',1400,5,2))
y_pred = model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),(y_test.reshape(len(y_test),1))),1))
