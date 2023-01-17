# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



# !pip install ipaddress

import ipaddress



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
import missingno as msno
order = pd.read_csv("../input/train_order_data.csv") #order information

merch = pd.read_csv("../input/train_merchant_data.csv") #Merchant information

target = pd.read_csv("../input/train.csv") #Target is not available as it is to be predicted

ip_country = pd.read_csv("../input/ip_boundaries_countries.csv") #Ip address boundries of each country
# Sanity check for Train Dataset(Orders)

order.head()

# order.count() #54213

order.count() #54213

order.tail()
import matplotlib.pyplot as plt

msno.bar(order)

plt.show()
order['Order_ID'].nunique() #54213-- No duplicates found in order_Id
#sanity check for Train Dataset(merchant)

merch.head()
merch.tail()


msno.bar(merch)

plt.show()
merch.count() #54213
merch["Merchant_ID"].nunique() #54213---No duplicates found in Merchant_ID
#Customer_id and Order ID are almost varying by record. Can be dropped



order["Customer_ID"].nunique() #34881
order['Order_ID'].nunique() #54213- No predictive power
order.head()
merged_data = pd.merge(order, merch, how='inner', on = "Merchant_ID")

merged_data.head()
target.head()

target['Merchant_ID'].nunique() #54231 -- No duplicates on Merchant ID. Ok to join
# target['Merchant_ID'].nunique() #54231 -- No duplicates on Merchant ID. Ok to join
merged_data = pd.merge(merged_data, target, how = 'inner', on = 'Merchant_ID')
merged_data.head()
merged_data.tail()
print(merged_data.count()) #54213 --Row count not affected, joins worked fine
X = merged_data.copy().drop("Fraudster",axis=1)

y = merged_data["Fraudster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y) 
print("X_train",X_train.shape)

print("X_test",X_test.shape)

print("y_train",y_train.shape)

print("y_test",y_test.shape)
y_train.head()
X_train.head()
device_id = X_train.groupby(["Registered_Device_ID"]).agg({"Merchant_ID":"nunique"}).reset_index()

device_id = device_id[(device_id["Merchant_ID"] > 1)].reset_index(drop=True)

device_id.head()
device_id.count()
device_id.count().nunique()
device_id["Multiple_Merchants"] = np.nan



for i in range(0, device_id.shape[0]):

    if device_id.loc[i, 'Merchant_ID'] > 1:

        device_id.loc[i,'Multiple_Merchants'] =1

    else:

        device_id.loc[i,'Multiple_Merchants'] = 0
device_id["Muliple_Merchants"] = device_id["Multiple_Merchants"].astype('category')

device_id.head()
device_id.drop(['Merchant_ID'],axis=1,inplace=True)
print(device_id.dtypes)

device_id.head()
device_id.count()
X_train = pd.merge(X_train, device_id, how='left', on = 'Registered_Device_ID') #get the device-id level flags on train data
X_train.count()
print(X_train.dtypes)

X_train.head()
X_train["target"] = y_train
ip_address = X_train.groupby(['IP_Address']).agg({'target':"sum",'Merchant_ID':'nunique'}).reset_index()

ip_address[ip_address['target']>1].head()