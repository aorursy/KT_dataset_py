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
from sklearn.preprocessing import LabelEncoder

dftest=pd.read_csv('../input/Test.csv')

dftrain=pd.read_csv('../input/Train.csv')

dftrain['source']='TRAIN'

dftest['source']='TEST'
data=pd.concat([dftrain,dftest],ignore_index=True)

print(data.apply(lambda x:sum(x.isnull())))

print(data.apply(lambda x:len(x.unique())))

categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

print(len(categorical_columns))

categorical_columns=[x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
for x in categorical_columns:

    print("\n frequency of %s"%x)

    print(data[x].value_counts())

data['Item_Weight'].interpolate(inplace=True)

data['Outlet_Size'].fillna("Medium",inplace=True)  # first find mode of Outlet_size using data['Outlet_Size'].mode() and then fillna

#grouped=data.groupby(["Outlet_Type","Outlet_Size"],as_index=False)

data['Item_Visibility'].interpolate(inplace=True)

meanofitemvis=data['Item_Visibility'].mean()

data=data.replace({'Item_Visibility': {0: meanofitemvis}})
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

data['Item_Type_Combined']=data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
pivot=pd.pivot_table(data,index=['Outlet_Establishment_Year'])

pivot2=pd.pivot_table(data,index=['Item_Identifier'],values=['Item_Weight','Item_Visibility'],aggfunc=np.sum)

grouped=data.groupby("Item_Visibility")

print(grouped)       

s=data['Item_Identifier'][data.Item_Fat_Content=='Low Fat'].value_counts()

for x in range(0,len(s)):

    if s[x]==s.max():

        print(x,s[x])

s.reset_index()
df=data['Item_Identifier'][data.Item_Fat_Content=='Low Fat'].value_counts()

df = df.to_frame().reset_index()

df["Item_Identifier"][df.Item_Identifier >= s.max()]

df.columns = ["Item_Id", "Item_Count"]   #RENAME COLUMNS

df["Item_Id"][df.Item_Count >= s.max()] 

df["New"]=df["Item_Id"].apply(lambda x:x[0:2])

t=df["Item_Id"][df.Item_Count >= s.max()]

##  print(t[0][0:2])   #FIRST TWO CHARACTERS OF SERIES t

le = LabelEncoder()

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

le = LabelEncoder()

for i in var_mod:

    data[i] = le.fit_transform(data[i])
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

train = data.loc[data['source']=="TRAIN"]   

test = data.loc[data['source']=="TEST"]

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)

train.drop(['source'],axis=1,inplace=True)

train.to_csv("train_modified.csv",index=False)

test.to_csv("test_modified.csv",index=False)

mean_sales = train['Item_Outlet_Sales'].mean()