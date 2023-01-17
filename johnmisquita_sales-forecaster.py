import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

%matplotlib inline



from subprocess import check_output



!head -15000000 ../input/train.csv > small_test.csv



dtype_holiday={'date':'O', 'type':'O', 'locale':'O', 'locale_name':'O', 'description':'O','transferred': 'bool'} 

holiday_df=pd.read_csv("../input/holidays_events.csv",dtype=dtype_holiday)

#print(holiday_df,"holiday_df")

dfo_data={'date': 'O', 'dcoilwtico':'float64'}

dfo=pd.read_csv("../input/oil.csv",dtype=dfo_data)



data_type={'date': 'O','id': 'int64','item_nbr':'int64','onpromotion':'float64','store_nbr': 'int64','unit_sales':'float64'}

dft=pd.read_csv("small_test.csv",dtype=data_type)



def to_ordinal(x):

 d =datetime.strptime(x,'%Y-%m-%d')

 z=datetime.date(d)

 y=z.toordinal()

 return y

dft["days"]=dft["date"].apply(to_ordinal)

dft=dft.drop("date",axis=1)

dfo["days"]=dfo["date"].apply(to_ordinal)

dfo=dfo.drop("date",axis=1)

holiday_df["days"]=holiday_df["date"].apply(to_ordinal)

holiday_df=holiday_df.drop("date",axis=1)

name_set=set(holiday_df["type"])

name_list=list(name_set)

holiday_code={}

for x,y in zip(name_list,range(len(name_set))):

 holiday_code[y]=x

holiday_df["type"]=holiday_df["type"].replace(name_list,range(len(name_set)))

#print(holiday_df.head())    

#merged=dft.merge(dfo,on="date")

merged=dft.merge(holiday_df,on="days")

merged=merged.merge(dfo,on="days")

#print(merged.head())

merged=merged.drop(["id","locale","description","locale_name","transferred","onpromotion",],axis=1)

merged=merged.dropna()

#print(merged.head())

items2=set(merged["item_nbr"])

items2=list(items2)

df_item_1=merged[merged.item_nbr==items2[0]]

store_sets=set(df_item_1["store_nbr"])

store_sets=list(store_sets)



def change_(x):

    return holiday_code[x]

df_item_1["type"]=df_item_1["type"].apply(change_)

##store number 9, item number 638977

df_store_and_item=df_item_1[df_item_1.store_nbr==store_sets[0]]

#print(df_store_and_item.head())

sns.pointplot("days","unit_sales",data=df_store_and_item,hue="type")

sns.pointplot("days","dcoilwtico",data=df_store_and_item,color="navy")

plt.show()

##store number 10, item number 638977

df_store_and_item1=df_item_1[df_item_1.store_nbr==store_sets[1]]

#print(df_store_and_item1.head())

sns.pointplot("days","unit_sales",data=df_store_and_item1,hue="type")

sns.pointplot("days","dcoilwtico",data=df_store_and_item1,color="navy")

plt.show()

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

%matplotlib inline



dict_={'date':'O','item_nbr':'int64','onpromotion':'float64','store_nbr': 'int64'}

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

!head -15000000 ../input/train.csv > small_test.csv

#!head -2 ../input/holidays_events.csv > holiday_test.csv

dtype_holiday={'date':'O', 'type':'O', 'locale':'O', 'locale_name':'O', 'description':'O','transferred': 'bool'} 

holiday_df=pd.read_csv("../input/holidays_events.csv",dtype=dtype_holiday)

#print(holiday_df,"holiday_df")

data_type={'date': 'O','id': 'int64','item_nbr':'int64','onpromotion':'float64','store_nbr': 'int64','unit_sales':'float64'}

dft=pd.read_csv("small_test.csv",dtype=data_type)



def to_ordinal(x):

 d =datetime.strptime(x,'%Y-%m-%d')

 z=datetime.date(d)

 y=z.toordinal()

 return y

dft["days"]=dft["date"].apply(to_ordinal)

dft=dft.drop("date",axis=1)

name_set=set(holiday_df["type"])

holiday_df["days"]=holiday_df["date"].apply(to_ordinal)

holiday_df=holiday_df.drop("date",axis=1)

name_list=list(name_set)

holiday_code={}

for x,y in zip(name_list,range(len(name_set))):

 holiday_code[y]=x

holiday_df["type"]=holiday_df["type"].replace(name_list,range(len(name_set)))

#print(holiday_df.head())    

#merged=dft.merge(dfo,on="date")

merged=dft.merge(holiday_df,on="days")

#print(merged.head())

merged=merged.drop(["id","locale","description","locale_name","transferred","onpromotion",],axis=1)

items2=set(merged["item_nbr"])

items2=list(items2)

df_item_1=merged[merged.item_nbr==items2[0]]

store_sets=set(df_item_1["store_nbr"])

store_sets=list(store_sets)

df_store_and_item=df_item_1[df_item_1.store_nbr==store_sets[1]]

#print(df_store_and_item.head())

##store number 9, item number 638977

#print("before fitting") 

lables=df_store_and_item["unit_sales"]

#print(df_store_and_item.head())

for_features=df_store_and_item.drop(["unit_sales","item_nbr","store_nbr","type"],axis=1)

for_features_train,for_features_test,lables_train,lables_test=train_test_split(for_features,lables,test_size=0.30,random_state=42)



rbf_clf = SVR(kernel="rbf")

#linear_clf = SVR(kernel="linear")

#poly_clf = SVR(kernel="poly")

draw_rbf=rbf_clf.fit(for_features_train,lables_train).predict(for_features_test)

#draw_linear=linear_clf.fit(for_features_train,lables_train).predict(for_features_test)

df_store_and_item1=df_item_1[df_item_1.store_nbr==store_sets[1]]

lables1=df_store_and_item1["unit_sales"]

for_features1=df_store_and_item1.drop(["unit_sales","item_nbr","store_nbr","type"],axis=1)

for_features_train1,for_features_test1,lables_train1,lables_test1=train_test_split(for_features1,lables1,test_size=0.30,random_state=42)

draw1_rbf=rbf_clf.fit(for_features_train1,lables_train1).predict(for_features_test1)



def change_(x):

    return holiday_code[x]



for_plotting=pd.DataFrame(columns=["unit_sales","days"])

for_plotting["unit_sales"]=lables_test

for_plotting["days"]=for_features_test["days"]

#sns.factorplot("ordinal","unit_sales",data=for_plotting,kind="point")

a=plt.subplot()

a.scatter(for_plotting["days"],for_plotting["unit_sales"],color="black")

a.scatter(for_plotting["days"],draw_rbf,color="orange")

a.set_title("store 9")

plt.xlabel('days')

plt.ylabel('Unit_sales')

plt.show()



for_plotting1=pd.DataFrame(columns=["unit_sales","days"])

for_plotting1["unit_sales"]=lables_test1

for_plotting1["days"]=for_features_test1["days"]

b=plt.subplot()

b.scatter(for_plotting1["days"],for_plotting1["unit_sales"],color="black")

b.scatter(for_plotting1["days"],draw1_rbf,color="orange")

b.set_title("store 10")

plt.xlabel('days')

plt.ylabel('Unit_sales')

plt.show()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor







from subprocess import check_output







data_type={'date': 'O','id': 'int64','item_nbr':'int64','onpromotion':'float64','store_nbr': 'int64','unit_sales':'float64'}

dft=pd.read_csv("../input/train.csv",dtype=data_type,usecols=[1],parse_dates=[0])



dft['year']=pd.DatetimeIndex(dft['date']).year

dft['month']=pd.DatetimeIndex(dft['date']).month

dft['day']=pd.DatetimeIndex(dft['date']).day

dft['day']=dft['day'].astype(np.uint8)

dft['month']=dft['month'].astype(np.uint8)

dft['year']=dft['year'].astype(np.uint16)

dft=dft.drop("date",axis=1)





dft1 = pd.read_csv("../input/train.csv",dtype=data_type,usecols=[2,3,4])

train=pd.concat([dft,dft1],axis=1)





item_list=set(train["item_nbr"])

list_items=list(item_list)



day_sort=set(train["day"])

list_day=list(day_sort)



month_sort=set(train["month"])

list_month=list(month_sort)



store_sort=set(train["store_nbr"])

list_store=list(store_sort)



sort_by_item=train[train.item_nbr==list_items[0]]



sort_by_month=sort_by_item[sort_by_item.month==list_month[2]]



sort_by_store=sort_by_month[sort_by_month.store_nbr==list_store[9]]



sns.pointplot("day","unit_sales",data=sort_by_store,size=4, aspect=2)

plt.show()





lables=sort_by_store["unit_sales"]

for_features=sort_by_store.drop(["unit_sales","item_nbr","year","month"],axis=1)

for_features_train,for_features_test,lables_train,lables_test=train_test_split(for_features,lables,test_size=0.30,random_state=42)



clif2=GradientBoostingRegressor(n_estimators=100,learning_rate=1.0,loss='ls',random_state=0,max_depth=1).fit(for_features_train,lables_train)

draw=clif2.predict(for_features_test)



b=plt.subplot()

b.scatter(for_features_test["day"],lables_test,color="black")

b.scatter(for_features_test["day"],draw,color="orange")

#b.set_title("store 10, item 638977")

plt.xlabel('days')

plt.ylabel('Unit_sales')

plt.show()
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#dtype_holiday={'date':'O', 'type':'O', 'locale':'O', 'locale_name':'O', 'description':'O','transferred': 'bool'} 

#holiday_df=pd.read_csv("../input/holidays_events.csv",dtype=dtype_holiday)

#print(holiday_df,"holiday_df")

#dfo_data={'date': 'O', 'dcoilwtico':'float64'}

#dfo=pd.read_csv("../input/oil.csv",dtype=dfo_data)





data_type={'date': 'O','id': 'int64','item_nbr':'int64','onpromotion':'float64','store_nbr': 'int64','unit_sales':'float64'}

dft=pd.read_csv("../input/train.csv",dtype=data_type,usecols=[1],parse_dates=[0])

#print(dft.head())

dft['year']=pd.DatetimeIndex(dft['date']).year

dft['month']=pd.DatetimeIndex(dft['date']).month

dft['day']=pd.DatetimeIndex(dft['date']).day

dft['day']=dft['day'].astype(np.uint8)

dft['month']=dft['month'].astype(np.uint8)

dft['year']=dft['year'].astype(np.uint16)

dft=dft.drop("date",axis=1)

#print(dft.head())



dft1 = pd.read_csv("../input/train.csv",dtype=data_type,usecols=[2,3,4])

train=pd.concat([dft,dft1],axis=1)

#print(dft1.head())



item_list=set(train["item_nbr"])

list_items=list(item_list)



day_sort=set(train["day"])

list_day=list(day_sort)



month_sort=set(train["month"])

list_month=list(month_sort)



store_sort=set(train["store_nbr"])

list_store=list(store_sort)

#print(list_store)

sort_by_item=train[train.item_nbr==list_items[0]]

#print(sort_by_item.head())

sort_by_month=sort_by_item[sort_by_item.month==list_month[0]]

#print(sort_by_month)

#sort_by_day=sort_by_month[sort_by_item.day==list_day[3]]

#print(sort_by_day.head())

sort_by_store=sort_by_month[sort_by_month.store_nbr==list_store[9]]

#print(sort_by_store.head())

sns.pointplot("day","unit_sales",data=sort_by_store)

plt.show()



#sort_by_store=sort_by_store.drop("year",axis=1)

lables=sort_by_store["unit_sales"]

for_features=sort_by_store.drop(["unit_sales","item_nbr","year"],axis=1)

for_features_train,for_features_test,lables_train,lables_test=train_test_split(for_features,lables,test_size=0.30,random_state=42)

rbf_clf = SVR(kernel="rbf")

draw_rbf=rbf_clf.fit(for_features_train,lables_train).predict(for_features_test)



b=plt.subplot()

b.scatter(for_features_test["day"],lables_test,color="black")

b.scatter(for_features_test["day"],draw_rbf,color="orange")

#b.set_title("store 10, item 638977")

plt.xlabel('days')

plt.ylabel('Unit_sales')

plt.show()