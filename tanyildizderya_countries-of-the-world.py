# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/countries of the world.csv')
data.head()
threshold = sum(data.Population)/len(data.Population)
print("threshold: ",threshold)
data['population'] = ['high' if i > threshold else 'low' for i in data.Population ]
data.loc[:10,["Population","population"]]
data.info()
data.describe()
data.columns
data.boxplot(column="Area (sq. mi.)", by='GDP ($ per capita)')
new_data=data.head()
new_data
#melting data
melted_data=pd.melt(frame=new_data, id_vars='Country', value_vars=['Region','Population','Area (sq. mi.)'])
melted_data
#pivoting data
melted_data.pivot(index='Country',columns='variable', values='value')
#concatenating data
data1 = data.head()
data2 = data.tail()
concat_data = pd.concat([data1,data2],axis=0,ignore_index=True)
concat_data
data1=data['Country'].head()
data2=data['Region'].head()
concat_data=pd.concat([data1,data2],axis=1)
concat_data
data.dtypes
data.info()
#missing value
data['Literacy (%)'].value_counts(dropna=False)
data.info()
data1=data
data1['Literacy (%)'].dropna(inplace=True)
assert data['Literacy (%)'].notnull().all()
assert data.Population.dtypes == np.int
data.Population.dtypes
data1=data['Country'].head()
data2=data['Region'].head()
list_label=['Country', 'Region']
list_col=[data['Country'].head(), data['Region'].head()]
zipped=list(zip(list_label,list_col))
data_dict=dict(zipped)
df1=pd.DataFrame(data_dict)
df1
df1["Capital"]=["Kabil","Tiran", "Cezayir", "Pago Pago", "Andorra la Vella"]
df1
data1 = data['Area (sq. mi.)'].head()
data1.plot()
data2 = data['Population'].head()
data2.plot()
data.describe()
data.head()
#TIME SERIES
data2=data.head()
time_list=["2018-04-12","2018-05-12","2018-06-12","2018-07-12","2018-08-12"]
datetime_object = pd.to_datetime(time_list)
data2["date"] = datetime_object
data2 = data2.set_index("date")
data2
data2.loc["2018-04-12" : "2018-07-12"]
data2.resample("A").mean()
#Annual
data2.resample("M").mean()
#monthly
data2.resample("M").first().interpolate("linear")
data.head()











































