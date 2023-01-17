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
data= pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head()
#bir liste yaratalım sonra onu dictionary'ye çevirelim, sonra da ondan bir dataframe yaratalım



Players= ["Steven Gerrard", "Alexandro Del Piero","Gabriel Batistuta","Clarence Seedorf","Edgar Davids"]

Nationality = ["England","Italy","Argentina","Cameroon","Nederlands"]

#players ve nationality olarak iki tane liste oluşturduk

list_label = ["Players","Nationality"]

List_Columns = [Players,Nationality]

#tablonun başlıklarını belirledik

zipped = list(zip(list_label,List_Columns))

#iki listeyi birbirine zipledik

data_dict = dict(zipped)

df = pd.DataFrame (data_dict)

df

#dataframeimizi oluşturduk
#bu dataframeimize yeni bir column yaratalım

df["Last Club"] =["LA Galaxy","Delhi Dynamos","Al Arabi","Botafogo","Barnet"]

df
### rage aralığı olan bir histogram grafiği çizelim

data1 = data.loc[:,["Overall"]]

data1.plot (kind="hist",y="Overall", bins=50, range= (85,100))
#kümülatif histogram çizelim

data1 = data.loc[:,["Overall"]]

data1.plot (kind="hist",y="Overall", bins=50, range= (85,100),cumulative = True)
#Yukarıda yaptığımız dataframein indexini datetime tipine çevirelim

birth_date_list=["30-05-1980","09-11-1974","01-02-1969","01-04-1976","13-03-1973"]

print(type(birth_date_list))

datetime_object = pd.to_datetime(birth_date_list)

print(type(datetime_object))
data2 = pd.DataFrame (data_dict)

birth_date_list=["30-05-1980","09-11-1974","01-02-1969","01-04-1976","13-03-1973"]

datetime_object = pd.to_datetime(birth_date_list)

data2["Birthday"] = datetime_object

data2 = data2.set_index ("Birthday")

data2
#bu dataframeimize yeni üç column daha yaratalım

data2["Value"] =["20000000","32000000","28000000","22000000","18000000"]

data2
#doğum tarihi 1976-01-04 olan futbolcuyu getirelim

print (data2.loc["1976-01-04"])
#doğum tarihi 1969 ve 1975 arasında olan futbolcuları getirelim

print (data2.loc["1969":"1975"])
data2.info()  #dataframeimizin data tiplerini görelim
data2["Value"] = pd.to_numeric(data2["Value"]) #Value datamızı numeric yapalım
data2.info()
data2
data= pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head(10)
# Oyuncuların doğum tarihlerini datetime formatında index olarak ekleyelim

data3 = data.head (10)

birth_date_list=["24-06-1987","05-02-1985","05-02-1992","07-11-1990","28-06-1991","07-01-1991","09-09-1985","24-01-1987","30-03-1986","07-01-1993"]

datetime_object = pd.to_datetime(birth_date_list)

data3["Birthday"] = datetime_object

data3 = data3.set_index ("Birthday")

data3


data3.resample("A").mean()
#NaN gelen değerleri de linear artan bir şekilde ortalama alarak dolduralım

data3.resample("A").mean().interpolate("linear")
#doğum tarihlerini yıla göre ayrıştırıp o yıllar içindeki datalardan max değerleri alalım (anladığım kadarıyla object olan değerler için max alınca alfabe sırasında en son agelen değeri yazdırıyor)

data3.resample("A").max()