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
data=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

data1=pd.read_csv("/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv")
#Exploratory Data Analysis(EDA)





data["test preparation course"].value_counts(dropna=False) #value_counts bir sütunda nelerin olduğu ve kaç adet bulunduğunu belirtir

# "dropna=False" Sütun değeri "None" Olanları göstermeyi sağlar
data.describe() #Belirtilen %(Yüzde) Değerleri Yukarıda Belirtilmiştir
# Outliers=Abartı değerler için kullanılan değerlendirme sistemidir

# Q1-1.5(Q3-Q1) Bu değerden küçükse Outliers tir

# Q3+1.5(Q3-Q1) Bu değerden büyükse Outliers tir

# Aşşağıda gösterildiği gibi grafiğe de yansıtılabilir

data.boxplot(column=["math score"],by=["reading score"])
#Tidy and Pivoting Data



#Tidy Data

#Bazı sütunları silerek karşılaştırmayı sağlar

newdata1=data1.head(10)

newdata2=pd.melt(frame=newdata1,id_vars="DEP_TIME",value_vars=["ARR_TIME"])

print(newdata2)



#Pivoting Data

#Yukarıdda oluşturulan grafiğimn geri kazandırılmasını sağlar

newdata2.pivot(index = 'DEP_TIME', columns = 'variable',values='value')

# Concatenating Data And Data Type



#İki listeyi dikey olarak birleştirme

DATA=data1.head()

DATA1=data1.tail()

yeniveri=pd.concat([DATA,DATA1],axis=0,ignore_index=True)

print(yeniveri)

# ignore_index=Kendine göre bir index sıralaması oluşturur
#İki listeyi yatay olarak birleştirme

yeniveri1=pd.concat([DATA,DATA1],axis=1)

print(yeniveri1)
#İki sütun birleştirme

DATA2=data1["DISTANCE"].head()

DATA3=data1["DEST"].head()

yeniveri2=pd.concat([DATA2,DATA3],axis=1)

print(yeniveri2)
#Veri Tipini Değiştirme

data1["DEST_AIRPORT_ID"]=data1["DEST_AIRPORT_ID"].astype("float") 

data1["ORIGIN"]=data1["ORIGIN"].astype("category")

data1.dtypes
#Missing Data And Testing With Assert



#Sağlama işlemi gerçekleştirir.Bir işlemin doğruluğu veya yanlışlığı assert sayesinde belirlenebilir

gecici=data

gecici["test preparation course"].dropna(inplace=True)

gecici["test preparation course"].value_counts(dropna=False)

assert 1==1

assert gecici["test preparation course"].notnull().all()