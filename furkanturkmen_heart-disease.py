# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
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
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
print(df.corr())

f,ax = plt.subplots(figsize=(10, 10)) 
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.show()

df.plot(kind='scatter', x="age", y='thalach',alpha = 0.5,color = 'red', grid = True)
plt.xlabel('age')           
plt.ylabel('thalach')
plt.title('age-thalach scatter')       
plt.show()  
print(df['age'].value_counts())

df.age.plot(kind = 'hist',bins = 100,figsize = (9,9))
plt.xlabel("Age")           
plt.show()
df.sex.plot(kind = 'hist',bins = 50,figsize = (9,9)) 
plt.xlabel("Sex")           
plt.show()
df.cp.plot(kind = 'hist',bins = 50,figsize = (9,9))
plt.xlabel("cp")           
plt.show()
x = df['target']>0     
X = df[x]

Z = df[np.logical_and(df['thalach']>180, df['sex']==1 )]
print(Z)
print(df.tail(20)) 
print(df.columns) 
print(df.shape) #data rows and columns sayilarini verir

print(df.info()) 
print(df['thal'].value_counts(dropna =False))  # if there are nan values that also be counted

print(df['ca'].value_counts(dropna =False))  # if there are nan values that also be counted
df.boxplot(column='age')
plt.show()
df.boxplot(column='chol')
plt.show()
df.boxplot(column='thalach')
plt.show()
df.boxplot(column='thalach', by ="sex") 
plt.show()
data_new = df.head() 
melted = pd.melt(frame=data_new, value_vars= ['chol','thalach'])
print(melted)
print(melted.pivot(columns = 'variable',values='value'))
data1 = df.head()
data2= df.tail()
data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # ignore_index = True bu dataframe yeni id ver demek
print(data_row)
data1 = df["age"].head()
data2 = df["thalach"].head()
data_col = pd.concat([data1,data2], axis = 1, ignore_index = True)
print(data_col)
print(df.dtypes)
'''
df['chol'] = df['chol'].astype('float')
df['age'] = df['age'].astype('float')
df.dtypes
'''
df1=df   
df1["age"].dropna(inplace = True)
assert  df['age'].notnull().all() 
df.info()
name = ["Abdullah", "Ramiz", "Ebru", "Sila", "Zeynep", "Akif", "Isa"]
age = [19, 23 ,29, 30, 21, 24, 20]
list_label = ["Name", "Age"]
list_col = [name, age]
print(list_col)
zipped = list(zip(list_label, list_col))
print(zipped)
data_dict = dict(zipped)
d_f2 = pd.DataFrame(data_dict)
d_f2["weight"] = [55, 68, 45, 47, 50, 60, 64] #dataframe yeni bir  column eklemek
d_f2["income"] = 100 # Broadcasting(Butun bir sutun olarak deger atama)
#d_f2["weight"] = 10
print(d_f2)
data1 = df.loc[:,["age","thalach","chol"]]
data1.plot()
plt.show()
data1.plot(subplots = True)
plt.show()
data2 = df.iloc[:,4:8]
data2.plot(subplots = True)
plt.show()
data2 = df.iloc[:,4:8]
data2.plot(subplots = False)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="age",y = "chol")
plt.show()
print(df.describe())
import warnings
warnings.filterwarnings("ignore") # suan icin salla
data3 = df.head()
date_list = ["2000-01-10","2000-02-10","2000-03-10","2002-03-15","2002-03-16"]
datetime_object = pd.to_datetime(date_list)
data3["date"] = datetime_object
# lets make date as index
data3= data3.set_index("date")
print(data3)

# Now we can select according to our date index
print(data3.loc["2002-03-16"])
print(data3.loc["2000-03-10":"2002-03-16"])

print(data3.resample("A").mean()) #icine A yazarak bunu yillara gore oldugunu bellirttik
print(data3.resample("M").mean()) #Bu sekilde aylara gore

print(data3.resample("M").first().interpolate("linear")) #datada bos olan yerleri linear sekilde arttirarak cozeriz
print(data3.resample("M").mean().interpolate("linear")) #ortalamasi ayni kalacak sekilde boslulari doldur.

