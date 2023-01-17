# This Python 3 environment comes with many helpful analytics libraries installed

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
data = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-mat.csv")

data.head()
data.info()
data.columns
data.describe()
data.corr()
#correlation map



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data["free_time_level"] = ["high" if i>2 else "low" for i in data["freetime"] ]

data.loc[:10,["free_time_level","freetime"]]

data["study_time_level"] = ["high" if i>7 else  "normal" if 8>i>4 else "low" for i in data["studytime"] ]

data.loc[:10,["study_time_level","studytime"]]

data.info()
data.shape
print(data["Mjob"].value_counts(dropna=False))
print(data["Fjob"].value_counts(dropna=True))
data.describe()
data["study_time_level"] = ["high" if i>2 else "low" for i in data["studytime"] ]

data.loc[:10,["study_time_level","studytime"]]
data.boxplot(column="goout", by="study_time_level" )
df_Fjob= data.loc[data.Fjob == "health"]

df_Fjob
melted_dads=pd.melt(frame=df_Fjob, id_vars= "Fjob", value_vars=["Dalc","Walc"])

melted_dads
df1=data.head()

df2=data.tail()

concat_df_rows=pd.concat([df1,df2],axis=0)

concat_df_rows
df3=data["Walc"]

df4=data["Dalc"]

df_concat_coloum=pd.concat([df3,df4],axis=1)

df_concat_coloum
data.info()
data["sex"]=data["sex"].astype("category")

data.info()
data["G3"]=data["G3"].astype("float")

data.dtypes
data["none"]=[None if i=="M" else 1 for i in data["sex"]]

data.head()
data["none"].value_counts(dropna =False)
data["none"].dropna(inplace=True)

data["none"].value_counts(dropna =False) #NaN is gone
assert data["none"].notnull().all()
student = ["Granger","Potter","Weasley"]

grade = ["AA", "BB","CC"]

list_label= ["student","grade"]

list_col=[student,grade]

zipped=list(zip(list_label,list_col))

hogw_dict=dict(zipped)

hogw_df=pd.DataFrame(hogw_dict)

hogw_df
hogw_df["SnapeHate"] = [5,10,7] #How much you hate Snape out of 10

hogw_df
hogw_df["HagridLove"]=10 #Broadcasting entire column

hogw_df
df2={"student":["Longbottom","Malfoy","Lovegood","Chang","Thomas"],

     "grade": ["CC","CB","BA","BB","CC"],"SnapeHate":[10,4,8,7,7],"HagridLove":[8,0,8,7,7]}

df2=pd.DataFrame(df2)

hogw_df =hogw_df.append(df2,ignore_index=True)

hogw_df
time_list=["1979-09-19","1980-07-31","1980-03-01","1980-07-30","1980-07-05","1979-02-13","1979-05-30","1979-10-20"]

datetime_object=pd.to_datetime(time_list)

hogw_df["birthday"]=datetime_object

hogw_df=hogw_df.set_index("birthday")

hogw_df
print(hogw_df.loc["1980-07-31"])
print(hogw_df.loc["1979-05-30":"1980-07-05"]) #between September 9,1979 and July,5 1980
hogw_df.resample("A").mean() #resample the data according to year,mounth by calculating means. A=year
hogw_df.resample("M").mean() #M=mounth. A lot of nan because hogw_df does not include all months
#to fill NaN's, interpolate from first value



hogw_df.resample("M").first().interpolate("linear")
hogw_df.resample("M").mean().interpolate("linear")
df_alc=data.loc[:,["Walc","Dalc"]]

df_alc.plot()
df_alc.plot(subplots=True)

plt.show()
data.plot(kind="scatter", x="age", y="Walc")

plt.show()
data.plot(kind = "hist",y = "age",bins = 50,range= (15,22))

plt.show()
data.head() # We can see that place for index is empty
data["index"] = np.arange(1, len(data)+1)

data.head() #we've created coloumn named index starting from 1
data= data.set_index("index")

data.head()
data["Fjob"][3] #one way way of sellecting data
data.Fjob[3] #another way of sellecting data
data[["Dalc","Walc"]] #choosing some coloumns
data.loc[5,["Walc"]] #using loc
print(type(data["Walc"]))

print(type(data[["Walc"]]))
data.loc[1:10,"G1":"none"] #1 to 10 for rows, from G1 to None for coloumns
data.loc[10:1:-1,"G1":"none"] #reverse
data.loc[1:10,"G1":] #coloumns from G1 to end
boolean_variable= data["G1"]<10

data[boolean_variable]
f1= data.G2 < 10 #first filter

f2 =data["G1"] > 10 #second filter

data[f1&f2] #their intersection, student scoring high in first exam but low in second exam
# Filtering column based

data.goout[data.Dalc>2]  #going out degree of students with high workday alcohol consumption
def daily(n):

    return n/7

data["dailystudy"]= data.studytime.apply(daily) #gives us daily study time

data.head()
#using lambda function 

data["studydaily"]=data.studytime.apply(lambda x: x/7 )

data.head()
data["totalscore"]= data.G1+data.G2+data.G3

data.head()
print(data.index.name)
#change index name

data.index.name="#"

data.head()
data1=data.set_index(["Mjob","Fjob"])

data1.head(100)
data.groupby("sex").mean()
data.groupby("Fjob").max()
data.groupby("school").Walc.mean()