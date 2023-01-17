# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization

import seaborn as sns # EDA

from collections import Counter

import warnings

warnings.filterwarnings('ignore')

#import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")
data.info #We learn what data has
data.columns #we learn data's features
data.head(8)
#We will practise how can we use 'lineplot'

data.chol.plot(kind="line",color="green",label="chol",grid=True,linestyle=":")

data.thalach.plot(kind="line",color="purple",label="thalach",grid=True)

data.age.plot(kind="line",color="pink",label="age",grid=True)

data.trestbps.plot(kind="line",color="orange",label="trestbps",grid=True)

plt.legend(loc="upper right") #legend: puts feature label into plot

plt.xlabel("indexes")

plt.ylabel("Features")

plt.title("Heart Diseases")

plt.show()
#We will practise how can we use Scatter Plot

#I want to compare 'age' and 'chol'. Is there any connection with these features?

data.plot(kind="scatter", x="age", y="chol", alpha= 0.5, color="brown")

plt.xlabel("age")

plt.ylabel("chol")

plt.title("age and chol with Scatter Plot")

plt.show()
#Histogram shows frequency of feature to us.

data.cp.plot(kind="hist",bins=50,figsize=(5,5))

plt.show()
# My first filter -> logical and

data[(data['cp']>2) & (data['sex']<1)]
# My second filter -> logical or

data[(data['age']>70) | (data['ca']>2)]
# Lets think of every feature as a list. 

print(data.age[0])

print('')

# we can see all ages with enumerate

for index, value in enumerate(data.age[0:5]): 

   print(index," : ",value)

print('')

# other option is iterrows(). We can see 5 sexes with details.    

for index, value in data[['sex']][0:5].iterrows(): 

     print(index," : ",value)
#usage of tuble func.

i=3; #global  variable(scope)

def tuble_ex():

  # there is no local scope however 't' has 3 variables. Because we have global scope.

    t=(data.age[i],data.age[i+1],data.age[i+2])

    return t

a,b,c = tuble_ex() # 3 variables returned.

print(a,b,c)

print(tuble_ex())
#Nested func. practise with oldpeak and cp values. 

for i in range(6): 

    def square():

        def add(): 

            x= data.oldpeak[i]

            y= data.cp[i]

            z= x+y

            return z

        return add()**2

    if square() > 5 and square() < 10:

        print(square(),"Normal")

    else:

        print(square(),"Critical")
#Map,Lambda,zip() practise

x= map(lambda k:k**2,data.sex[0:5])

y= map(lambda k:k,data.age[0:5])

#print(x,y)

list1= list(x)

list2= list(y)

z= zip(list1,list2)

z_list=list(z)

print(z_list)
# I can find the average age of patients

# I want to create new section about probabilities according to ages.

threshold = sum(data.age)/len(data.age)

print("Threshold:",threshold)

#list comprehension

data["probability"]= ["high" if i>threshold else "low" for i in data.age]

data.loc[:10,["probability","age"]]
data.info()
# I want to learn frequency of ages

print("age","frequency")

print(data.age.value_counts(dropna=False))
#If we want to find outlier datas, we should use describe() method.

#Its easier to see statistical calculations.

#Also its ignore null entries.

data.describe()

#We can see visualization of statistical calculations.

data.boxplot(column="age", by="sex")

# ages value by sex

plt.show()
#Lets practise how can we melt of data_new.

data_new = data.head()

data_new
melted = pd.melt(frame=data_new,id_vars='age',value_vars=['restecg','exang'])

melted
# rivot: reverse of melting

melted.pivot(index='age',columns='variable',values='value')
#How can we concatenate to dataframes?

data1= data.head()

data2= data.tail()

conc_data_row= pd.concat([data1,data2],axis=0,ignore_index=True) 

conc_data_row
data1= data.sex.head()

data2= data.chol.head()

conc_data_col= pd.concat([data1,data2],axis=1) #axis=1: adds df in column

conc_data_col
data.dtypes
#I want to convert object to categorical and float to int.

data.probability= data.probability.astype('category')

data.oldpeak=data.oldpeak.astype('int')

data.dtypes
#Lets find missing values. 

data.info()

#As we can see this dataframe hasn't got any null entry or missing value. What am I lucky!!!✔

#I think,this dataset has already been cleared but anyway..
# We can check lots of things with 'assert'

assert data.target.notnull().all()

#returns nothing it means we don't have any nan values.
assert data.slope.dtypes == np.int
# high level filter :d

filter1=data.thal>2

filter2= data.slope<1

data.age[filter1 & filter2]
#I just want to practise usage of plain python func.

def cross(n):

    return n*2

data["new"]= data.age.apply(cross)+data.cp

#data.new

data.loc[:5,"slope":]
# Setting index: type1 is outer, type 2 is inner index

data1=data.set_index(["age","chol"])

data1.head(10)
# Lets group all features according to cp

data.groupby("cp").mean()

#data.groupby("cp").age.max()
# How many patients fasting blood sugar is > 120 mg/dl? (true=1, false=0)

data.fbs.value_counts(dropna=False)
# Which age range is more likely to be a heart patient?

age_list= list(data.age)

# I am gonna use 'Counter' method. We should import it at the beginning.

age_count= Counter(age_list)

most_common_age= age_count.most_common(15)

x,y = zip(*most_common_age)

x,y = list(x), list(y)



#Visualization

plt.figure(figsize=(15,5))

sns.barplot(x=x, y=y, palette= sns.cubehelix_palette(len(x)))

plt.ylabel('Frequency')

plt.xlabel('Ages')

plt.title('Most common ages of heart patients')
# As we can see;

data.age.value_counts()
#Lets find type of heart attack by all ages. (cp values)

age_list=list(data.age.unique())

#cp_list= list(data.cp.unique())

cp_zero=[]

cp_one=[]

cp_two=[]

cp_three=[]

for i in age_list:

    x= data[data['age']==i]

    cp_zero.append(sum(x.cp==0)/len(x))

    cp_one.append(sum(x.cp==1)/len(x))

    cp_two.append(sum(x.cp==2)/len(x))

    cp_three.append(sum(x.cp==3)/len(x))

#Visualization

f,ax= plt.subplots(figsize=(15,9))

sns.barplot(y=cp_zero,x=age_list,color='purple',alpha=0.5,label='Type 0')

sns.barplot(y=cp_one,x=age_list,color='green',alpha=0.7,label='Type 1')

sns.barplot(y=cp_two,x=age_list,color='yellow',alpha=0.6,label='Type 2')

sns.barplot(y=cp_three,x=age_list,color='blue',alpha=0.6,label='Type 3')



ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Ages', ylabel='Cp values', title='Type of heart attack by age')
#As we can see;

data.loc[:20,["age","cp"]]
#Sorted Chol values by age

age_list= list(data.age.unique())

chol_ratio=[]

for i in age_list:

    x=data[data['age']==i]

    chol_rate=sum(x.chol)/len(x)

    chol_ratio.append(chol_rate)

datac= pd.DataFrame({'age_list': age_list,'chol_ratio': chol_ratio})

new_index=(datac['chol_ratio'].sort_values(ascending=False)).index.values

sorted_data=datac.reindex(new_index)

sorted_data.head()
#Sorted trestbps values by age

age_list= list(data.age.unique())

tbps_ratio=[]

for i in age_list:

    x=data[data['age']==i]

    tbps_rate=sum(x.trestbps)/len(x)

    tbps_ratio.append(tbps_rate)

datat= pd.DataFrame({'age_list': age_list,'tbps_ratio': tbps_ratio})

new_index=(datat['tbps_ratio'].sort_values(ascending=False)).index.values

sorted_data2=datat.reindex(new_index)

sorted_data2.head()
#We have values in two different ranges so I normalized them.

sorted_data['chol_ratio']=sorted_data['chol_ratio']/max(sorted_data['chol_ratio'])

sorted_data2['tbps_ratio']=sorted_data2['tbps_ratio']/max(sorted_data2['tbps_ratio'])

data_all=pd.concat([sorted_data,sorted_data2['tbps_ratio']],axis=1)

data_all.sort_values('chol_ratio',inplace=True)

data_all.head()
#Visualization with point plot

f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x=data_all['age_list'],y=data_all['chol_ratio'],data_all=data_all,color='lime',alpha=0.8)

sns.pointplot(x=data_all['age_list'],y=data_all['tbps_ratio'],data_all=data_all,color='purple',alpha=0.8)

plt.text(40,0.58,'chol ratio',color='lime',fontsize=18,style='normal')

plt.text(40,0.55,'trestbps ratio',color='purple',fontsize=18,style='normal')

plt.xlabel('Ages',fontsize=15,color='orange')

plt.ylabel('Values',fontsize=15,color='orange')

plt.title('Chol vs Trestbps Values',fontsize=20,color='orange')

plt.grid()
# Visualization with joint plot

from scipy import stats 

#I include it for see pearsonr value.

g= sns.jointplot(data_all['age_list'],data_all['tbps_ratio'],kind="kde",height=5)

g = g.annotate(stats.pearsonr)

plt.savefig('graph.png')

plt.show()
g= sns.jointplot(data_all.chol_ratio,data_all.tbps_ratio,height=5,ratio=3,color="purple")

g = g.annotate(stats.pearsonr)
data_all.head()
#Linear model

sns.lmplot(x="chol_ratio",y="age_list",data=data_all)

plt.show()
sns.kdeplot(data_all.age_list,data_all.chol_ratio,shade=True,cut=5,color="brown")

plt.show()
pal= sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=data_all.loc[:,["chol_ratio","tbps_ratio"]],palette=pal,inner="points")

plt.show()

#It means the most value we have is about 0.8 for chol_ratio and the most value we have is between 0.8 and 0.9 for tbps_ratio. 
data_all.corr()

#As we can see; values of data_all are positive correlation so values are directly proportional.

#The similarity between chol and tbps value are very low.
#Correlation map

f,ax= plt.subplots(figsize=(5,5))

sns.heatmap(data_all.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
#I want to learn chest pain type (cp) according in heart data.

labels= data.cp.value_counts().index

colors=["orange","pink","brown","gray"]

explode= [0,0,0,0]

sizes=data.cp.value_counts().values

#Visualization with pie plot

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Chest pain type(cp) according in heart.csv',color="blue",fontsize=15)
data.head(10)
#I want to learn age, gender and chest pain type correlations.

sns.boxplot(x="sex", y="age", hue="cp", data=data, palette="PRGn")

plt.title("0: female, 1:male",color="gray")

plt.show()
#Other option is;

sns.swarmplot(x="sex", y="age", hue="cp", data=data)

plt.title("0: female, 1:male",color="gray")

plt.show()
#I used swarmplot to learn about age,gender and the possibility of having heart disease.

sns.swarmplot(x="sex", y="age", hue="probability", data=data)

plt.title("0: female, 1:male",color="gray")

plt.show()
data.ca.value_counts()
#Lets see "ca" value(number of major vessels (0-3) colored by flourosopy).

plt.figure(figsize=(10,6))

count= data.ca.value_counts()

sns.barplot(x=count.index, y=count.values)

plt.ylabel("Number of ca")

plt.xlabel("Ca values")

plt.title("Ca values in data", color="black", fontsize="12")
sns.pairplot(data_all.loc[:,["chol_ratio","age_list"]])

plt.show()
sns.pairplot(data.loc[:,["chol","age","ca","oldpeak"]])

plt.show()