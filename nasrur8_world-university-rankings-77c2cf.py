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
data1=pd.read_csv('../input/cwurData.csv')
print(data1.info())

print(data1.head(50))

universities=data1['institution']
print(universities)
tr_uni=data1[data1.country=="Turkey"]
print(tr_uni)
import matplotlib.pyplot as plt
new=data1.head(20)
plt.scatter(new.country,new.institution,) #scatter plot example
plt.xlabel('countries')
plt.ylabel('universities')
plt.title('rank')
plt.show()
filtered=tr_uni[(tr_uni.world_rank)&(tr_uni.year==2015)]
plt.hist(filtered.world_rank,bins=20) #histogram
plt.title('Universities of Turkey')
plt.xlabel("rank of institutions")
plt.show()
#is there any university top of worl rank from turkey

print(tr_uni.world_rank < 100) #return boolean value


        
#1st, 2nd and 3th universities from turkey
ilk_3=tr_uni.institution[0:3]
print(ilk_3)


usa_uni=data1[data1.country=="USA"]
unis1=usa_uni.institution
unis2=tr_uni.institution
count_of_usa_uni=unis1.count() #573
count_of_tr_uni=unis2.count() #20
sizes=[573,20]
labels=['usa','tr']
colors=['red','blue']
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%')
plt.show()

#universities and patents will show together
list1=data1.institution.head(10) #only first 10 universities
list2=data1.patents.head(10)
z=zip(list1,list2)
z_list=list(z)
print(z_list)
avg=sum(data1.publications)/2100 #count of university
data1["publication_level"]=["high" if i<avg else "low" for i in data1.publications]
data1[["publication_level","institution"]]
data1.describe()
#melt()
exp_data=data1.head(20)    #copy first 20 datas for melt example
exp_data

melted_data=pd.melt(frame=exp_data, id_vars='institution',value_vars=['country','world_rank'])
melted_data
#create dataframes
frame1=data1.head()
frame2=data1.tail()

#conc in rows
conc_data_row = pd.concat([frame1,frame2],axis =0,ignore_index =True) #axis=0 add rows
conc_data_row
frame1 = data1['institution'].head()
frame2= data1[['world_rank','publications']].head()
conc_data_col = pd.concat([frame1,frame2],axis =1) #axis=1 : add columns
conc_data_col
#plotting all data
plot_example = data1[["world_rank","quality_of_education","score"]]
plot_example.plot()
# subplots
plot_example = data1[["world_rank","quality_of_education","score"]]
plot_example.plot(subplots=True)
plt.show()
# scatter plot  
plot_example.plot(kind = "scatter",x="world_rank",y = "score")
plt.show()
# hist plot  
plot_example.plot(kind = "hist",y ="score",bins = 50)
# indexing using square brackets
data1["institution"][1]
# using column attribute and row label
data1.institution[1]
# using loc accessor
data1.loc[1,["institution"]]
# Selecting only some columns
data1[["world_rank","institution"]]
#difference
print(type(data1["institution"]))     # series
print(type(data1[["institution"]]))   # data frames
#slicing
data1.loc[0:20,"institution":"national_rank"]   # 0 to 20 and institution to national rank
# Reverse slicing 
data1.loc[20:0:-1,"institution":"national_rank"] 
# From something to end
data1.loc[0:20,"publications":] 
#combining filters
filter1=data1.score>70
filter2=data1.country=='Japan'
data1[filter1&filter2]
data1[(data1.country=='France') & (data1.publications >=500)]
#group by,aggregation and sorting
data1.groupby('institution').patents.sum().sort_values(ascending=True)