import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly 







data=pd.read_csv('/kaggle/input/suicides-rate-1985-to-2016/master.csv')
data.head()
columns= data.columns

# columns

print("------------------------")

print("Columns of The data Set")

print("------------------------")

for i in columns:

    print("-" , i)
shape=data.shape

print(f'Columns: {shape[1]}')

print(f'Rows: {shape[0]}')

data.info()
data.isnull().sum()
data['HDI for year']= data['HDI for year'].fillna(data['HDI for year']).mean()

data.drop(['country-year'],axis=1 , inplace=True)
countries=data.country.unique()

print("-----------------------------------")

print("Name of Attempted Suicides Coutries")

print("-----------------------------------")



for i in countries:

    print("-",i,end="|")
print("---------------------------------")

print("Total no of suicides in countries")

print("---------------------------------")

suicides_no=data.suicides_no.sum()

print("Total Suicides : " , suicides_no)
years=data.year.unique()

for i in years:

    print(i, end=" |")
data.describe()
data.groupby(['sex'])['suicides_no'].sum()
plt.figure(num=None, figsize=(8, 6))

data.groupby(['sex'])['suicides_no'].sum().plot(kind='bar' ,color='#007bff')

plt.title("Gender - Male | Female \n" ,  fontsize=15)

plt.xlabel("Gender" , fontsize=14)

plt.ylabel("Number of Male and Female" , fontsize=14)

plt.show()
data.groupby(['age'])['suicides_no'].sum()
plt.figure(num=None, figsize=(8, 6))

data.groupby(['age'])['suicides_no'].sum().plot(kind='bar', color='#fd7e14')

plt.title("Ages Range of Commite Suicide \n" ,  fontsize=15)

plt.xlabel("Ages" , fontsize=14)

plt.ylabel("Number of Male and Female" , fontsize=14)

plt.show()
data.groupby(data['generation'])['suicides_no'].sum()
plt.figure(num=None, figsize=(8, 6))

data.groupby(data['generation'])['suicides_no'].sum().plot(kind='bar', color='#4FC3F7')

plt.title(" Generation of Commite Suicide \n" ,  fontsize=15)

plt.xlabel("Genderation" , fontsize=14)

plt.ylabel("Number of Male and Female" , fontsize=14)

plt.show()
data.groupby('year')['suicides_no'].sum().head()
plt.figure(num=None, figsize=(13, 6))

data.groupby('year')['suicides_no'].sum().plot(kind='bar', color='#4FC3F7')

plt.title(" Number of Suicides In each Year\n" ,  fontsize=15)

plt.xlabel("Years" , fontsize=14)

plt.ylabel("Number of Suicides" , fontsize=14)

plt.show()
data.groupby(['year'])['gdp_per_capita '].sum().head()
plt.figure(num=None, figsize=(13, 6))

data.groupby(['year'])['gdp_per_capita '].sum().plot(kind='line', color='#01579B')

plt.title(" Growth  Development Product \n" ,  fontsize=15)

plt.xlabel("Years" , fontsize=14)

plt.ylabel("Growth  Development Product" , fontsize=14)

plt.show()
Year_age=pd.DataFrame(data.groupby(['age','year'])['suicides_no'].sum().unstack())
MainGraph= Year_age.T

MainGraph.loc[:,:].plot(kind='bar',stacked = True, figsize=(14,6))

plt.legend(bbox_to_anchor=(1,1), title = 'Age group')

plt.title('Suicide number by year')

plt.xlabel('Year')

plt.ylabel('Suicide number')
Year_sex=data.groupby(['sex','year'])['suicides_no'].sum().unstack()
MainGraph= Year_sex.T

MainGraph.loc[:,:].plot(kind='bar',stacked = True, figsize=(14,6))

plt.legend(bbox_to_anchor=(1,1), title = 'Age group')

plt.title('Suicide number by year')

plt.xlabel('Year')

plt.ylabel('Suicide number')
SYA=pd.DataFrame(data.groupby(['age','sex','year'])['suicides_no'].sum().unstack())
#male

gsdm = pd.DataFrame(SYA.iloc[[1,3,5,7,9,11],:])

gsdm



#female

gsdf = pd.DataFrame(SYA.iloc[[0,2,4,6,8,10],:])

gsdf


for i in range(1985,2016):

    plt.figure(num=None, figsize=(8, 6))

    gsdm.loc[:,i].plot(kind='bar', color = ('skyblue'))

    plt.xticks(range(6),['15-24 years','25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'],

               rotation = 60)

    plt.xlabel('Age group')

    plt.ylabel('Suicide number')

    plt.title('Suicide population of male in '+ str(i))

    

    plt.show()


for i in range(1985,2016):

    plt.figure(num=None, figsize=(8, 6))

    gsdf.loc[:,i].plot(kind='bar', color = ('#FFDD3C'))

    plt.xticks(range(6),['15-24 years','25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'],rotation = 60)

    plt.xlabel('Age group')

    plt.ylabel('Suicide number')

    plt.title('Suicide population of female in'+ str(i))

    

    plt.show()
Age_sex=data.groupby(['age','sex'])['suicides_no'].sum().unstack()

gsd02_02 = Age_sex.T.sum()

age=['05-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']

colors = ['darkviolet','dodgerblue','turquoise','mediumblue','mediumseagreen','lawngreen']

fig = plt.subplots(figsize = (8,6))

plt.pie(gsd02_02,

        colors=colors,

        

               labels = age,

               autopct = '%.1f%%',

               startangle =0,

               radius = 1.5,

               frame = 0,

               center = (4.5,4.5),

               explode=(0.2,0.1,0,0,0,0),

#                shadow=True

               )

plt.legend(age,fontsize=10) 

plt.show()
sexsum = Age_sex

sexsum = pd.DataFrame(sexsum.sum())

sexsum = sexsum.reset_index()
sexsum.head()
plt.figure(num=None, figsize=(8, 6))

Age_sex.iloc[:,1].plot(kind='bar', color='skyblue', width = 1, figsize=(8,5))

Age_sex.iloc[:,0].plot(kind='bar', color='#FFDD3C', width = 1, figsize=(8,5))

plt.ylabel('Suicide number')

plt.xlabel('Age group')

plt.xticks(rotation = 60)

plt.title('Suicide number')

plt.legend(['Male','Female'], bbox_to_anchor=(1, 1),title = 'Sex')

plt.show()


By_year=data.groupby(['year','country'])['suicides_no'].sum().unstack()

By_year['Suicide number'] = By_year.sum(axis=1)



By_year.loc[:,'Suicide number'].plot(kind='line',figsize=(10,6),marker='o')

plt.title('Suicide number from 1985 to 2016')

plt.xlabel('year')

plt.ylabel('suicide number')

plt.show()
plt.figure(num=None, figsize=(8, 6))

Top10Country= data.groupby(['country','sex'])['suicides_no'].sum().unstack()/1000000

Top10Country['Suicide number']=Top10Country.apply(lambda Top10Country: Top10Country['female']+Top10Country['male'], axis = 1)

Top10Country = Top10Country.sort_values(by='Suicide number',ascending=False)







Top10Country.iloc[0:10,2].plot(kind='barh')

plt.ylabel('Country',fontsize=15)

plt.xlabel('Suicide number (million)',fontsize=15)

plt.title('Top 10 country of suicide from 1987-2016',fontsize=17)



plt.show()

plt.figure(num=None, figsize=(8, 6))



Country100k= data.groupby(['country','sex'])['suicides/100k pop'].sum().unstack()

Country100k['Suicide number']=Country100k.apply(lambda Country100k: Country100k['female']

                                                    +Country100k['male'], axis = 1)

Country100k = Country100k.sort_values(by='Suicide number',ascending=False)

Country100k.head(10)



Country100k.iloc[0:10,2].plot(kind='barh')

plt.ylabel('Country',fontsize=15)

plt.xlabel('Suicide population (per 100k)',fontsize=15)

plt.title('Top 10 country of suicide from 1987-2016',fontsize=17)



plt.show()

fig = plt.subplots(figsize = (10,6))

percountry10 = (Top10Country.iloc[0:10,2])



top10country = ["Russian Federation","Unites States","Japan","France","Ukraine","Germany","Republic of Korea","Brazil","Poland","United Kingdom"]



plt.pie(percountry10,

               labels = top10country,

               autopct = '%.1f%%',

               startangle =0,

               radius = 1.5,

               frame = 0,

               center = (4.5,4.5),

               explode=(0.2,0,0,0,0,0,0,0,0,0),



               )

plt.legend(top10country,fontsize=10,loc="upper center")

plt.show()