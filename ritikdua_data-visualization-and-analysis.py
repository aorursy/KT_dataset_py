import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go
data=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

data.head()
data.isnull().sum()
data.drop(["HDI for year","country-year"],axis=1,inplace=True)

data["age_start"]=data["age"].str.split("-")

data["age_end"]=data["age"].str.split("-")

data.head(3)
data["age_start"]=data["age_start"].apply(lambda x: x[0])

data["age_end"]=data["age_end"].apply(lambda x: x[-1])


data["age_start"]=data['age_start'].str.extract('(\d+)').astype(str)

data["age_end"]=data['age_end'].str.extract('(\d+)').astype(str)



data.columns
data["age_start"]=data["age_start"].astype("int64")

data["age_end"]=data["age_end"].astype("int64")

data[" gdp_for_year ($) "]=data[" gdp_for_year ($) "].str.split(",")

data[" gdp_for_year ($) "]=data[" gdp_for_year ($) "].apply("".join)

data[" gdp_for_year ($) "]=data[" gdp_for_year ($) "].astype("int64")
data.info()
categorical=["generation","country","age","sex"]

numerical=data.columns ^ categorical

print(categorical)

print(numerical)
data[numerical].describe()
data[categorical].describe().head(4)
explodeUniversal=([0.1]*10) #create an explode lists for pie chart
fig,ax=plt.subplots(3,2,figsize=(21,17.5))

categories=list(categorical)

categories.remove("country")

index=0

for i in range(3):

  df=data.groupby([categories[index]]).sum();

  sns.barplot(y=df["suicides_no"],x=df.index,ax=ax[i][0]);



  labels = df.index

  ax[i][0].set_title(categories[index])

  

  ax[i][1].set_title(categories[index],fontsize=18)

  sizes=[df.iloc[x]["suicides_no"]/df.suicides_no.sum() for x in range(len(labels))]

  explode = list(explodeUniversal[:len(labels)])

    



  ax[i][1].pie(sizes, explode=tuple(explode), labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, textprops={'fontsize': 11})

  ax[i][1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



  index+=1

  

df=data.groupby(["year"]).sum()

fig,ax=plt.subplots(1,1,figsize=(11,6))

x=[]

y=[]

for i in range(1985,2017,4):

  s=0

  label=[]

  label.append(i)

  tmep=i

  for j in range(i,i+4):

  

    if (i>2016):

      break

    temp=j

    s+=df.loc[df.index==j]["suicides_no"][j]

  label.append(temp)

  y.append(s)

  x.append("-".join(str(k) for k in label))



plt.bar(x=x,height=y);
fig,ax=plt.subplots(1,2,figsize=(21,7.9))

df=data.groupby(["country"]).sum();

df=df.sort_values(by="suicides_no",ascending=False).head(10)

sns.barplot(y=df.index,x=df["suicides_no"],ax=ax[0])



countries=df.index



labels = countries

sizes=[df.iloc[i]["suicides_no"]/df.suicides_no.sum() for i in range(len(countries))]

explode = (0.1, 0.1, 0.1, 0,0,0,0,0,0,0.2) 



ax[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax[1].axis('equal') 



plt.show()

fig,ax=plt.subplots(1,2,figsize=(21,7.9))

df=data.groupby(["country"]).sum();

df=df.sort_values(by="suicides/100k pop",ascending=False).head(10)

sns.barplot(y=df.index,x=df["suicides/100k pop"],ax=ax[0])

# ax[0].set_xticklabels( df.index,rotation = 40, ha="right")





labels = df.index

sizes=[df.iloc[i]["suicides/100k pop"]/df["suicides/100k pop"].sum() for i in range(len(countries))]

explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

ax[0].set_xlim([7000,None])

ax[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
fig,ax=plt.subplots(1,1,figsize=(11,6.3))



# countries=countries

for i in range(len(countries)):

  d=data.loc[data['country']==countries[i]]

  df=pd.concat([df,d])

sns.barplot(data=df,x=" gdp_for_year ($) ",y="country",ax=ax);

# countries=np.array(df.index)[:5]

fig,ax=plt.subplots(3,2,figsize=(17,13.2))

index=0

for i in range(3):

  for j in range(2):

    if(i==2 and j==1):

      break

    sns.lineplot(x="year",data=data.loc[data["country"]==countries[index]],y="suicides_no",label=countries[index],ax=ax[i][j]);

    ax[i][j].set_ylabel("suicides_no");

    

    index+=1

ax[-1, -1].axis('off')



plt.tight_layout()

!pip install pycountry
import pycountry



df1=data

list_countries = data['country'].unique().tolist()

d_country_code = {} 



for country in list_countries:

    try:

        country_data = pycountry.countries.search_fuzzy(country)

        

        country_code = country_data[0].alpha_3

        d_country_code.update({country: country_code})

    except:

        print('could not add ISO 3 code for ->', country)

        

        d_country_code.update({country: ' '})



for k, v in d_country_code.items():

    df1.loc[(df1.country == k), 'iso_alpha'] = v



fig = px.choropleth(data_frame = df1,

                    locations= "iso_alpha",

                    color= "suicides_no", 

                    hover_name= "country",

                    color_continuous_scale= 'RdYlGn',  

                    animation_frame= "year")



fig.show()