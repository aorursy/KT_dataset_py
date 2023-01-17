# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns 
import matplotlib.pyplot as plt  
import warnings            
warnings.filterwarnings("ignore") 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")
data.info()
display(data.shape)
display(data.describe())
display(data.isnull().sum())

data.sample(5)
display(data["age"].unique())
display(data["sex"].unique())
display(data["generation"].unique())

plt.figure(figsize=(10,7))
sns.stripplot(x="year",y='suicides/100k pop',data=data)
plt.xticks(rotation=45)
plt.show()
year_scd=data[data["year"].between(1985,2015)]
scd_no=year_scd.groupby("year")["suicides_no"].sum()
scd_no
# plt.figure(figsize=(10,7))
# sns.stripplot(x=scd_no.index,y=scd_no)
# plt.xticks(rotation=45)
# plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(y=scd_no,
              x=scd_no.index,color='red',alpha=0.8)

plt.xlabel('Year',fontsize = 15,color='blue')
plt.ylabel('Total SUICIDES of Year ',fontsize = 15,color='blue')
plt.title('Global Total SUICIDES of Year Trend Over Time 1985-2015 ',fontsize = 20,color='blue')
plt.grid()
plt.show()
year_suicides=data[data["year"].between(1985,2015)]
suicides=year_suicides.groupby("year")["suicides/100k pop"].mean()
year=year_suicides["year"].unique()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(y=[i for i in suicides],
              x=year,data=year_suicides,color='red',alpha=0.8)

plt.xlabel('Year',fontsize = 15,color='blue')
plt.ylabel('SUICIDES(per 100K)  mean',fontsize = 15,color='blue')
plt.title('Global Suicides(per 100K)-trend over time 1985-2015 ',fontsize = 20,color='blue')
plt.grid()
plt.show()
data.info()
data.sample(5)
#male-female suicides per100k
gender_suicides=data[data["year"].between(1985,2015)]
gender=gender_suicides.groupby("sex")
male=gender.get_group("male")
female=gender.get_group("female")

#male-female Age('5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years')
age_male=male.groupby("age")["suicides/100k pop"].mean()
age_female=female.groupby("age")["suicides/100k pop"].mean()


display(gender_suicides.info())
print("male",male.shape)
print("female",female.shape)

 # 1985-2015 Suicides PER 100K Male-Female
    
fig1, ax1 = plt.subplots()
ax1.pie([male["suicides/100k pop"].mean(),female["suicides/100k pop"].mean()],labels=["Male","Female"], autopct='%1.1f%%',
         startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Suicides PER 100K Male-Female",fontdict={"color":"red"})

#Suicides all age(5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years)
age=data.groupby("age")["suicides/100k pop"].mean()

labels=['5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years'  ]
sizes=[age[0],age[1],age[2],age[3],age[4],age[5]]

fig1, ax0 = plt.subplots()
ax0.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Suicides PER 100K Age",fontdict={"color":"black"})


#male 
labels=['5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years'  ]
sizes=[age_male[0],age_male[1],age_male[2],age_male[3],age_male[4],age_male[5]]

fig1,ax2 = plt.subplots()
ax2.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Suicides PER 100K Male-Age",fontdict={"color":"green"})


#Female 
labels=['5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years'  ]
sizes=[age_female[0],age_female[1],age_female[2],age_female[3],age_female[4],age_female[5]]

fig1, ax3 = plt.subplots()
ax3.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Suicides PER 100K  Female-Age",fontdict={"color":"blue"})



plt.show()
#for male
suicides_male=male.groupby("year")["suicides/100k pop"].mean()
year_male=gender_suicides["year"].unique()
print("suicides/100k pop",suicides_male)
print(year_male)


#for female
suicides_female=female.groupby("year")["suicides/100k pop"].mean()
year_female=gender_suicides["year"].unique()
print("suicides/100k pop",suicides_female)
print(year_female)
f,ax1 = plt.subplots(figsize =(25,10))
sns.pointplot(y=[i for i in suicides_male],x=year_male,color='red',alpha=0.8)
sns.pointplot(y=[i*3 for i in suicides_female],x=year_female,color='green',alpha=0.8)

sns.pointplot(y=[i for i in suicides_female],x=year_female,color='blue',alpha=0.8)

plt.text(5.300,18.60,'male',color='red',fontsize = 17,style = 'italic')
plt.text(10.300,13.90,'3x female',color='green',fontsize = 18,style = 'italic')
plt.text(10.300,7.90,'orginal female',color='blue',fontsize = 18,style = 'italic')

plt.xticks(rotation=45)
plt.xlabel('Year',fontsize = 15,color='blue')
plt.ylabel('Suicides mean Values',fontsize = 15,color='blue')
plt.title('Global Suicides(per 100K)-(Male-Female) trend over time 1985-2015 ',fontsize = 20,color='blue')
plt.grid()
plt.show()
 # 1985-2015 Suicides Male-Female
    
    
fig1, ax1 = plt.subplots()
ax1.pie([male["suicides/100k pop"].mean(),female["suicides/100k pop"].mean()],labels=["Male","Female"], autopct='%1.1i%%',
         startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 total Suicides of year Male-Female",fontdict={"color":"red"})

#male-female Age('5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years')
MALE_SCD=male.groupby("age")["suicides_no"].sum()
FEMALE_SCD=female.groupby("age")["suicides_no"].sum()
MALE_SCD


#Suicides all age(5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years)
age=data.groupby("age")["suicides_no"].sum()

labels=['5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years'  ]
sizes=[age[0],age[1],age[2],age[3],age[4],age[5]]

fig1, ax0 = plt.subplots()
ax0.pie(sizes,labels=labels, autopct='%1.1i%%',
         startangle=180)
ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 total Suicides of year Age",fontdict={"color":"black"})


#male 
labels=['5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years'  ]
sizes=[MALE_SCD[0],MALE_SCD[1],MALE_SCD[2],MALE_SCD[3],MALE_SCD[4],MALE_SCD[5]]

fig1,ax2 = plt.subplots()
ax2.pie(sizes,labels=labels, autopct='%1.1i%%',
         startangle=180)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 total Suicides of year Male-Age",fontdict={"color":"green"})


#Female 
labels=['5-14 years','15-24 years', '25-34 years','35-54 years', '55-74 years','75+ years'  ]
sizes=[FEMALE_SCD[0],FEMALE_SCD[1],FEMALE_SCD[2],FEMALE_SCD[3],FEMALE_SCD[4],FEMALE_SCD[5]]

fig1, ax3 = plt.subplots()
ax3.pie(sizes,labels=labels, autopct='%1.1i%%',
         startangle=180)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 total Suicides of year Female-Age",fontdict={"color":"blue"})



plt.show()



#"1985-2015 total Suicides of year Male-Female-Age"
scd_female=female.groupby("year")["suicides_no"].sum()
scd_male=male.groupby("year")["suicides_no"].sum()


f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(y=scd_female,x=scd_female.index,color='red',alpha=.8)
sns.pointplot(y=3*scd_female,x=scd_male.index,color='black',alpha=.8)
sns.pointplot(y=scd_male,x=scd_male.index,color='blue',alpha=.8)

plt.text(15.5,5.15,'Female',color='red',fontsize = 17,style = 'italic')
plt.text(10.5,5.5,'3x female',color='black',fontsize = 18,style = 'italic')
plt.text(5.5,5.5,'Male',color='blue',fontsize = 18,style = 'italic')

plt.xlabel('Year',fontsize = 20,color='blue')
plt.ylabel('Total Suicides of year  Values',fontsize = 20,color='blue')
plt.title('Global Suicides -(Male-Female) trend over time 1985-2015 ',fontsize = 20,color='blue')
plt.grid()
plt.show()
#1985-2015 total populations of countries,and gdp_per_capita ($)

pop_gdp=data.groupby(["year","country"]).agg({"population":"sum","gdp_per_capita ($)":"mean"})
pop_gdp.info()
# pop_gdp.loc[1985:1985]

#max gdp_per_capita ($) ,Country 1985-2015
gdpC_max=pop_gdp.loc[1985:2016].nlargest(1,columns="gdp_per_capita ($)")
#min gdp_per_capita ($) ,Country 1985-2015
gdpC_min=pop_gdp.loc[1985:2016].nsmallest(1,columns="gdp_per_capita ($)")
display(gdpC_max)
display(gdpC_min)

fig=sns.jointplot(y='gdp_per_capita ($)',x='year',data=data)
fig=sns.jointplot(y='gdp_per_capita ($)',x='population',data=data)
plt.show()
sns.FacetGrid(data,hue='year',size=5).map(plt.scatter,'gdp_per_capita ($)','population').add_legend()
plt.show()
sns.lmplot(y="population", x="gdp_per_capita ($)", data=data)
plt.show()
data.sample(5)
sns.heatmap(data.corr(),annot=True,linecolor="black",lw=0.5)


scd_gdp=data.groupby(["year","country"]).agg({"suicides/100k pop":"sum","gdp_per_capita ($)":"mean"})
scd_gdp.info()
display(scd_gdp.sample(7))
x=scd_gdp.groupby("country").agg({"suicides/100k pop":"mean","gdp_per_capita ($)":"mean"})
sns.scatterplot(y="suicides/100k pop",x="gdp_per_capita ($)",size="suicides/100k pop",data=x)
x.corr()
x.head(3)
display(gdpC_max)
display(gdpC_min)
gdpC_max.index[0][1]
scd_gdp["country"]=[i[1] for i in scd_gdp.index]
scd_gdp_max=scd_gdp[scd_gdp["country"]==gdpC_max.index[0][1]]
scd_gdp_max["year"]=[i[0] for i in scd_gdp_max.index]
scd_gdp_max.head(3)

from plotly.offline import init_notebook_mode,iplot,plot
import plotly.graph_objs as go
import plotly.figure_factory as ff


new_scd_gdp_max = scd_gdp_max.loc[:,["suicides/100k pop","gdp_per_capita ($)"]]
# new_d_2015
new_scd_gdp_max["index"] = np.arange(1,len(new_scd_gdp_max)+1)
fig = ff.create_scatterplotmatrix(new_scd_gdp_max, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="'Luxembourg' 1985-2015 suicides/100k pop-gdp_per_capita ($)")
iplot(fig)





sns.FacetGrid(scd_gdp_max,hue="year",size=5).map(plt.scatter,'gdp_per_capita ($)','suicides/100k pop').add_legend()
plt.title("Luxembourg ")
plt.show()
scd_gdp_min=scd_gdp[scd_gdp["country"]==gdpC_min.index[0][1]]
scd_gdp_min["year"]=[i[0] for i in scd_gdp_min.index]


new_scd_gdp_min = scd_gdp_max.loc[:,["suicides/100k pop","gdp_per_capita ($)"]]
# new_d_2015
new_scd_gdp_min["index"] = np.arange(1,len(new_scd_gdp_min)+1)
fig = ff.create_scatterplotmatrix(new_scd_gdp_min, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="'Albania' 1985-2015 suicides/100k pop-gdp_per_capita ($)")
iplot(fig)



data.sample(3)
sns.FacetGrid(scd_gdp_min,hue="year",size=5).map(plt.scatter,'gdp_per_capita ($)','suicides/100k pop').add_legend()
plt.title("Albania ")
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(y=scd_gdp_min["suicides/100k pop"],x=scd_gdp_min.year,color='red',alpha=.8)
sns.pointplot(y=scd_gdp_max["suicides/100k pop"],x=scd_gdp_max.year,color='black',alpha=.8)


plt.text(10.5,5.5,'Albania ',color='red',fontsize = 17,style = 'italic')
plt.text(10.5,180.5,'Luxembourg ',color='black',fontsize = 18,style = 'italic')


plt.xlabel('Year',fontsize = 20,color='blue')
plt.ylabel('Total Suicides/100k pop of year  Values',fontsize = 20,color='blue')
plt.title('Global Suicides/100k pop-Albania-Luxembourg trend over time 1985-2015 ',fontsize = 20,color='blue')
plt.grid()
plt.show()
#suicides_No--gdp_per_capita ($)
scdn_gdp=data.groupby(["year","country"]).agg({"suicides_no":"sum","gdp_per_capita ($)":"mean"})
scdn_gdp["countrys"]=[i[1] for i in scdn_gdp.index ]
scdn_gdp.info()
display(scdn_gdp.head())
print("max suicides country")
display(scdn_gdp.groupby("countrys")["suicides_no"].sum().nlargest(1))
print("min suicides country")
display(scdn_gdp.groupby("countrys")["suicides_no"].sum().nsmallest(3))
data.sample(5)

generation=data.groupby(["generation","sex"]).agg({"suicides_no":"sum","population":"sum","suicides/100k pop":"sum"})
generation
generation.loc["Boomers"]
generation.loc["Boomers"].loc["male"][0]
#generation male suicidesno
labels=['Boomers','G.I. Generation', 'Generation X','Generation Z', 'Millenials','Silent'  ]
sizes=[generation.loc["Boomers"].loc["male"][0],
       generation.loc["G.I. Generation"].loc["male"][0],
       generation.loc["Generation X"].loc["male"][0],
       generation.loc["Generation Z"].loc["male"][0],
       generation.loc["Millenials"].loc["male"][0],
       generation.loc["Silent"].loc["male"][0]]
       

fig1,ax2 = plt.subplots()
ax2.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Generation Male-suicides count",fontdict={"color":"green"})

#generation male population
labels=['Boomers','G.I. Generation', 'Generation X','Generation Z', 'Millenials','Silent'  ]
sizes=[generation.loc["Boomers"].loc["male"][1],
       generation.loc["G.I. Generation"].loc["male"][1],
       generation.loc["Generation X"].loc["male"][1],
       generation.loc["Generation Z"].loc["male"][1],
       generation.loc["Millenials"].loc["male"][1],
       generation.loc["Silent"].loc["male"][1]]
       

fig1,ax2 = plt.subplots()
ax2.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Generation Male-Population",fontdict={"color":"green"})

plt.show()




#generation female suicides
labels=['Boomers','G.I. Generation', 'Generation X','Generation Z', 'Millenials','Silent'  ]
sizes=[generation.loc["Boomers"].loc["female"][0],
       generation.loc["G.I. Generation"].loc["female"][0],
       generation.loc["Generation X"].loc["female"][0],
       generation.loc["Generation Z"].loc["female"][0],
       generation.loc["Millenials"].loc["female"][0],
       generation.loc["Silent"].loc["female"][0]]
       

fig1,ax2 = plt.subplots()
ax2.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Generation Female-suicides count",fontdict={"color":"green"})

#generation female popolation
labels=['Boomers','G.I. Generation', 'Generation X','Generation Z', 'Millenials','Silent'  ]
sizes=[generation.loc["Boomers"].loc["female"][1],
       generation.loc["G.I. Generation"].loc["female"][1],
       generation.loc["Generation X"].loc["female"][1],
       generation.loc["Generation Z"].loc["female"][1],
       generation.loc["Millenials"].loc["female"][1],
       generation.loc["Silent"].loc["female"][1]]
       

fig1,ax2 = plt.subplots()
ax2.pie(sizes,labels=labels, autopct='%1.1f%%',
         startangle=180)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("1985-2015 Generation Female-population ",fontdict={"color":"green"})
plt.show()
generation.nlargest(5,columns="suicides_no")
generation.nsmallest(5,columns="suicides_no")
