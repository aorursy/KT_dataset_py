!pip install chart_studio
import csv

import pprint



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import matplotlib as ml

import seaborn as sns

import plotly

import chart_studio.plotly as py

import warnings

from scipy import stats


data = list(csv.DictReader(open("../input/suicide-rates-overview-1985-to-2016/master.csv")))

data2 =  pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv", thousands=r',')

#data2 =  pd.read_csv("suicide-rates-overview-1985-to-2016\master.csv")
#printing data description

print(data2.describe())
data2.head()
#Seems like HDI for year from above has quite some Null values

#Hence replacing the null values with 0

data2["HDI for year"].replace(np.nan,0, inplace=True)


#print(data2.head())

data2.columns = [c.replace('$', '').replace('(', '').replace(')', '').strip().replace(' ', '_').replace('/','_') for c in data2.columns]

print(data2.describe())
import matplotlib.pyplot as plt



data2.groupby(['country']).suicides_no.mean().nlargest(10).plot(kind='barh')

plt.xlabel('Average Suicides_no', size=20)

plt.ylabel('Country', fontsize=20);

plt.show()

plt.clf()

plt.cla()

plt.close()





data2.groupby(['country']).suicides_no.sum().nlargest(10).plot(kind='barh')

plt.xlabel('Total Suicides_no', size=20)

plt.ylabel('Country', fontsize=20);

plt.show()

plt.clf()

plt.cla()

plt.close()





data2.groupby(['country']).suicides_100k_pop.sum().nlargest(10).plot(kind='barh')

plt.title('Top 10 country of suicide per 100k from 1987-2016')

plt.xlabel('Total Suicides per 100k', size=20)

plt.ylabel('Country', fontsize=20);

plt.show()

plt.clf()

plt.cla()

plt.close()
data2.columns = [c.replace('$', '').replace('(', '').replace(')', '').strip().replace(' ', '_').replace('/','_') for c in data2.columns]

#Print all countries

countries = data2.country.unique()
year = data2.groupby('year').year.unique()







totalpyear = pd.DataFrame(data2.groupby('year').suicides_no.sum())



plt.plot(year.index[0:31], totalpyear[0:31])

plt.xlabel('year', fontsize=14)

plt.ylabel('Total number of suicides in the world', fontsize=14)
year = data2.groupby('year').year.unique()







totalpyear = pd.DataFrame(data2.groupby('year').suicides_100k_pop.sum())

#plt.figure(9)

plt.plot(year.index, totalpyear)

plt.xlabel('year')

plt.ylabel('Total number of suicides per 100k in the world')

plt.show()

plt.clf()

plt.cla()

plt.close()
labels = 'Male', 'Female'

values = [np.sum(data2[data2.sex.eq("male")].suicides_no), np.sum(data2[data2.sex.eq("female")].suicides_no)]

fig1, ax1 = plt.subplots()

ax1.pie(values,  labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title('Distribution of suicides by Gender')

plt.show()

plt.clf()

plt.cla()

plt.close()
labels = '5-14 years', '15-24 years','25-34 years','35-54 years','55-74 years','75+ years'

values =[]

for eachlab in labels:

    values.append(np.sum(data2[data2.age.eq(eachlab)].suicides_no))



fig1, ax1 = plt.subplots()

ax1.pie(values,  labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title('Distribution of suicides by Age')

plt.show()

plt.clf()

plt.cla()

plt.close()
countries = data2['country'].unique()



data2.columns = [c.replace('$', '').replace('(', '').replace(')', '').strip().replace(' ', '_') for c in data2.columns]





        #print(data2[data2.country.eq(eachcon)].population)



from matplotlib.pyplot import figure



def getvalofcountries(data, column_key):

    print(column_key)

    values = []

    for eachcon in countries:

        if column_key == 'population':

            values.append(np.sum(data2[data2.country.eq(eachcon)].population))

        elif column_key == 'HDI_for_year':

            values.append(float(data2[data2.country.eq(eachcon)].HDI_for_year.iloc[0]))

        elif column_key == 'gdp_for_year':

            values.append(float(data2[data2.country.eq(eachcon)].gdp_for_year.iloc[0]))



    df = pd.DataFrame(values, index = countries,columns =['values'])

    df = df[(df.values != 0)] # remove empty values

    df.plot(kind='barh',figsize=(6,20))

    plt.xlabel('Values', size=20)

    plt.ylabel('Country', fontsize=20)

    plt.title("Country vs "+column_key)

    #plt.show()

    figure(figsize=(200,10))

    plt.show()

    plt.clf()

    plt.cla()

    plt.close()

    



getvalofcountries(data2,'population')



getvalofcountries(data2,'HDI_for_year')
getvalofcountries(data2,'gdp_for_year')
#Total suicide number by year

gsdtt=pd.DataFrame(data2.groupby(['age','year'])['suicides_100k_pop'].sum().unstack())

gsdtt = gsdtt.fillna(0)

gsdtt





Tgsdtt = gsdtt.T

Tgsdtt.iloc[:,:].plot(kind='bar',stacked = True, figsize=(10,6))

plt.legend(bbox_to_anchor=(1,1), title = 'Age group')

plt.title('Suicide number by year')

plt.xlabel('Year')

plt.ylabel('Suicide number')

warnings.filterwarnings('ignore')

plt.show()

plt.clf()

plt.cla()

plt.close()
#Group data by age gender of each year

gsd=pd.DataFrame(data2.groupby(['age','sex','year'])['suicides_no'].sum().unstack())

gsd = gsd.fillna(0)

gsd



#male

gsdm = pd.DataFrame(gsd.iloc[[1,3,5,7,9,11],:])

gsdm



#female

gsdf = pd.DataFrame(gsd.iloc[[0,2,4,6,8,10],:])

gsdf
#Suicide population for male

for i in range(1985,2016):

    gsdm.loc[:,i].plot(kind='bar', color = ('skyblue'))

    plt.xticks(range(6),['15-24 years','25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'],

               rotation = 60)

    plt.xlabel('Age group')

    plt.ylabel('Suicide number')

    plt.title('Suicide population of male in '+ str(i))

    

    plt.show()

    plt.clf()

    plt.cla()

    plt.close()




#Suicide population for female

for i in range(1985,2016):

    gsdf.loc[:,i].plot(kind='bar', color = ('lightpink'))

    plt.xticks(range(6),['15-24 years','25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'],rotation = 60)

    plt.xlabel('Age group')

    plt.ylabel('Suicide number')

    plt.title('Suicide population of female in'+ str(i))

    

    plt.show()

    plt.clf()

    plt.cla()

    plt.close()



#Total number by age group

gsd02=pd.DataFrame(data2.groupby(['age','sex'])['suicides_no'].sum().unstack())

gsd02

gsd02_02 = pd.DataFrame(gsd02.T.sum())

gsd02_02
#Pie chart by age group



age=['05-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']

plt.pie(gsd02_02,

               labels = age,

               autopct = '%.1f%%',

               startangle =0,

               radius = 1.5,

               frame = 0,

               center = (4.5,4.5),

               explode=(0.2,0.1,0,0,0,0),

               shadow=True

               )

plt.show()

plt.clf()

plt.cla()

plt.close()

#Total by gender

sexsum = gsd02

sexsum = pd.DataFrame(sexsum.sum())

sexsum = sexsum.reset_index()

sexsum
#Compare the suicide number of male and female

gsd02.iloc[:,1].plot(kind='bar', color='skyblue', width = 1, figsize=(8,5))

gsd02.iloc[:,0].plot(kind='bar', color='lightpink', width = 1, alpha = 0.8,figsize=(8,5))

plt.ylabel('Suicide number')

plt.xlabel('Age group')

plt.xticks(rotation = 60)

plt.title('Suicide number')

plt.legend(['Male','Female'], bbox_to_anchor=(1, 1),title = 'Sex')

plt.show()

plt.clf()

plt.cla()

plt.close()
#Total number by age group

gsd03=pd.DataFrame(data2.groupby(['age','sex'])['suicides_100k_pop'].sum().unstack())

gsd03

gsd03_02 = pd.DataFrame(gsd03.T.sum())

gsd03_02



#Compare the suicide number of male and female

gsd03.iloc[:,1].plot(kind='bar', color='skyblue', width = 1, figsize=(8,5))

gsd03.iloc[:,0].plot(kind='bar', color='lightpink', width = 1, alpha = 0.8,figsize=(8,5))

plt.ylabel('Suicide count 100k pop')

plt.xlabel('Age group')

plt.xticks(rotation = 60)

plt.title('Suicide number')

plt.legend(['Male','Female'], bbox_to_anchor=(1, 1),title = 'Sex')

plt.show()

plt.clf()

plt.cla()

plt.close()
#Plot by year (line)

gsd_year=pd.DataFrame(data2.groupby(['year','country'])['suicides_no'].sum().unstack())

gsd_year = gsd_year.fillna(0)

gsd_year['Suicide number'] = gsd_year.sum(axis=1)



gsd_year.loc[:,'Suicide number'].plot(kind='line',figsize=(10,6),marker='o')

plt.title('Suicide number from 1985 to 2016')

plt.xlabel('year')

plt.ylabel('suicide number')

plt.show()

plt.clf()

plt.cla()

plt.close()




#Group data by country (absolute)

gsdcountry= (pd.DataFrame(data2.groupby(['country','sex'])['suicides_no'].sum().unstack()))/1000000

gsdcountry['Suicide number']=gsdcountry.apply(lambda gsdcountry: gsdcountry['female']+gsdcountry['male'], axis = 1)

gsdcountry = gsdcountry.sort_values(by='Suicide number',ascending=False)

gsdcountry.head(10)



gsdcountry.iloc[0:10,2].plot(kind='barh')

plt.ylabel('Country')

plt.xlabel('Suicide number (million)')

plt.title('Top 10 country of suicide from 1987-2016')



plt.show()

plt.clf()

plt.cla()

plt.close()

#Original suicide number data

gsdcountrynormal = gsdcountry*1000000

gsdcountrynormal = pd.DataFrame(gsdcountrynormal['Suicide number'])

gsdcountrynormal = gsdcountrynormal.reset_index()

gsdcountrynormal
py.sign_in('hbin7552', 'kI7QRK2ZvMwh5vSJP9az')

print(plotly.__version__)
#Draw a choropleth map of world to show the suicide numnber by country

plotly.offline.init_notebook_mode()



#data to graph

my_data = [dict(type='choropleth', 

        autocolorscale=True,

        locations=gsdcountrynormal['country'],

        z=gsdcountrynormal['Suicide number'].astype(float),

        locationmode='country names',

        text=gsdcountrynormal['country'],

        hoverinfo='location+z',

        marker=dict(line=dict(color='rgb(180,180,180)',width=0.5)),

        colorbar=dict(title='Suicide number'))]



#layout

my_layout = dict(title='Suicide number',

                 geo=dict(scope='world',

                          projection=dict(type='mercator'),

                          showcoastlines= False,

                          showframe= False))



fig = dict(data=my_data, layout=my_layout)

py.iplot(fig, validata=False, filename='Suicide number')







#Group data by country (per 100k)

gsdcountryper= pd.DataFrame(data2.groupby(['country','sex'])['suicides_100k_pop'].sum().unstack())

gsdcountryper['Suicide number']=gsdcountryper.apply(lambda gsdcountryper: gsdcountryper['female']

                                                    +gsdcountryper['male'], axis = 1)

gsdcountryper = gsdcountryper.sort_values(by='Suicide number',ascending=False)

gsdcountryper.head(10)



gsdcountryper.iloc[0:10,2].plot(kind='barh')

plt.ylabel('Country')

plt.xlabel('Suicide population (per 100k)')

plt.title('Top 10 country of suicide from 1987-2016')



plt.show()

plt.clf()

plt.cla()

plt.close()





#Top ten suicide country by percentage

gsdcountry10 = pd.DataFrame(gsdcountry.iloc[0:10,2])



top10country = ["Russian Federation","Unites States","Japan","France","Ukraine","Germany","Republic of Korea","Brazil","Poland","United Kingdom"]



plt.pie(gsdcountry10,

               labels = top10country,

               autopct = '%.1f%%',

               startangle =0,

               radius = 1.5,

               frame = 0,

               center = (4.5,4.5),

               explode=(0.2,0,0,0,0,0,0,0,0,0)

               )

plt.show()

plt.clf()

plt.cla()

plt.close()



        #print(data2[data2.country.eq(eachcon)].population)

years = data2['year'].unique()



values2 = []



gdp_df = pd.DataFrame(columns=['country','gdp','gdp_per_capita'])

gdp_df = gdp_df.fillna(0)





for eachcon in countries:

    #print("--------------------------")

    #print(eachcon)

    gdp_for_country = 0

    gdp_for_every_year = []

    gdp_per_capita_every_year = []

    for eachyear in years:

                    #print(eachyear)

                    try:

                        gdp_for_country +=float(data2[data2.country.eq(eachcon) & data2.year.eq(eachyear)].gdp_for_year.iloc[0])

                        gdp_for_every_year.append(float(data2[data2.country.eq(eachcon) & data2.year.eq(eachyear)].gdp_for_year.iloc[0]))

                        #print(float(data2[data2.country.eq(eachcon) & data2.year.eq(eachyear)].gdp_per_capita.iloc[0]))

                        gdp_per_capita_every_year.append(float(data2[data2.country.eq(eachcon) & data2.year.eq(eachyear)].gdp_per_capita.iloc[0]))

                    except Exception as e:

                        #print(e)

                        print("Info : GDP for",eachcon,eachyear,"not found")

    #print(gdp_for_country)

    gdp_df = gdp_df.append({'country' : eachcon , 'gdp' : np.average(gdp_for_every_year),'gdp_per_capita':np.average(gdp_per_capita_every_year)} , ignore_index=True)
gdp_df.head()
gsdcountry
#print(list(data2.groupby(['country']).suicides_100k_pop.sum().to_frame().suicides_100k_pop))

gdp_df['suicides_100k_pop'] = list(data2.groupby(['country']).suicides_100k_pop.sum().to_frame().suicides_100k_pop)

gdp_df['suicides_no'] = list(data2.groupby(['country']).suicides_no.sum().to_frame().suicides_no)



len(gdp_df)
print(gdp_df.head())
#The correlation between perGDP vs suicide number

sns.lmplot(x = "gdp",y = "suicides_100k_pop",

                 data = gdp_df)



g = sns.JointGrid(x = "gdp",y = "suicides_100k_pop",

                 data = gdp_df)

g = g.plot_joint(plt.scatter,

               color="g",s=40,edgecolor="white")

g=g.plot_marginals(sns.distplot, kde=False, color="g")

rsquare = lambda a,b: stats.pearsonr(a,b)[0]**2

g = g.annotate(rsquare, template="{stat}:{val:.2f}",

              stat="$R^2$",loc= "upper right", fontsize=12)
#The correlation between perGDP vs suicide number

sns.lmplot(x = "gdp_per_capita",y = "suicides_100k_pop",

                 data = gdp_df)



g = sns.JointGrid(x = "gdp_per_capita",y = "suicides_100k_pop",

                 data = gdp_df)

g = g.plot_joint(plt.scatter,

               color="g",s=40,edgecolor="white")

g=g.plot_marginals(sns.distplot, kde=False, color="g")

rsquare = lambda a,b: stats.pearsonr(a,b)[0]**2

g = g.annotate(rsquare, template="{stat}:{val:.2f}",

              stat="$R^2$",loc= "upper right", fontsize=12)
#The correlation between perGDP vs suicide number

sns.lmplot(x = "gdp_per_capita",y = "suicides_no",

                 data = gdp_df)



g = sns.JointGrid(x = "gdp_per_capita",y = "suicides_no",

                 data = gdp_df)

g = g.plot_joint(plt.scatter,

               color="g",s=40,edgecolor="white")

g=g.plot_marginals(sns.distplot, kde=False, color="g")

rsquare = lambda a,b: stats.pearsonr(a,b)[0]**2

g = g.annotate(rsquare, template="{stat}:{val:.2f}",

              stat="$R^2$",loc= "upper right", fontsize=12)
#The correlation between perGDP vs suicide number

sns.lmplot(x = "gdp",y = "suicides_no",

                 data = gdp_df)



g = sns.JointGrid(x = "gdp",y = "suicides_no",

                 data = gdp_df)

g = g.plot_joint(plt.scatter,

               color="g",s=40,edgecolor="white")

g=g.plot_marginals(sns.distplot, kde=False, color="g")

rsquare = lambda a,b: stats.pearsonr(a,b)[0]**2

g = g.annotate(rsquare, template="{stat}:{val:.2f}",

              stat="$R^2$",loc= "upper right", fontsize=12)


        #print(data2[data2.country.eq(eachcon)].population)

years = data2['year'].unique()







hdi_df = pd.DataFrame(columns=['country','HDI'])

hdi_df = hdi_df.fillna(0)





for eachcon in countries:

    #print("--------------------------")

    #print(eachcon)

    hdi_for_year = []

    for eachyear in years:

                    #print(eachyear)

                    try:

                        temphd = float(data2[data2.country.eq(eachcon) & data2.year.eq(eachyear)].HDI_for_year.iloc[0])

                        if temphd != 0:

                            hdi_for_year.append(temphd)

                    except Exception as e:

                        #print(e)

                        print("Info : HDI for",eachcon,eachyear,"not found")

    #print(gdp_for_country)

    hdi_df = hdi_df.append({'country' : eachcon , 'HDI' : np.average(hdi_for_year)} , ignore_index=True)
#print(list(data2.groupby(['country']).suicides_100k_pop.sum().to_frame().suicides_100k_pop))

hdi_df['suicides_100k_pop'] = list(data2.groupby(['country']).suicides_100k_pop.sum().to_frame().suicides_100k_pop)
hdi_df.head()




#Draw a choropleth map of world to show the HDI by country

plotly.offline.init_notebook_mode()



colorscale = [[0,"#f7fbff"], 

              [0.1,"#ebf3fb"], 

              [0.2,"#deebf7"], 

              [0.3,"#d2e3f3"], 

              [0.4,"#c6dbef"], 

              [0.45,"#b3d2e9"], 

              [0.5,"#9ecae1"],

              [0.55,"#85bcdb"],

              [0.6,"#6baed6"], 

              [0.65,"#57a0ce"], 

              [0.7,"#4292c6"],

              [0.75,"#3082be"],

              [0.8,"#2171b5"],

              [0.85,"#1361a9"],

              [0.9,"#08519c"],

              [0.95,"#0b4083"],

              [1.0,"#08306b"]]





#data to graph

my_data01 = [dict(type='choropleth', 

        colorscale=colorscale,

        locations=hdi_df['country'],

        z=hdi_df['HDI'],

        locationmode='country names',

        text=gdp_df['country'],

        hoverinfo='location+z',

        marker=dict(line=dict(color='rgb(180,180,180)',width=0.5)),

        colorbar=dict(title='HDI'))]



#layout

my_layout01 = dict(title='HDI',

                 geo=dict(scope='world',

                          projection=dict(type='mercator'),

                          showcoastlines= False,

                          showframe= False))



fig = dict(data=my_data01, layout=my_layout01)

py.iplot(fig, validata=False, filename='HDI')



hdi_df2 = hdi_df.dropna()

len(hdi_df2)
#The correlation between HDI vs suicide number

sns.lmplot(x = "HDI",y = "suicides_100k_pop",

                 data = hdi_df2)



g = sns.JointGrid(x = "HDI",y = "suicides_100k_pop",

                 data = hdi_df2)

g = g.plot_joint(plt.scatter,

               color="g",s=40,edgecolor="white")

g=g.plot_marginals(sns.distplot, kde=False, color="g")

rsquare = lambda a,b: stats.pearsonr(a,b)[0]**2

g = g.annotate(rsquare, template="{stat}:{val:.2f}",

              stat="$R^2$",loc= "upper right", fontsize=12)
gdp_n_HDI = gdp_df



#print(list(data2.groupby(['country']).suicides_100k_pop.sum().to_frame().suicides_100k_pop))

gdp_n_HDI['HDI'] = hdi_df['HDI']

gdp_n_HDI.head()
#Correlation between 4 variables

correlation= gdp_n_HDI.corr()

plt.figure(figsize=(10,8))

ax = sns.heatmap(correlation, vmax=1, square=True, annot=True,fmt='.2f', 

                 cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)

plt.title('Correlation between the features', fontsize=20)

plt.show()

plt.clf()

plt.cla()

plt.close()





#Correlation between 4 variables

pd.plotting.scatter_matrix(gdp_n_HDI, figsize=(8, 8))

plt.show()

plt.clf()

plt.cla()

plt.close()





suic_sum_m = data2['suicides_no'].groupby([data2['country'],data2['sex']]).sum()

suic_sum_m = suic_sum_m.reset_index().sort_index(ascending=False)

most_cont_m = suic_sum_m.head(10)

most_cont_m.head(10)

fig = plt.figure(figsize=(20,5))

plt.title('Count of suicides for 31 years.')

sns.set(font_scale=1.5)

sns.barplot(y='suicides_no',x='country',hue='sex',data=most_cont_m,palette='Set2');

plt.ylabel('Count of suicides')

plt.tight_layout()







suic_sum_yr = pd.DataFrame(data2['suicides_no'].groupby(data2['year']).sum())

suic_sum_yr = suic_sum_yr.reset_index().sort_index(ascending=False)

most_cont_yr = suic_sum_yr

fig = plt.figure(figsize=(30,10))

plt.title('Count of suicides for years.')

sns.set(font_scale=2.5)

sns.barplot(y='suicides_no',x='year',data=most_cont_yr,palette="OrRd");

plt.ylabel('Count of suicides')

plt.xlabel('')

plt.xticks(rotation=45)

plt.tight_layout()







suic_sum_yr = pd.DataFrame(data2['suicides_no'].groupby([data2['generation'],data2['year']]).sum())

suic_sum_yr = suic_sum_yr.reset_index().sort_index(ascending=False)

most_cont_yr = suic_sum_yr

fig = plt.figure(figsize=(30,10))

plt.title('The distribution of suicides by age groups')



sns.set(font_scale=2)

sns.barplot(y='suicides_no',x='year',hue='generation',data=most_cont_yr,palette='deep');

plt.ylabel('Count of suicides')

plt.xticks(rotation=45)

plt.tight_layout()

year = data2.groupby('year').year.unique()



malesuicides = pd.DataFrame(data2[data2.sex == 'male'].groupby('year').suicides_100k_pop.sum())

femalesuicides = pd.DataFrame(data2[data2.sex == 'female'].groupby('year').suicides_100k_pop.sum())

plt.figure(figsize=(16,8))

plt.plot(year.index, malesuicides,label="Male suicides")

plt.plot(year.index, femalesuicides,label="Female suicides")

plt.xlabel('year', fontsize=18)

plt.ylabel('Total number of suicides per 100k in the world', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=18)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()

#print(data2[data2.sex == 'male'].groupby('year').suicides_100k_pop.sum())

agegroups = data2.age.unique()

print(agegroups)



agegone_suicides = pd.DataFrame(data2[data2.age == '5-14 years'].groupby('year').suicides_100k_pop.sum())

agegtwo_suicides = pd.DataFrame(data2[data2.age == '15-24 years'].groupby('year').suicides_100k_pop.sum())

agegthr_suicides = pd.DataFrame(data2[data2.age == '25-34 years'].groupby('year').suicides_100k_pop.sum())

agegfou_suicides = pd.DataFrame(data2[data2.age == '35-54 years'].groupby('year').suicides_100k_pop.sum())

agegfiv_suicides = pd.DataFrame(data2[data2.age == '55-74 years'].groupby('year').suicides_100k_pop.sum())

agegsix_suicides = pd.DataFrame(data2[data2.age == '75+ years'].groupby('year').suicides_100k_pop.sum())



#print(agegone_suicides.suicides_100k_pop.columns)

#print(year.index)

plt.figure(figsize=(16,8))

plt.plot( agegone_suicides,label='5-14 years')

plt.plot(agegtwo_suicides,label='15-24 years')

plt.plot(year.index, agegthr_suicides,label='25-34 years')

plt.plot(year.index, agegfou_suicides,label='35-54 years')

plt.plot(year.index, agegfiv_suicides,label='55-74 years')

plt.plot(year.index, agegsix_suicides,label='75+ years')

plt.xlabel('year', fontsize=18)

plt.ylabel('Total number of suicides per 100k in the world', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=18)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()

#print(data2[data2.sex == 'male'].groupby('year').suicides_100k_pop.sum())
import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
gdp_n_HDI = gdp_n_HDI.dropna()

len(gdp_n_HDI)
regressor = LinearRegression()  

regressor.fit(gdp_n_HDI[['gdp','gdp_per_capita','HDI']], gdp_n_HDI[['suicides_100k_pop']])

import statsmodels.api as sm



X =gdp_n_HDI[['gdp','gdp_per_capita','HDI']] ## X usually means our input variables (or independent variables)



X = sm.add_constant(X)

# Note the difference in argument order

model = sm.OLS(gdp_n_HDI[['suicides_100k_pop']],X).fit()

#predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
import statsmodels.api as sm



# Note the difference in argument order



X =gdp_n_HDI[['HDI']] ## X usually means our input variables (or independent variables)



X = sm.add_constant(X)





model = sm.OLS(gdp_n_HDI[['suicides_100k_pop']],X).fit()

#predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
plt.scatter(gdp_n_HDI.HDI, gdp_n_HDI[['suicides_100k_pop']])

beta = 4846.0429

alpha = 0

st_err =  362.311

#plt.xlabel('Number of friends')

#plt.ylabel('Average minutes per day on site')

plt.plot(gdp_n_HDI.HDI,  alpha+ np.array(gdp_n_HDI.HDI)*beta, '-',color="red",label="Regression line")

y1 = alpha+ np.array(gdp_n_HDI.HDI)*beta - 2*st_err

y2 = alpha+ np.array(gdp_n_HDI.HDI)*beta + 2*st_err

plt.fill_between(gdp_n_HDI.HDI, y1, y2, facecolor=(1,0,0,.4), edgecolor=(0,0,0,.5), label="95% confidence interval")

plt.plot(gdp_n_HDI.HDI,y1 , "--", color="0.5", label="95% Prediction Limits")

plt.plot(gdp_n_HDI.HDI,y2, "--", color="0.5")

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
import matplotlib.pyplot as plt

%matplotlib inline

from scipy.interpolate import interp1d

import statsmodels.api as sm



# introduce some floats in our x-values



X =gdp_n_HDI.HDI

Y= gdp_n_HDI.suicides_100k_pop

# lowess will return our "smoothed" data with a y value for at every x-value

lowess = sm.nonparametric.lowess(Y, X, frac=.3)



# unpack the lowess smoothed points to their values

lowess_x = list(zip(*lowess))[0]

lowess_y = list(zip(*lowess))[1]



# run scipy's interpolation. There is also extrapolation I believe

f = interp1d(lowess_x, lowess_y, bounds_error=False)



xnew = [i/10. for i in range(400)]



# this this generate y values for our xvalues by our interpolator

# it will MISS values outsite of the x window (less than 3, greater than 33)

# There might be a better approach, but you can run a for loop

#and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))

ynew = f(xnew)





plt.plot(X, Y, 'o',label="Actual data")

plt.plot(lowess_x, lowess_y, '-',label="Local regression")

#plt.plot(xnew, ynew, '-')

plt.xlabel('HDI', fontsize=18)

plt.ylabel('suicides per 100k', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=18)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()



gdp_n_HDI.head()
import matplotlib.pyplot as plt

%matplotlib inline

from scipy.interpolate import interp1d

import statsmodels.api as sm



# introduce some floats in our x-values



X =gdp_n_HDI.gdp

Y= gdp_n_HDI.suicides_100k_pop

# lowess will return our "smoothed" data with a y value for at every x-value

lowess = sm.nonparametric.lowess(Y, X, frac=.3)



# unpack the lowess smoothed points to their values

lowess_x = list(zip(*lowess))[0]

lowess_y = list(zip(*lowess))[1]



# run scipy's interpolation. There is also extrapolation I believe

f = interp1d(lowess_x, lowess_y, bounds_error=False)



xnew = [i/10. for i in range(400)]



# this this generate y values for our xvalues by our interpolator

# it will MISS values outsite of the x window (less than 3, greater than 33)

# There might be a better approach, but you can run a for loop

#and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))

ynew = f(xnew)





plt.plot(X, Y, 'o',label="Actual data")

plt.plot(lowess_x, lowess_y, '-',label="Local regression")

#plt.plot(xnew, ynew, '-')

plt.xlabel('gdp', fontsize=18)

plt.ylabel('suicides per 100k', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=18)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()



import matplotlib.pyplot as plt

%matplotlib inline

from scipy.interpolate import interp1d

import statsmodels.api as sm



# introduce some floats in our x-values



X =gdp_n_HDI.gdp_per_capita

Y= gdp_n_HDI.suicides_100k_pop

# lowess will return our "smoothed" data with a y value for at every x-value

lowess = sm.nonparametric.lowess(Y, X, frac=.3)



# unpack the lowess smoothed points to their values

lowess_x = list(zip(*lowess))[0]

lowess_y = list(zip(*lowess))[1]



# run scipy's interpolation. There is also extrapolation I believe

f = interp1d(lowess_x, lowess_y, bounds_error=False)



xnew = [i/10. for i in range(400)]



# this this generate y values for our xvalues by our interpolator

# it will MISS values outsite of the x window (less than 3, greater than 33)

# There might be a better approach, but you can run a for loop

#and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))

ynew = f(xnew)





plt.plot(X, Y, 'o',label="Actual data")

plt.plot(lowess_x, lowess_y, '-',label="Local regression")

#plt.plot(xnew, ynew, '-')

plt.xlabel('gdp_per_capita', fontsize=18)

plt.ylabel('suicides per 100k', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=18)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()



data3 = data2.drop(['country','country-year','generation','suicides_no'], axis = 1) 

data3 = data3.drop(['HDI_for_year'], axis = 1) 

#data3 = data3.drop(['generation'], axis = 1) 
data3.head()
#Converting sex into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)

data3.head()
#Converting age into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

data3.head()
#Converting age into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['year'], prefix='year',drop_first=True)],axis=1).drop(['year'],axis=1)

data3.head()
import statsmodels.api as sm

#data3 = data3.drop(['gdp_for_year'], axis = 1)

X =data3.drop(['suicides_100k_pop'], axis = 1)   	 ## X usually means our input variables (or independent variables)



X = sm.add_constant(X)

# Note the difference in argument order

model = sm.OLS(data3.suicides_100k_pop,X).fit()

#predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
data3 = data2.drop(['country','country-year','generation','suicides_no','year'], axis = 1) 

data3 = data3.drop(['HDI_for_year'], axis = 1) 

#data3 = data3.drop(['generation'], axis = 1) 



#Converting sex into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)

data3.head()



#Converting age into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

data3.head()





import statsmodels.api as sm

#data3 = data3.drop(['gdp_for_year'], axis = 1)

X =data3.drop(['suicides_100k_pop'], axis = 1)   	 ## X usually means our input variables (or independent variables)



X = sm.add_constant(X)

# Note the difference in argument order

model = sm.OLS(data3.suicides_100k_pop,X).fit()

#predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
data3 = data2.drop(['country','country-year','generation','suicides_no'], axis = 1) 

#data3 = data3.drop(['HDI_for_year'], axis = 1) 

#data3 = data3.drop(['generation'], axis = 1) 



#Converting sex into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)

data3.head()



#Converting age into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

data3.head()





data3 = pd.concat([data3,pd.get_dummies(data3['year'], prefix='year',drop_first=True)],axis=1).drop(['year'],axis=1)

data3.head()



data3 = data3.dropna()  #Remove empty HDI



import statsmodels.api as sm

#data3 = data3.drop(['gdp_for_year'], axis = 1)

X =data3.drop(['suicides_100k_pop'], axis = 1)   	 ## X usually means our input variables (or independent variables)



X = sm.add_constant(X)

# Note the difference in argument order

model = sm.OLS(data3.suicides_100k_pop,X).fit()

#predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
data3 = data2.drop(['country','country-year','generation','suicides_no','year'], axis = 1) 

#data3 = data3.drop(['HDI_for_year'], axis = 1) 

#data3 = data3.drop(['generation'], axis = 1) 



#Converting sex into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)

data3.head()



#Converting age into onehot encoding



data3 = pd.concat([data3,pd.get_dummies(data3['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

data3.head()









data3 = data3.dropna()  #Remove empty HDI



import statsmodels.api as sm

#data3 = data3.drop(['gdp_for_year'], axis = 1)

X =data3.drop(['suicides_100k_pop'], axis = 1)   	 ## X usually means our input variables (or independent variables)



X = sm.add_constant(X)

# Note the difference in argument order

model = sm.OLS(data3.suicides_100k_pop,X).fit()

#predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
plt.hist(data2.suicides_100k_pop,bins=20)

plt.title("Distribution of suicides_100k_pop", fontsize=14)

plt.xlabel('suicides_100k_pop', fontsize=16)

plt.ylabel('Frequency', fontsize=16)

#plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=16)

plt.rc('ytick',labelsize=16)

plt.show()

plt.clf()

plt.cla()

plt.close()
# > 80 Z3

#  40 - 79 Z2

#  20 - 39 Z1

# 0 - 19 Z0

data4 = data2

data4['suicides_class'] = pd.cut(x=data2['suicides_100k_pop'], bins=[-0.1,19, 39, 79, 500], labels=['Z0', 'Z1', 'Z2','Z3'])



plt.hist(data4['suicides_class'])

plt.title("Distribution of suicides_class", fontsize=14)

plt.xlabel('Suicide class', fontsize=16)

plt.ylabel('Frequency', fontsize=16)

#plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=16)

plt.rc('ytick',labelsize=16)

plt.show()

plt.clf()

plt.cla()

plt.close()



from sklearn.utils import resample



df_majority = data4[data4.suicides_class=="Z0"]

df_minority1 = data4[data4.suicides_class=="Z1"]

df_minority2 = data4[data4.suicides_class=="Z2"]

df_minority3 = data4[data4.suicides_class=="Z3"]

 

# Upsample minority class

df_minority1_upsampled = resample(df_minority1, 

                                 replace=True,     # sample with replacement

                                 n_samples=len(df_majority),    # to match majority class

                                 random_state=123) # reproducible results

df_minority2_upsampled = resample(df_minority2, 

                                 replace=True,     # sample with replacement

                                 n_samples=len(df_majority),    # to match majority class

                                 random_state=123) # reproducible results

 

df_minority3_upsampled = resample(df_minority3, 

                                 replace=True,     # sample with replacement

                                 n_samples=len(df_majority),    # to match majority class

                                 random_state=123) # reproducible results

 

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled, df_minority3_upsampled])
plt.hist(df_upsampled['suicides_class'])

plt.title("Distribution of suicides_class - balanced", fontsize=14)

plt.xlabel('Suicide class', fontsize=16)

plt.ylabel('Frequency', fontsize=16)

#plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=16)

plt.rc('ytick',labelsize=16)

plt.show()

plt.clf()

plt.cla()

plt.close()
data4 = df_upsampled

data5 = data4.drop(['country','country-year','generation','suicides_no','suicides_100k_pop'], axis = 1) 

#data5 = df_upsampled.drop(['country','country-year','generation','suicides_no','suicides_100k_pop'], axis = 1) 
data5.head()



data5 = pd.concat([data5,pd.get_dummies(data5['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)





#Converting age into onehot encoding



data5 = pd.concat([data5,pd.get_dummies(data5['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)







data5 = pd.concat([data5,pd.get_dummies(data5['year'], prefix='year',drop_first=True)],axis=1).drop(['year'],axis=1)

data5.head()



#data3 = data3.dropna()  #Remove empty HDI
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data5.drop(['suicides_class'],axis=1), data5.suicides_class, test_size=0.25,random_state=5)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score



complexities = []

train_errors = []

test_errors = []

for n_estimators in [1,2,4,8,16,32,64]:

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=8)

    #sample_size = len(y_train)

    

    clf.fit(X_train, y_train)

    #train_error = 1-clf.score(X_train,y_train)#error(clf,X_train,y_train)

    

    

    

    #test_error =  1-clf.score(X_test,y_test)

    #train_error = sum(train_errors)/len(train_errors)

    #test_error = sum(test_errors)/len(test_errors)

    complexities.append(n_estimators)

    train_errors.append(f1_score(y_train, clf.predict(X_train), average='weighted'))

    test_errors.append(f1_score(y_test, clf.predict(X_test), average='weighted'))

    #print(clf.predict(X_test))

plt.plot(complexities, train_errors, c='b', label='Training f1-score')

plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

plt.ylim(0,1)

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('Random forest')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
data5 = data4.drop(['country','country-year','generation','suicides_no','suicides_100k_pop','year'], axis = 1) 



data5.head()



data5 = pd.concat([data5,pd.get_dummies(data5['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)





#Converting age into onehot encoding



data5 = pd.concat([data5,pd.get_dummies(data5['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

print(data5.head())



#data3 = data3.dropna()  #Remove empty HDI



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data5.drop(['suicides_class'],axis=1), data5.suicides_class, test_size=0.25,random_state=5)











from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score



complexities = []

train_errors = []

test_errors = []

for n_estimators in [1,2,4,8,16,32,64]:

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=8)

    #sample_size = len(y_train)

    

    clf.fit(X_train, y_train)

    #train_error = 1-clf.score(X_train,y_train)#error(clf,X_train,y_train)

    

    

    

    #test_error =  1-clf.score(X_test,y_test)

    #train_error = sum(train_errors)/len(train_errors)

    #test_error = sum(test_errors)/len(test_errors)

    complexities.append(n_estimators)

    train_errors.append(f1_score(y_train, clf.predict(X_train), average='weighted'))

    test_errors.append(f1_score(y_test, clf.predict(X_test), average='weighted'))

    #print(clf.predict(X_test))

plt.plot(complexities, train_errors, c='b', label='Training f1-score')

plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

plt.ylim(0,1)

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('Random forest')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
data5 = data4.drop(['country','country-year','generation','suicides_no','suicides_100k_pop','year','gdp_for_year'], axis = 1) 



data5.head()



data5 = pd.concat([data5,pd.get_dummies(data5['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)





#Converting age into onehot encoding



data5 = pd.concat([data5,pd.get_dummies(data5['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

data5.head()



#data3 = data3.dropna()  #Remove empty HDI



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data5.drop(['suicides_class'],axis=1), data5.suicides_class, test_size=0.25,random_state=5)











from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score



complexities = []

train_errors = []

test_errors = []

for n_estimators in [1,2,4,8,16,32,64]:

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=8)

    #sample_size = len(y_train)

    

    clf.fit(X_train, y_train)

    #train_error = 1-clf.score(X_train,y_train)#error(clf,X_train,y_train)

    

    

    

    #test_error =  1-clf.score(X_test,y_test)

    #train_error = sum(train_errors)/len(train_errors)

    #test_error = sum(test_errors)/len(test_errors)

    complexities.append(n_estimators)

    train_errors.append(f1_score(y_train, clf.predict(X_train), average='weighted'))

    test_errors.append(f1_score(y_test, clf.predict(X_test), average='weighted'))

    #print(clf.predict(X_test))

plt.plot(complexities, train_errors, c='b', label='Training f1-score')

plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

plt.ylim(0,1)

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('Random forest')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
data5 = data4.drop(['country','country-year','generation','suicides_no','suicides_100k_pop','year','gdp_for_year'], axis = 1) 



data5.head()



data5 = pd.concat([data5,pd.get_dummies(data5['sex'], prefix='sex',drop_first=True)],axis=1).drop(['sex'],axis=1)





#Converting age into onehot encoding



data5 = pd.concat([data5,pd.get_dummies(data5['age'], prefix='age',drop_first=True)],axis=1).drop(['age'],axis=1)

data5.head()



data5 = data5[data5.HDI_for_year != 0] #Remove empty HDI



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data5.drop(['suicides_class'],axis=1), data5.suicides_class, test_size=0.25,random_state=5)











from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score



complexities = []

train_errors = []

test_errors = []

for n_estimators in [1,2,4,8,16,32,64]:

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=8)

    clf.fit(X_train, y_train)

    complexities.append(n_estimators)

    train_errors.append(f1_score(y_train, clf.predict(X_train), average='weighted'))

    test_errors.append(f1_score(y_test, clf.predict(X_test), average='weighted'))

plt.plot(complexities, train_errors, c='b', label='Training f1-score')

plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

plt.ylim(0,1)

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('Random forest')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
#tree = DecisionTreeClassifier(max_depth=2,criterion,splitter)

#_ = tree.fit(X_train, Y_train)



# Evaluate

#print('Classification report ({}):\n'.format(key))

#print(classification_report(Y_test, tree.predict(X_test)))



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , RandomForestClassifier())])

param_grid = [

    {'classifier' : [RandomForestClassifier()],

     'classifier__criterion' : ['entropy', 'gini'],

    'classifier__max_depth' : [2,3,4,5,6,8,16,32,None],

     'classifier__n_estimators':[1,2,4,8,16,32,64],

     'classifier__max_features' : [6, 11, 16, 21, 26, 31]

    }

]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=False, n_jobs=-1,scoring='f1_macro')



# Fit on data



best_clf = clf.fit(X_train, y_train)

print("Best paramters are:")

print(best_clf.best_params_)

print("Best f1 score (training) :",best_clf.best_score_)

print("Best f1 score (validation) :",f1_score(y_test, best_clf.predict(X_test), average='macro'))





%matplotlib inline

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt



from sklearn.metrics import classification_report

import pandas as pd

key=', '.join(['{}={}'.format(i,name) for i,name in enumerate(y_test)])

#print('Classification report ({}):\n'.format(key))

#print(confusion_matrix(best_clf.predict(X_test) , y_test ))



#y_actu = pd.Series(y_test, name='Actual')

#y_pred_s1 = pd.Series(best_clf.predict(X_test), name='Predicted')

#confusion_matrix = pd.crosstab(y_pred_s1, y_actu)

#print(confusion_matrix)



#print('Confusion matrix ({}):\n'.format(key))

_ = plt.matshow(confusion_matrix(best_clf.predict(X_test) , y_test ), cmap=plt.cm.binary, interpolation='nearest')

_ = plt.colorbar()

_ = plt.ylabel('true label')

_ = plt.xlabel('predicted label')

plt.show()

#print(confusion_matrix(best_clf.predict(X_test) , y_test ))

print(classification_report(y_test, best_clf.predict(X_test)))
from sklearn.tree import DecisionTreeClassifier



complexities = []

train_errors = []

test_errors = []

for max_depth in [2,4,8,16,32,None]:

    clf = DecisionTreeClassifier(max_depth=max_depth)

    clf.fit(X_train, y_train)

    complexities.append(max_depth)

    train_errors.append(f1_score(y_train, clf.predict(X_train), average='macro'))

    test_errors.append(f1_score(y_test, clf.predict(X_test), average='macro'))

plt.plot(complexities, train_errors, c='b', label='Training f1-score')

plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

#plt.ylim(0,1)

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('DecisionTreeClassifier')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
#tree = DecisionTreeClassifier(max_depth=2,criterion,splitter)

#_ = tree.fit(X_train, Y_train)



# Evaluate

#print('Classification report ({}):\n'.format(key))

#print(classification_report(Y_test, tree.predict(X_test)))



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , DecisionTreeClassifier())])

param_grid = [

    {'classifier' : [DecisionTreeClassifier()],

     'classifier__criterion' : ['entropy', 'gini'],

    'classifier__max_depth' : [2,3,4,5,6,8,16,32,None], #2,4,8,16,32,None

     'classifier__splitter' : ['best', 'random']

    }

]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=False, n_jobs=-1,scoring='f1_macro')



# Fit on data



best_clf = clf.fit(X_train, y_train)

print("Best paramters are:")

print(best_clf.best_params_)

print("Best f1 score (training) :",best_clf.best_score_)

print("Best f1 score (validation) :",f1_score(y_test, best_clf.predict(X_test), average='macro'))
from sklearn.neighbors import KNeighborsClassifier



complexities = []

train_errors = []

test_errors = []

for neighbour in range(3,10):

    clf = KNeighborsClassifier(n_neighbors=neighbour)#DecisionTreeClassifier(max_depth=max_depth)

    clf.fit(X_train, y_train)

    complexities.append(neighbour)

    train_errors.append(f1_score(y_train, clf.predict(X_train), average='macro'))

    test_errors.append(f1_score(y_test, clf.predict(X_test), average='macro'))

plt.plot(complexities, train_errors, c='b', label='Training f1-score')

plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

#plt.ylim(0,1)

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('KNeighborsClassifier')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
k_range = list(range(1, 31))

print(k_range)
#tree = DecisionTreeClassifier(max_depth=2,criterion,splitter)

#_ = tree.fit(X_train, Y_train)



# Evaluate

#print('Classification report ({}):\n'.format(key))

#print(classification_report(Y_test, tree.predict(X_test)))



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , KNeighborsClassifier())])

param_grid = [

    {'classifier' : [KNeighborsClassifier()],

     'classifier__n_neighbors' :k_range,

     'classifier__weights':['uniform','distance'],

'classifier__metric':['euclidean','manhattan'],

    }

]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=False, n_jobs=-1,scoring='f1_macro')



# Fit on data



best_clf = clf.fit(X_train, y_train)

print("Best paramters are:")

print(best_clf.best_params_)

print("Best f1 score (training) :",best_clf.best_score_)

print("Best f1 score (validation) :",f1_score(y_test, best_clf.predict(X_test), average='macro'))
#from sklearn import svm

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)#DecisionTreeClassifier(max_depth=max_depth)

clf.fit(X_train, y_train)

    #complexities.append(neighbour)

train_err = f1_score(y_train, clf.predict(X_train), average='macro')

test_err = f1_score(y_test, clf.predict(X_test), average='macro')

#plt.plot(complexities, train_errors, c='b', label='Training f1-score')

#plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

#plt.ylim(0,1)

print(train_err)

print(test_err)

plt.bar([1,2],[train_err,test_err])

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('Linear SVM')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
#tree = DecisionTreeClassifier(max_depth=2,criterion,splitter)

#_ = tree.fit(X_train, Y_train)



# Evaluate

#print('Classification report ({}):\n'.format(key))

#print(classification_report(Y_test, tree.predict(X_test)))



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , LinearSVC())])

param_grid = [

    {'classifier' : [LinearSVC()],

     'classifier__C' :np.arange(0.01,100,10)

    }

]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=False, n_jobs=-1,scoring='f1_macro')



# Fit on data



best_clf = clf.fit(X_train, y_train)

print("Best paramters are:")

print(best_clf.best_params_)

print("Best f1 score (training) :",best_clf.best_score_)

print("Best f1 score (validation) :",f1_score(y_test, best_clf.predict(X_test), average='macro'))
from sklearn.linear_model import LogisticRegression

clf =LogisticRegression()#DecisionTreeClassifier(max_depth=max_depth)

clf.fit(X_train, y_train)

    #complexities.append(neighbour)

train_err = f1_score(y_train, clf.predict(X_train), average='macro')

test_err = f1_score(y_test, clf.predict(X_test), average='macro')

#plt.plot(complexities, train_errors, c='b', label='Training f1-score')

#plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

#plt.ylim(0,1)

print(train_err)

print(test_err)

plt.bar([1,2],[train_err,test_err])

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('LogisticRegression')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
#tree = DecisionTreeClassifier(max_depth=2,criterion,splitter)

#_ = tree.fit(X_train, Y_train)



# Evaluate

#print('Classification report ({}):\n'.format(key))

#print(classification_report(Y_test, tree.predict(X_test)))



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , LogisticRegression())])

param_grid = [

    {'classifier' : [LogisticRegression()],

     'classifier__penalty' : ['l1', 'l2'],

    'classifier__C' : np.logspace(-4, 4, 20),

    'classifier__solver' : ['liblinear']}

]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=False, n_jobs=-1,scoring='f1_macro')



# Fit on data



best_clf = clf.fit(X_train, y_train)

print("Best paramters are:")

print(best_clf.best_params_)

print("Best f1 score (training) :",best_clf.best_score_)

print("Best f1 score (validation) :",f1_score(y_test, best_clf.predict(X_test), average='macro'))
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



clf = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)

    #complexities.append(neighbour)

train_err = f1_score(y_train, clf.predict(X_train), average='macro')

test_err = f1_score(y_test, clf.predict(X_test), average='macro')

#plt.plot(complexities, train_errors, c='b', label='Training f1-score')

#plt.plot(complexities, test_errors, c='r', label='Generalisation f1-score')

#plt.ylim(0,1)

print(train_err)

print(test_err)

plt.bar([1,2],[train_err,test_err])

plt.ylabel('f1-score')

plt.xlabel('Model complexity')

plt.title('LinearDiscriminantAnalysis')

plt.legend()

plt.show()

plt.clf()

plt.cla()

plt.close()
#tree = DecisionTreeClassifier(max_depth=2,criterion,splitter)

#_ = tree.fit(X_train, Y_train)



# Evaluate

#print('Classification report ({}):\n'.format(key))

#print(classification_report(Y_test, tree.predict(X_test)))



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , LinearDiscriminantAnalysis())])

param_grid = [

    {'classifier' : [LinearDiscriminantAnalysis()],

     'classifier__solver' : ['svd', 'lsqr','eigen'],

    'classifier__shrinkage' : [None,'auto']

    }

]







clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=False, n_jobs=-1,scoring='f1_macro')



# Fit on data



best_clf = clf.fit(X_train, y_train)

print("Best paramters are:")

print(best_clf.best_params_)

print("Best f1 score (training) :",best_clf.best_score_)

print("Best f1 score (validation) :",f1_score(y_test, best_clf.predict(X_test), average='macro'))
from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



scoring_func = "f1_macro"  #'f1_weighted'



#train_test_split(data5.drop(['suicides_class'],axis=1), data5.suicides_class, test_size=0.25,random_state=5)

clf = LinearDiscriminantAnalysis(shrinkage=None, solver='svd')

scores_lda = cross_val_score(clf, data5.drop(['suicides_class'],axis=1), data5.suicides_class, cv=10, scoring=scoring_func)



from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(criterion= 'gini', max_depth=None, max_features=6, n_estimators=32)

scores_randfor = cross_val_score(clf2, data5.drop(['suicides_class'],axis=1), data5.suicides_class, cv=10, scoring=scoring_func)



#'classifier__criterion': 'entropy', 'classifier__max_depth': 32, 'classifier__max_features': 6, 'classifier__n_estimators': 32}



from sklearn.tree import DecisionTreeClassifier

clf3 = DecisionTreeClassifier(criterion= 'entropy', max_depth=None,splitter='best')

scores_tree = cross_val_score(clf3, data5.drop(['suicides_class'],axis=1), data5.suicides_class, cv=10, scoring=scoring_func)





from sklearn.neighbors import KNeighborsClassifier

clf4 = KNeighborsClassifier(metric='manhattan',n_neighbors=1, weights='uniform')

scores_knn = cross_val_score(clf4, data5.drop(['suicides_class'],axis=1), data5.suicides_class, cv=10, scoring=scoring_func)



from sklearn.svm import LinearSVC

clf5 = LinearSVC(C=0.01)

scores_linsvc = cross_val_score(clf5, data5.drop(['suicides_class'],axis=1), data5.suicides_class, cv=10, scoring=scoring_func)



from sklearn.linear_model import LogisticRegression

clf6 = LogisticRegression(C=206.913808111479,penalty='l1',solver='liblinear')

scores_log = cross_val_score(clf6, data5.drop(['suicides_class'],axis=1), data5.suicides_class, cv=10, scoring=scoring_func)



#C': 206.913808111479, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'}



x_val = list(range(1,11))

from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

axes = plt.gca()

#axes.set_xlim([xmin,xmax])

axes.set_ylim([0,1])

#plt.plot(x_val, scores_lda,label="LDA")

plt.plot(x_val, scores_randfor,label="Random Forest")

plt.plot(x_val, scores_tree,label="Decision Tree")

#plt.plot(x_val, scores_knn,label="kNN")

#plt.plot(x_val, scores_linsvc,label="Linear SVC")

#plt.plot(x_val, scores_log,label="Logistic Regression")



plt.xlabel('Fold', fontsize=18)

plt.ylabel('F1-score', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=18)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()
'{:.10f}'.format(stats.ttest_rel(scores_randfor, scores_tree).pvalue*0.5)
'{:.18f}'.format(stats.ttest_rel(scores_randfor, scores_lda).pvalue*0.5)
'{:.10f}'.format(stats.ttest_rel(scores_randfor, scores_knn).pvalue*0.5)
'{:.18f}'.format(stats.ttest_rel(scores_randfor, scores_linsvc).pvalue*0.5)
'{:.18f}'.format(stats.ttest_rel(scores_randfor, scores_log).pvalue*0.5)
trainf1= [0.976,0.97,0.92,0.16,0.57,0.57]

validationf1 = [0.98,0.976,0.93,0.17,0.57,0.57]





x = np.array([0,1,2,3,4,5])

my_xticks = ['Random Forest','Decision Tree','KNeighbors','Linear SVC','LogisticRegression','LDA']

plt.xticks(x, my_xticks,rotation=45)



plt.plot(x,trainf1,label="Training f1-score")

plt.plot(x,validationf1,label="Validation f1-score")

#plt.plot(x_val, scores_knn,label="kNN")

#plt.plot(x_val, scores_linsvc,label="Linear SVC")

#plt.plot(x_val, scores_log,label="Logistic Regression")



#plt.xlabel('Fold', fontsize=18)

plt.ylabel('F1-score', fontsize=18)

plt.legend(fontsize='medium')

plt.rc('xtick',labelsize=10)

plt.rc('ytick',labelsize=18)

plt.show()

plt.clf()

plt.cla()

plt.close()