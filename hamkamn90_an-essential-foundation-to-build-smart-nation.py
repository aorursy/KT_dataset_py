import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



raw_data1 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv' , header=0, skiprows=[1])

raw_data2 = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv', header=0, skiprows=[1])

country_list = pd.read_excel('../input/income-countries/CLASS.xls', header=0, skiprows=[0,1,2,3,4])

pd.set_option('display.max_columns', None)



country_list = country_list.drop(columns=['x','x.1','x.4','x.7','x.8'])

country_list = country_list.rename(columns={"x.2": "Country", "x.3": "Abbre", "x.5":"Region", "x.6":"Income Group"})



country_data = country_list

country_data = country_data.replace({'Country': {'Russian Federation': 'Russia', 'Timor-Leste': 'East Timor', 

                                        'United States': 'United States of America',

                                       'Korea, Rep.': 'South Korea', "Korea, Dem. People's Rep.": 'Republic of Korea', 

                                        'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',

                                       'Hong Kong SAR, China': 'Hong Kong (S.A.R.)', 'Taiwan, China': 'Taiwan', 

                                        'Egypt, Arab Rep.': 'Egypt', 'Vietnam': 'Viet Nam', 

                                        'Iran, Islamic Rep.': 'Iran, Islamic Republic of...'}})                          

data = pd.DataFrame(raw_data1.merge(country_data, left_on='Q3', right_on='Country'))

data = data.replace({'Country': {'Republic of Korea': 'North Korea'}})

data = data.replace({'Q3': {'Republic of Korea': 'North Korea'}})
c=data[1:]

countries=c[["Q3","Income Group", "Q1"]].groupby(["Income Group","Q3"]).count().stack().reset_index()

countries=pd.DataFrame(countries)

countries=countries.rename({'Q3':'country'}, axis='columns')



def colors (row):

    if row['Income Group'] == 'Low income' :

      return 1

    if row['Income Group'] == 'Lower middle income' :

      return 2

    if row['Income Group'] == 'Upper middle income' :

      return 3

    if row['Income Group'] == 'High income' :

      return 4



countries['Color'] = countries.apply (lambda row: colors (row) , axis=1)



import plotly.express as px

fig = px.choropleth(countries,

                    locations="country",

                    color="Color",

                    hover_name= 'Income Group',

                    locationmode = 'country names',

                    color_continuous_scale='haline'

                   )

                    

fig.show()
table1 = data['Income Group'].value_counts()

table1 = pd.DataFrame(table1)

table1
analysis_data = data[data['Income Group'] !='Low income']



fig, ax = plt.subplots(figsize=(15,7))

labels = ['High income', 'Lower middle income', 'Upper middle income']

sizes = [8762, 6505, 3323]

labels_gender = ['Female','Male','Prefer not to say/self-describe',

                 'Female','Male','Prefer not to say/self-describe',

                 'Female','Male','Prefer not to say/self-describe']

labels_gender2 = ['Female','Male','Prefer not to say/self-describe']

sizes_gender = [1453, 7120, 189, 1099, 5334, 72, 490, 2780, 53]



cmap1 = plt.get_cmap("tab20c")

cmap2 = plt.get_cmap("Set3")

outer_colors = cmap1(np.arange(3))

inner_colors = cmap2(np.arange(3)*3)

inner_colors = np.concatenate((inner_colors, inner_colors, inner_colors))



explode = (0.1,0.15,0.2) 

explode_gender = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)



plt.pie(sizes, explode=explode, colors=outer_colors, labels=labels, frame=True, autopct='%1.1f%%', radius=3, 

        pctdistance = 0.8, shadow=True, startangle=90)



patches, texts = plt.pie(sizes_gender, explode=explode_gender, radius=2, colors=inner_colors, labeldistance = 0.7,

                         labels=sizes_gender, startangle=95)

plt.legend(patches, labels_gender2, loc="best")



centre_circle = plt.Circle((0,0),1,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

 

plt.axis('equal')

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(21,8))

ax = sns.countplot(x="Q1", hue = "Income Group", data=analysis_data, 

                   order = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+'])
cmap = plt.get_cmap("tab20")

bar_colors = cmap(np.arange(12))

g = sns.catplot(x="Q1", hue="Q5", col="Income Group", col_wrap=1,

                data=analysis_data, kind="count", sharex=False, sharey=False,

                col_order= ['High income', 'Upper middle income', 'Lower middle income'],

                height=4, aspect=4, legend_out = False,

                order = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+'],

               palette=bar_colors)
cmap = plt.get_cmap("tab20")

bar_colors = cmap(np.arange(12))

g = sns.catplot(x="Q15", hue="Q5", col="Income Group", col_wrap=1,

                data=analysis_data, kind="count", sharex=False, sharey=False,

                col_order= ['High income', 'Upper middle income', 'Lower middle income'],

                height=4, aspect=4, legend_out = False,

                order = ['20+ years','10-20 years','5-10 years','3-5 years','1-2 years','< 1 years',

                         'I have never written code'], palette=bar_colors)
cmap = plt.get_cmap("tab20")

bar_colors = cmap(np.arange(12))

ax = sns.catplot(x="Q4", hue="Q5", col="Income Group", col_wrap=1,

                data=analysis_data, kind="count", sharex=True, sharey=False,

                col_order= ['High income', 'Upper middle income', 'Lower middle income'],

                height=4, aspect=4, legend_out = False,

                order = ["Doctoral degree","Master’s degree","Professional degree","Bachelor’s degree",

                         "Some college/university study without earning a bachelor’s degree",

                         "No formal education past high school","I prefer not to answer"], palette=bar_colors)

ax.set_xticklabels(rotation=30, horizontalalignment='right')
ax = sns.catplot(x="Q5", hue="Q8", col="Income Group", col_wrap=1,

                data=analysis_data, kind="count", sharex=True, sharey=False,

                col_order= ['High income', 'Upper middle income', 'Lower middle income'],

                height=4, aspect=4, legend_out = False)

ax.set_xticklabels(rotation=30, horizontalalignment='right')
data_compensation = analysis_data[analysis_data['Q5'] !='Student']

fig, ax = plt.subplots(figsize=(21,8)) 

ax = sns.countplot(x="Q10", hue = "Income Group", data=data_compensation,

                   order = ['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999',

                            '10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999',

                           '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999','100,000-124,999',

                           '125,000-149,999','150,000-199,999','200,000-249,999','250,000-299,999','300,000-500,000',

                           '> $500,000'])

ax.set_xticklabels(ax.get_xticklabels(), rotation = 25, horizontalalignment='right') 
