

# import 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

import plotly as pl

#import plotly.plotly as pl

import plotly.graph_objs as gobj

import pandas as pd

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

pd.options.display.max_colwidth =  200

pd.set_option('display.max_columns', None)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings

import warnings

warnings.filterwarnings('ignore')
# Importing the 2017,2018 and 2019 survey dataset



#Importing the 2019 Dataset

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df_2019.columns = df_2019.iloc[0]

df_2019=df_2019.drop([0])



df_2019 = df_2019.rename(columns={'Duration (in seconds)': 'Duration',

        'What is your age (# years)?': 'Age', 

        'What is your gender? - Selected Choice': 'Gender',

        'In which country do you currently reside?':'Country',

        'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Education',

        'What is the size of the company where you are employed?':'CompanySize',

        'What is your current yearly compensation (approximate $USD)?':'Salary',

        'Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?':'MoneyDS',

                            })

df_2019 = df_2019[['Duration','Age','Gender','Country','Education','Salary','CompanySize','MoneyDS']] #'CompanySize''MoneyDS'



# Replacing the ambigious countries name with Standard names

df_2019['Country'].replace({'United States of America':'United States',

                            'Viet Nam':'Vietnam',

                             "People 's Republic of China":'China',

                             "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                             "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)

#Importing the 2018 Dataset

df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df_2018.columns = df_2018.iloc[0]

df_2018=df_2018.drop([0])



df_2018 = df_2018.rename(columns={'Duration (in seconds)': 'Duration',

        'What is your age (# years)?': 'Age', 

        'What is your gender? - Selected Choice': 'Gender',

        'In which country do you currently reside?':'Country',

        'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Education',

         #'What is the size of the company where you are employed?':'CompanySize',

        'What is your current yearly compensation (approximate $USD)?':'Salary',

         #'Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?':'MoneyDS',

                            })

df_2018 = df_2018[['Duration','Age','Gender','Country','Education','Salary']]

# #Importing the 2017 Dataset

df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
gender_2019 = df_2019['Gender'].value_counts(sort=True)

gender_2019 = gender_2019[:3]

sns.set(style="whitegrid")

ax = sns.barplot(x=gender_2019.index, y=gender_2019.values)

ax.axhline(0, color="k", clip_on=False) 

ax.set_xlabel("Gender")

ax.set_ylabel("Count")

ax.set_title("Females to Males")
df_N = pd.DataFrame(data = [len(df_2017),len(df_2018),len(df_2019)],

                          columns = ['Numresponses'])

df_F = pd.DataFrame(data = [(df_2017['GenderSelect'] == 'Female').sum(), (df_2018['Gender'] == 'Female').sum(),

                             (df_2019['Gender'] == 'Female').sum()], columns = ['Females'])

df_M = pd.DataFrame(data = [(df_2017['GenderSelect'] == 'Male').sum(), (df_2018['Gender'] == 'Male').sum(),

                             (df_2019['Gender'] == 'Male').sum()], columns = ['Males'])

df_A = pd.concat([df_N, df_F,df_M] , axis=1)

#df_A['Year'] = ['2017','2018','2019']

                     

df_A.index = ['2017','2018','2019']

df_A.plot()
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), sharex=True)

# x=df_A.index

# values =df_A["Females"]/df_A["Males"]

# ax1.stem(x, values)

# values1 =df_A["Females"]/df_A["Numresponses"]

# ax2.stem(x, values1)

df_A.plot(kind="bar")
df=df_2019.groupby('Country')['Gender'].apply(lambda x: (x=='Female').count()).reset_index(name='ctFemales')

df = df.sort_values(by='ctFemales', ascending=False)

df = df.iloc[:20,:]

x = df['Country']

values = df['ctFemales']

my_range=range(1,len(df)+1)

# # Vertical version.

plt.figure(figsize=(12,6))

plt.hlines(y=my_range, xmin=0, xmax=df['ctFemales'], color='red')

plt.plot(values, my_range, "D")

plt.yticks(my_range, x)

plt.show()
df1 = df_2019.copy()

gkk = df1.groupby(['Country','Gender']).size()

gkk = pd.DataFrame(gkk).reset_index()

gkk = gkk.rename(columns={0:'count'})

gkk = gkk.pivot(index='Country', columns='Gender', values='count')

gkk.columns.name = None

gkk['Female'] = gkk['Female'].astype(int)

gkk['Male'] = gkk['Male'].astype(int)

gkk =gkk.sort_values(by = 'Female', ascending = False)

gkk = gkk.iloc[:21,:]

value1 = gkk.Female

value2 = gkk.Male

gF = pd.DataFrame({'group':gkk.index.tolist(), 'value1':value1 , 'value2':value2 })

my_range=range(1,len(gF.index)+1)

import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(12,8))

plt.hlines(y=my_range, xmin=gF['value1'], xmax=gF['value2'], color='blue', alpha=1)

plt.scatter(gF['value1'], my_range, color='red', alpha=1, label='Females')

plt.scatter(gF['value2'], my_range, color='green', alpha=1 , label='Males')

plt.legend()

 

# Add title and axis names

plt.yticks(my_range, gF['group'])

plt.title("Top 20 countries of Females compared to Males", loc='center')

plt.xlabel('Num of Females & Males')

plt.ylabel('Countries')
df2 = df_2019.copy()

gkk1 = df2.groupby(['Country','Gender']).size()

gkk1 = pd.DataFrame(gkk1).reset_index()

gkk1 = gkk1.rename(columns={0:'count'})

gkk1 = gkk1.pivot(index='Country', columns='Gender', values='count')

gkk1.columns.name = None

gkk1['Female'] = gkk1['Female'].astype(int)

gkk1['Male'] = gkk1['Male'].astype(int)

gkk1 =gkk1.sort_values(by = 'Female', ascending = False)

gkn = gkk1.reset_index()



data = dict(type = 'choropleth',

            locations = gkn['Country'],

            locationmode = 'country names',

            colorscale= 'Reds',

            #text= ['IND','NEP','CHI','PAK','BAN','BHU', 'MYN','SLK'],

            z= gkn['Female'],#[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],

            colorbar = {'title':'Country Colours', 'len':200,'lenmode':'pixels' })

#initializing the layout variable

layout = dict(title = 'The Nationality of Female Respondents in 2019',geo = {'scope':'world'})

# Initializing the Figure object by passing data and layout as arguments.

col_map = gobj.Figure(data = [data],layout = layout)



#plotting the map

iplot(col_map)
f_2019 = df_2019[df_2019['Gender']=='Female'].copy()

f_2018 = df_2018[df_2018['Gender']=='Female'].copy()

f_2017 = df_2017[df_2017['GenderSelect']=='Female'].copy()

f_2017['Age'] = pd.cut(x=f_2017['Age'], bins=[18,21,25,29,34,39,44,49,54,59,69,79], 

                                                        labels=['18-21',

                                                                '22-24',

                                                                '25-29',

                                                                '30-34',

                                                                '35-39',

                                                                '40-44',

                                                                '45-49',

                                                                '50-54',

                                                                '55-59',

                                                                '60-69',

                                                                '70+'])

x = f_2017['Age'].value_counts()

y = f_2018['Age'].value_counts()

z = f_2019['Age'].value_counts()

w = pd.DataFrame(data = [x,y,z],index = ['2017','2018','2019'])

w.fillna(0,inplace=True)



w.loc['2017'] = w.loc['2017']/len(f_2017)*100

w.loc['2018'] = w.loc['2018']/len(f_2018)*100

w.loc['2019'] = w.loc['2019']/len(f_2019)*100



w.T[['2019']].plot(subplots=True, layout=(1,1),kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=False)

plt.gcf().set_size_inches(10,8)

plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Percentage of Female Respondents',fontsize=15)

plt.show()



# create data

education = f_2019['Education'].value_counts(sort = True)

labels = education.index

values = education.values

pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Gender Distribution in 2019',font=dict(size=10), legend=dict(orientation="h"))

layout = dict(title = 'Top-10 Countries with Respondents in 2019', font=dict(size=10), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20))])

              



fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)

 

plt.show()


