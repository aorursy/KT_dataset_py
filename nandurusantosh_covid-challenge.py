# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

import pycountry

import plotly.express as px

from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings('ignore')
df=pd.read_excel('/kaggle/input/covid-19.xlsx',sheet_name='Sheet1')

df['travel_history_dates']= pd.to_datetime(df['travel_history_dates'],infer_datetime_format=True) 

df['date_admission_hospital']= pd.to_datetime(df['date_admission_hospital'],infer_datetime_format=True)

df['date_onset_symptoms']= pd.to_datetime(df['date_onset_symptoms'],infer_datetime_format=True)

df['date_confirmation']= pd.to_datetime(df['date_confirmation'],infer_datetime_format=True)
df.info()
df['city']= df['city'].fillna('')

df['province']= df['province'].fillna('')

df['symptoms']= df['symptoms'].fillna('')

df['travel_history_location']=df['travel_history_location'].fillna(value='Not Available')
df.head()
df.isnull().sum()
df['country'].value_counts()
datagroup_country= df.groupby('country')['ID','city'].count().reset_index()

datagroup_country.rename(columns={'ID':'Number of Cases'},inplace=True)
fig = px.bar(datagroup_country.sort_values('Number of Cases'), 

             x='Number of Cases', y="country", title='Number of Cases by Country', text='Number of Cases', orientation='h', 

             width=1000, height=500, range_x = [0,400],color_discrete_sequence=px.colors.qualitative.Dark2)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
fig = px.choropleth(datagroup_country,locations="country",locationmode='country names',

                    color="Number of Cases",range_color=[1,400],

                    color_continuous_scale="algae",hover_name='country')



fig.show()
fig = px.choropleth(datagroup_country,locations="country",locationmode='country names',

                    color="Number of Cases",range_color=[1,400],

                    color_continuous_scale="algae",hover_name='country', scope='asia')



fig.show()
datagroup_gender= pd.DataFrame(df.groupby('sex')['ID'].count().reset_index())

datagroup_gender.rename(columns={'sex':'Gender','ID':'Number of Cases'},inplace=True)

datagroup_gender['Percentage of Cases']=round(((datagroup_gender['Number of Cases'])/(datagroup_gender['Number of Cases'].sum()))*100,0)
datagroup_gender.style.background_gradient(cmap='Greens')
fig = px.box(df, y="age",color_discrete_sequence=px.colors.qualitative.Dark2)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
wuhantravel=pd.DataFrame(df['travel_history_location'].value_counts().reset_index()).iloc[1:20]

wuhantravel.rename(columns={'index':'Coutry/Province','travel_history_location':'Number of Travelers'},inplace=True)

wuhantravel.style.background_gradient(cmap='Greens')
feverandrespiratorycondition=df[

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('respiratory symptoms'))) |

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('rhinorrhoea'))) |

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('expectoration'))) |

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('pharyngeal discomfort')))|

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('pneumonia')))|

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('pharyngalgia')))|

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('sputum')))|

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('shortness of breath')))|

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('dyspnea'))) |

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('pleuritic chest pain'))) | 

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('pleural effusion')))|

    ((df['symptoms'].str.contains('fever')) & (df['symptoms'].str.contains('pneumonitis')))

  ]
len(feverandrespiratorycondition)
df['country'].unique()
country_nochina=['France', 'Japan', 'Nepal', 'Singapore', 

 'South Korea','Thailand', 'United States', 

 'Vietnam', 'Australia', 'Canada',

 'Cambodia', 'Malaysia', 'Spain', 

 'Romania']

othercountries_data=df[df['country'].isin(country_nochina)]
gendersplit_othercountries=pd.crosstab(index=othercountries_data['country'],columns=othercountries_data['sex'],margins=True,margins_name='Total People').reset_index()

gendersplit_othercountries.style.background_gradient(cmap='Greens')
gendersplit_othercountries['femaleproportion']=round(gendersplit_othercountries['female']/gendersplit_othercountries['Total People'],2)

gendersplit_othercountries['maleproportion']=round(gendersplit_othercountries['male']/gendersplit_othercountries['Total People'],2)

x=gendersplit_othercountries[['country','femaleproportion','maleproportion']]
x.style.background_gradient(cmap='Greens')
othercountries_data['city'].unique()
city=othercountries_data['city'].value_counts().to_frame().reset_index().drop(axis=0,index=0).rename(columns={'index':'City','city':'Number of cases'})
city.style.background_gradient(cmap='Greens')
maxcitycases=city[city['City'].str.contains('Singapore')]

maxcitycases['Number of cases'].sum()
china_data=df[df['country']=='China']
chinaprovince_data=pd.DataFrame(china_data['province'].value_counts()).reset_index()

chinaprovince_data.rename(columns={'index':'Province in China','province':'Number of Cases'},inplace=True)

chinaprovince_data.style.background_gradient(cmap='Greens')
fig = px.bar(chinaprovince_data.sort_values('Number of Cases'), 

             x='Number of Cases', y="Province in China", title='Number of Cases across provinces of China', orientation='h', 

             range_x = [1,90],color_discrete_sequence=px.colors.qualitative.Dark2,width=1000, height=1000,text='Number of Cases')

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
chinacity_data=pd.DataFrame(china_data['city'].value_counts()).reset_index()

chinacity_data.rename(columns={'index':'City in China','city':'Number of cases'},inplace=True)

chinacity_data.drop(axis=0,index=0)
len(china_data[china_data['city'].str.contains('Beijing')])
len(china_data[china_data['city'].str.contains('Zhengzhou')])
wuhantravel_data=df[df['travel_history_location'].str.contains('Wuhan')]
len(wuhantravel_data)
symptoms_wuhantravel_data=pd.DataFrame(wuhantravel_data['symptoms'].value_counts()).reset_index()
symptoms_wuhantravel_data.rename(columns={'index':'Symptoms','symptoms':'Number of cases'},inplace=True)

symptoms_wuhantravel_data.drop(axis=0,index=0,inplace=True)

symptoms_wuhantravel_data.style.background_gradient(cmap='Greens')
len(symptoms_wuhantravel_data[symptoms_wuhantravel_data['Symptoms'].str.contains('fever')])
df['city'].unique()
shaanxiprovince=df[df['province']=='Shaanxi']

shaanxiprovince['city'].value_counts()
xianyang_city=len(shaanxiprovince[shaanxiprovince['city'].str.contains('Xi')])

xianyang_city
shaanxiprovince=df[df['province']=='Shaanxi']

ankang_city=len(shaanxiprovince[shaanxiprovince['city'].str.contains('Ankang City')])

ankang_city
anhuiprovince=df[df['province']=='Anhui']

anhuiprovince['city'].value_counts()
hefei_city=len(anhuiprovince[anhuiprovince['city'].str.contains('Hefei City')])

hefei_city
gansuprovince=df[df['province']=='Gansu']

gansuprovince['city'].value_counts()
lanzhou_city=len(gansuprovince[gansuprovince['city'].str.contains('Lanzhou')])

lanzhou_city
tianshui_city=len(gansuprovince[gansuprovince['city'].str.contains('Tianshui')])

tianshui_city
beijingprovince=df[df['province']=='Beijing']

beijingprovince['city'].value_counts()
beijing_city=len(beijingprovince[beijingprovince['city'].str.contains('Beijing')])

beijing_city
guangxiprovince=df[df['province']=='Guangxi']

guangxiprovince['city'].value_counts()
fang_city=len(guangxiprovince[guangxiprovince['city'].str.contains('Fangchenggang')])

fang_city
hongkongprovince=df[df['province']=='Hong Kong']

hongkongprovince['city'].value_counts()
hongkongcity=len(hongkongprovince[hongkongprovince['city'].str.contains('Hong')])

hongkongcity
tianjinprovince=df[df['province']=='Tianjin']

tianjinprovince['city'].value_counts()
japan=df[df['country']=='Japan']

japan['province'].value_counts()
southkorea=df[df['country']=='South Korea']

southkorea['province'].value_counts()
seoul=len(southkorea[southkorea['city'].str.contains('Seoul')])

seoul
topcities=pd.Series({'Seoul':9,

 'Tianjin City':12,

 'Hong Kong City':15,

 'Fangchenggang City':16,

 'Beijing City':23,

 'Lanzhou City':19,

 'Hefei City':54,

 'Xianyang City':48,

 'Singapore City':65}).to_frame().reset_index().rename(columns={'index':'City',0:'Cases'})
topcities
fig = px.bar(topcities.sort_values('Cases'), 

             x='Cases', y='City', title='Top 10 cities with highest number of cases', orientation='h', 

             range_x = [0,70],color_discrete_sequence=px.colors.qualitative.Dark2,width=500, height=500,text='Cases')

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
df['lag']= df['date_confirmation']- df['date_onset_symptoms']
genderimpact=pd.crosstab(columns=df['sex'],index=df['lag']).reset_index().iloc[1:]

genderimpact['lag']=genderimpact['lag'].astype(str)

genderimpact['lag']=genderimpact['lag'].str.slice(start=0,stop=7)

genderimpact
fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Lag in Male","Lag in Female"))
temp = genderimpact.sort_values('male', ascending=False)



fig.add_trace(go.Bar( y=temp['male'], x=temp['lag'],  

                     marker=dict(color=temp['male'], coloraxis="coloraxis")),1, 1)

                     

temp1 = genderimpact.sort_values('female', ascending=False)



fig.add_trace(go.Bar( y=temp1['female'], x=temp1['lag'],  

                     marker=dict(color=temp1['male'], coloraxis="coloraxis")),

              1, 2)                     

                     



fig.update_layout(coloraxis=dict(colorscale='Greens'), showlegend=False,title_text="Lag in Females and Males",plot_bgcolor='rgb(250, 242, 242)')

fig.show()
countrylag=pd.crosstab(columns=df['country'],index=df['lag']).reset_index()

countrylag['lag'].astype(str)

countrylag['lag']=genderimpact['lag'].str.slice(start=0,stop=7)

countrylag
temp_china = countrylag.sort_values('China', ascending=False)



fig = px.bar(temp_china.sort_values('China'), 

             y='lag', x="China", title='Lag in China', orientation='h', 

             range_x = [0,50],color_discrete_sequence=px.colors.qualitative.Dark2,width=1000, height=1000,text='China')

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
increasingtrend= df.groupby(['date_confirmation', 'country']).count().reset_index()
increasingtrend['date_confirmation'] = increasingtrend['date_confirmation'].dt.strftime('%m/%d/%Y')
increasingtrend['size']=increasingtrend['ID']*1000000000000000
fig = px.scatter_geo(increasingtrend, locations="country", locationmode='country names', 

                     color='ID', 

                     range_color= [0, 33],  

                     color_continuous_scale="algae", animation_frame='date_confirmation',projection='natural earth',

                     size='size')

fig.show()

agewisedata =df['age'].value_counts().to_frame().reset_index()

agewisedata.rename(columns={'index':'age','age':'Number of Cases'},inplace=True)
fig = px.scatter(data_frame=agewisedata,y='Number of Cases',x='age',

                 color_discrete_sequence=px.colors.qualitative.Dark2,

                title='Cases across different ages',width=800,

    height=800)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
df['travel_history_location'].value_counts()
df[df['travel_history_location']=='no']
singa=df[df['country']=='Singapore']

singa['travel_history_location'].value_counts()
singa[singa['date_onset_symptoms']=='2020-01-21'].head(1)
wuhantravel= df[df['travel_history_location']=='Wuhan']

wuhantravel['country'].value_counts()
japtravel=wuhantravel[wuhantravel['country']=='Japan']['age'].value_counts().to_frame().reset_index()

japtravel.rename(columns={'index':'Age','age':'Number of people traveled'},inplace=True)

fig = px.bar(japtravel.sort_values('Number of people traveled'), 

             y='Age', x='Number of people traveled', title='Travel of Japenese People', orientation='h', 

             range_x = [0,5],color_discrete_sequence=px.colors.qualitative.Dark2,width=500, height=500,text='Number of people traveled'

            , category_orders=dict(age=japtravel['Age']))

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
jap=df[(df['country']=='Japan')&(df['travel_history_location']!='Wuhan')]

japimpact=jap['age'].value_counts().to_frame().reset_index()

japimpact.rename(columns={'index':'Age','age':'Number of Cases'},inplace=True)

japimpact
plt.figure(figsize=(10,10))

sns.barplot(y='Age', x='Number of Cases', data= japimpact, palette='Greens', orient='h',ci=0)

plt.show()
df[(df['province']=='Osaka') | (df['travel_history_location']=='Osaka')]
wuhantravel['country'].value_counts()
df[df['country']=='Japan']['date_onset_symptoms'].min()
df[(df['country']=='Japan')&(df['date_onset_symptoms']=='2020-01-03')]
df['country'].value_counts()
df[(df['country']=='China')& (df['date_onset_symptoms']=='2019-12-31')]
df[df['country']=='Nepal']
df[df['country']=='Cambodia']
df[df['country']=='Malaysia']
df[df['country']=='France']
df[df['country']=='Canada']
df[df['country']=='United States']
df[df['country']=='Australia']
df[df['country']=='Vietnam']
df[df['country']=='Spain']
df[df['country']=='Romania']
df[(df['country']=='China')& (df['travel_history_location']=='Hubei')]['province'].value_counts()
df[(df['country']=='China')& (df['travel_history_location']=='Wuhan')]['province'].value_counts()
df[df['country']=='China']['country'].value_counts()
df[(df['country']=='China')& (df['travel_history_location']=='Wuhan')]['province'].value_counts().to_frame()['province'].sum()
traveltodeath=df[(df['travel_history_location']=='Wuhan') |(df['travel_history_location']=='Hubei')| (df['travel_history_location']=='China')]
traveltodeath['country'].value_counts()
traveltodeath[traveltodeath['country']=='Singapore']['travel_history_dates'].min()
df[df['country']=='Singapore']['date_onset_symptoms'].min()
traveltodeath[traveltodeath['country']=='Japan']['travel_history_dates'].min()
df[df['country']=='Japan']['date_onset_symptoms'].min()
traveltodeath[traveltodeath['country']=='United States']['travel_history_dates'].min()
df[df['country']=='United States']['date_onset_symptoms'].min()
traveltodeath[traveltodeath['country']=='Australia']['travel_history_dates'].min()
df[df['country']=='Australia']['date_onset_symptoms'].min()
traveltodeath[traveltodeath['country']=='France']['travel_history_dates'].min()
df[df['country']=='France']['date_onset_symptoms'].min()
df[df['travel_history_location']=='Thailand'].iloc[3:7][['ID','age','sex','city','province','date_onset_symptoms','date_confirmation','symptoms','travel_history_dates','travel_history_location']]
numero=df.groupby(by='date_confirmation')['ID'].count().to_frame().reset_index()

numero
fig=px.line(numero,x='date_confirmation',y='ID',color_discrete_sequence=px.colors.qualitative.Dark2)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()