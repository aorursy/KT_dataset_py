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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



class style:

    BOLD = '\033[1m'

    END = '\033[0m'



import warnings

warnings.filterwarnings("ignore")



from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



# Graphics in retina format

%config InlineBackend.figure_format = 'retina'



# set figure size

plt.rcParams['figure.figsize'] = 10, 10
df = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df.drop(index=0,inplace=True)



df['Q3'] = df['Q3'].str.replace('United States of America','USA')



top_10_countries = pd.DataFrame(round(df['Q3'].value_counts()/len(df)*100)[:10])
plt.rcParams['axes.labelsize'] = 14

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 18

plt.rcParams['axes.titleweight'] = 'bold'







plt.figure(figsize=(10,5))

ax = sns.barplot(x=top_10_countries['Q3'],y=top_10_countries.index,data=top_10_countries,palette='Blues_d',orient='h')

for p in ax.patches:

    ax.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y() + 0.9),

                xytext=(5, 10), textcoords='offset points')

plt.ylabel('Country')

plt.xlabel('')

plt.title('Top 10 Countries participated in the 2019 Survey ')

plt.show()

india_Age = pd.DataFrame(df[df['Q3']=='India']['Q1'].value_counts())

usa_Age = pd.DataFrame(df[df['Q3']=='USA']['Q1'].value_counts())





pie_india = go.Pie(labels=india_Age.index,values=india_Age['Q1'],name="India",hole=0.3,domain={'x': [0,0.46]})

pie_USA = go.Pie(labels=usa_Age.index,values=usa_Age['Q1'],name="USA",hole=0.3,domain={'x': [0.52,1]})



layout = dict(title = 'Age of Respondents in 2019', font=dict(size=10), legend=dict(orientation="v"),

              annotations = [dict(x=0.2, y=0.5, text='India', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='USA', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_india, pie_USA], layout=layout)

py.iplot(fig)

df['All'] = 1

Education = df.pivot_table(index='Q4',columns='Q3',values='All',aggfunc=len)

top2_countries = top_10_countries[:2].index.tolist()

Education = Education[top2_countries]

for i in top2_countries:

    Education[i] = round((Education[i]/sum(Education[i]))*100)





ax = Education.plot.barh()

plt.ylabel('Education')

plt.title('India vs USA Education Comparison')

plt.legend(title='Country')

plt.show()
india_edu = pd.DataFrame(df[df['Q3']=='India']['Q5'].value_counts())

usa_edu = pd.DataFrame(df[df['Q3']=='USA']['Q5'].value_counts())





pie_india = go.Pie(labels=india_edu.index,values=india_edu['Q5'],name="India",hole=0.4,domain={'x': [0,0.46]})

pie_USA = go.Pie(labels=usa_edu.index,values=usa_edu['Q5'],name="USA",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Occupation of Respondents in 2019', font=dict(size=10), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='India', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='USA', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_india, pie_USA], layout=layout)

py.iplot(fig)

df['income'] = df['Q10'].apply(lambda x: 35000 if x=='30,000-39,999' else 6250 if x=='5,000-7,499' else 275000

if x=='250,000-299,999' else 4500 if x=='4,000-4,999' else  65000 if x=='60,000-69,999' else

12500 if x=='10,000-14,999' else

85000 if x=='80,000-89,999' else

500 if x=='$0-999' else

2500 if x=='2,000-2,999' else

75000 if x=='70,000-79,999' else

95000 if x=='90,000-99,999' else

137500 if x=='125,000-149,999' else

45000 if x=='40,000-49,999' else

22500 if x=='20,000-24,999' else

17500 if x=='15,000-19,999' else

112500 if x=='100,000-124,999' else

8750 if x=='7,500-9,999' else

175000 if x=='150,000-199,999' else

27500 if x=='25,000-29,999' else

3500 if x=='3,000-3,999' else

1500 if x=='1,000-1,999' else

225000 if x=='200,000-249,999' else

55000 if x=='50,000-59,999' else

750000 if x=='> $500,000' else

400000 if x=='300,000-500,000' else 0)
df_IndUSA = df[df['Q3'].isin(['India','USA'])]

Sal = round(df_IndUSA[~df_IndUSA['Q5'].isin(['Not employed','Student'])].groupby('Q3')['income'].mean())



x = Sal.index

y = Sal.values





# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=x,

            y=y,

            text=y,

            width=0.4,

            textposition='auto',

            marker=dict(color='green')

 )])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.update_layout(yaxis=dict(title='Salary (in USD)'),width=700,height=500,

                  title='Salary Overall level India vs USA',

                  xaxis=dict(title='Country'))

fig.show()

Salary = round(df_IndUSA.pivot_table(index="Q5", columns="Q3", values="income").fillna(0))

India_Salary = Salary['India'].values*-1

USA_Salary = Salary['USA'].values



y = Salary.index



layout = go.Layout(yaxis=go.layout.YAxis(title='salary'),

                   xaxis=go.layout.XAxis(

                       range=[-150000, 150000],

                       tickvals=[-150000, 0, 150000],

                       ticktext=[150000, 0, 150000]),

                   barmode='overlay',

                   title='Salary Comparision by Job titile',

                   bargap=0.1)



data = [go.Bar(y=y,

               x=USA_Salary,

               orientation='h',

               name='USA',

               hoverinfo='x',

               marker=dict(color='darkgreen')

               ),

        go.Bar(y=y,

               x=India_Salary,

               orientation='h',

               name='India',

               text= -1* India_Salary.astype('int'),

               hoverinfo='text',

               marker=dict(color='blueviolet')

               )]



py.iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid')

df_IndUSA = df[df['Q3'].isin(['India','USA'])]

Company_size = df_IndUSA.pivot_table(index='Q7',columns=['Q6','Q3'],values='All',aggfunc=len)

Company_cols = Company_size.columns.to_list()

for i in Company_cols:

    Company_size[i] = round((Company_size[i]/sum(Company_size[i]))*100)



Company_size.style.apply(lambda x: ["background: lightgreen" if v > 30 else "" for v in x], axis = 1)