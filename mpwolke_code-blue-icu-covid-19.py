#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQxkkAPfOtW7Gbg-gyXxmW_vq1VMnHsIFWP4K8gWELPCYByyHNG&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs00134-020-06022-5/MediaObjects/134_2020_6022_Fig2_HTML.png?as=webp',width=400,height=400)
df = pd.read_csv("../input/cleveland-clinic-heart-disease-dataset/processed_cleveland.csv")

df.head().style.background_gradient(cmap='summer')
#sample codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

fig = go.Figure(go.Treemap(

    labels = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],

    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"]

))



fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs00134-020-06022-5/MediaObjects/134_2020_6022_Fig1_HTML.png?as=webp',width=400,height=400)
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#make a df it's grouped by "Genre"

gb_age =df.groupby("age").sum()



gb_age.head()
# codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

age = list(gb_age.index)

score = list(gb_age.num)



print(age)

print(score)
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#first treemap

test_tree = go.Figure(go.Treemap(

    labels =  age,

    parents=[""]*len(age),

    values =  score,

    textinfo = "label+value"

))



test_tree.show()
import plotly.express as px



# Grouping it by Age and Chest Pain

plot_data = df.groupby(['age', 'cp'], as_index=False).sex.sum()



fig = px.bar(plot_data, x='age', y='sex', color='cp')

fig.update_layout(

    title_text='Chest Pain by Age and Gender',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Age and all the others Codes by Andre Sionek (Kaggle's Survey)   

plot_data = df.groupby(['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'], as_index=False).sex.sum()



fig = px.line_polar(plot_data, theta='age', r='sex', color='cp')

fig.update_layout(

    title_text='Heart Disease Risk Factors',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Age and other Risk Factors - Code Andre Sionek (Kaggle Survey) 

plot_data = df.groupby(['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'], as_index=False).sex.sum()



fig = px.line(plot_data, x='age', y='sex', color='cp')

fig.update_layout(

    title_text='Heart Disease Risk Factors',

    height=500, width=1000)

fig.show()
# venn2

from matplotlib_venn import venn2

age = df.iloc[:,0]

cp = df.iloc[:,1]

trestbps = df.iloc[:,2]

chol = df.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(age)-15, len(cp)-15, 15), set_labels = ('age', 'cp'))

plt.show()
df1 = pd.read_csv("../input/country-health-indicators/country_health_indicators_v2.csv")

df1.head()
cnt_srs = df1['Diabetes, blood, & endocrine diseases (%)'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Diabetes and Endocrine Diseases',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Diabetes, blood, & endocrine diseases (%)")
cnt_srs = df1['Cancers (%)'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Cancers',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Cancers (%)")
cnt_srs = df1['Respiratory diseases (%)'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='Respiratory diseases (%)',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Respiratory diseases (%)")
cnt_srs = df1['HIV/AIDS and tuberculosis (%)'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Oranges',

        reversescale = True

    ),

)



layout = dict(

    title='HIV/AIDS and tuberculosis ',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="HIV/AIDS and tuberculosis (%)")
cnt_srs = df1['Malaria & neglected tropical diseases (%)'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Malaria & neglected tropical diseases',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Malaria & neglected tropical diseases (%)")
#Let's visualise the evolution of results

hindicators = df1.groupby('Country_Region').sum()[['Cancers (%)', 'Diabetes, blood, & endocrine diseases (%)', 'HIV/AIDS and tuberculosis (%)', 'Cardiovascular diseases (%)', 'Respiratory diseases (%)']]

#evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100

#evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

hindicators.head()
plt.style.use('dark_background')

plt.figure(figsize=(20,7))

plt.plot(hindicators['Cancers (%)'], label='Cancers')

plt.plot(hindicators['Diabetes, blood, & endocrine diseases (%)'], label='Diabetes')

plt.plot(hindicators['HIV/AIDS and tuberculosis (%)'], label='HIV and Tuberculosis')

plt.plot(hindicators['Cardiovascular diseases (%)'], label='Cardiovascular diseases')

plt.plot(hindicators['Respiratory diseases (%)'], label='Respiratory diseases')

plt.legend()

#plt.grid()

plt.title('Countries Health Indicators')

plt.xticks(hindicators.index,rotation=90)

plt.xlabel('Countries')

plt.ylabel('Health Indicators')

plt.show()
#What about the evolution of China Diagnosed Worldometer rate ?

plt.figure(figsize=(20,7))

plt.plot(hindicators['Respiratory diseases (%)'], label='Respiratory Diseases')

plt.legend()

plt.grid()

plt.title('')

plt.xticks(hindicators.index,rotation=90)

plt.ylabel('Rate %')

plt.show()
#This is another way of visualizing the evolution: plotting the increase evolution (difference from day to day)

diff_hindicators = hindicators.diff().iloc[1:]

plt.figure(figsize=(20,7))

plt.plot(diff_hindicators['Diabetes, blood, & endocrine diseases (%)'], label='Diabetes and Endocrine Diseases')

plt.legend()

#plt.grid()

plt.title('Diabetes')

plt.xticks(hindicators.index,rotation=90)

plt.ylabel('Rate %')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQPxQ5MQkxM1W1t54J81FXCCX8RD8vGDJ7I_JSWompOtfq_U1SV&usqp=CAU',width=400,height=400)
df1.plot.area(y=['Cancers (%)', 'Diabetes, blood, & endocrine diseases (%)','Liver disease (%)','Cardiovascular diseases (%)','HIV/AIDS and tuberculosis (%)'],alpha=0.4,figsize=(12, 6));
#Code from Prashant Banerjee @Prashant111

labels = df['age'].value_counts().index

size = df['age'].value_counts()

colors=['cyan','pink']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Age', fontsize = 20)

plt.legend()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQuT-7drSNDUZFOdEnrXVNfPeEoCafLd6krggpkPo96OIuVlcWn&usqp=CAU',width=400,height=400)