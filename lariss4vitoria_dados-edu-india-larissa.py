

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.offline as py

import matplotlib.pyplot as plt

from matplotlib.pyplot import pie

import plotly.express as px

import seaborn as sns



py.init_notebook_mode(connected=True)





dataframe = pd.read_csv('../input/xAPI-Edu-Data/xAPI-Edu-Data.csv')
dataframe.head(480)
print(dataframe.info())
# data só com alunos do ensino e medio

df = dataframe.loc[dataframe['StageID']=='HighSchool']
t = df[['Topic', 'gender']].groupby(['Topic'], as_index=False).count().sort_values(by=['gender'], ascending=False)

fig = go.Figure(data=[

    go.Bar(x=t['Topic'], y=t['gender'], text=t['gender'], textposition='auto' )

])

fig.update_layout(title='Quantidade de Alunos por Curso')

fig.show()

td = df[[ 'Topic', 'gender','GradeID']].groupby(['Topic','gender'], as_index=False).count().sort_values(by=['GradeID'], ascending=False)

fig = px.bar(td, x="Topic", y="GradeID", color='gender', text='GradeID', barmode='group',height=500)

fig.update_layout(title='Quantidade de Meninos e Meninas por Curso')

fig.show()
td = df[[ 'Topic','ParentschoolSatisfaction', 'gender']].groupby(['ParentschoolSatisfaction','Topic'], as_index=False).count().sort_values(by=['gender'], ascending=False)

fig = px.bar(td, x="Topic", y="gender", color='ParentschoolSatisfaction', text='gender', barmode='group',height=500)

fig.show()
temp = df[['Topic', 'raisedhands']].groupby(['Topic'], as_index=False).mean().sort_values(by=['raisedhands'], ascending=False)

fig = go.Figure()

fig.add_trace(go.Bar(x=temp['Topic'], y=temp['raisedhands'], text=temp['raisedhands'], textposition='auto' ))

fig.update_layout(title='Média de participação ativa em sala')

td = df[[ 'StudentAbsenceDays','ParentschoolSatisfaction', 'gender']].groupby(['ParentschoolSatisfaction','StudentAbsenceDays'], as_index=False).count().sort_values(by=['gender'], ascending=False)

fig = px.bar(td, x="ParentschoolSatisfaction", y="gender", color='StudentAbsenceDays', barmode='group',height=500)

fig.update_layout(title='Faltas X Sastifação Escolar')

fig.show()
td = df[['Discussion', 'ParentschoolSatisfaction', 'Topic']].groupby(['ParentschoolSatisfaction','Topic'], as_index=False).mean().sort_values(by=['Discussion'], ascending=False)

fig = px.bar(td, x="Topic", y="Discussion", color='ParentschoolSatisfaction',text='Discussion', barmode='group',height=500)

fig.update_layout(title='Grupo de Discussão X Satisfação Escolar')

fig.show()
td = df[['VisITedResources', 'ParentschoolSatisfaction', 'Topic']].groupby(['ParentschoolSatisfaction','Topic'], as_index=False).mean().sort_values(by=['VisITedResources'], ascending=False)

fig = px.bar(td, x="Topic", y="VisITedResources", color='ParentschoolSatisfaction',text='VisITedResources', barmode='group',height=500)

fig.update_layout(title='Visitas ao Conteudo X Satisfação Escolar')

fig.show()
td = df[['AnnouncementsView', 'ParentschoolSatisfaction', 'Topic']].groupby(['Topic','ParentschoolSatisfaction'], as_index=False).mean().sort_values(by=['AnnouncementsView'], ascending=False)

fig = px.bar(td, x="Topic", y="AnnouncementsView", color='ParentschoolSatisfaction',text='AnnouncementsView', barmode='group',height=600)

fig.update_layout(title='Visitas ao anuncio X Satisfação Escolar')

fig.show()
fig, axarr  = plt.subplots(2,2,figsize=(20,10))

sns.barplot(x='Topic', y='VisITedResources', data=df, ax=axarr[0,0],palette="pastel")

sns.barplot(x='Topic', y='AnnouncementsView', data=df, ax=axarr[0,1],palette="pastel")

sns.barplot(x='Topic', y='raisedhands', data=df, ax=axarr[1,0],palette="pastel")

sns.barplot(x='Topic', y='Discussion', data=df, ax=axarr[1,1],palette="pastel")