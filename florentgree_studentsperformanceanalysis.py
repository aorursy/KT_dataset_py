# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
data.info()
data['gender_num'] = data['gender'].replace({'female':0,'male':1})

data['race/ethnicity_num'] = data['race/ethnicity'].replace({'group A':0,'group B':1,'group C':2,'group D':3,'group E':4})

data['parental level of education_num'] = data['parental level of education'].replace({"bachelor's degree": 4, 'some college':2,

                                                                                       "master's degree":5,"associate's degree":3,

                                                                                       'high school':1, 'some high school':0})

data['lunch_num'] = data['lunch'].replace({'standard':0,'free/reduced':1})

data['test preparation course_num'] = data['test preparation course'].replace({'none':0,'completed':1})

data.head()
# Create a dataframe only with numerical columns

pcaData = data.select_dtypes(include=[np.number])

pcaData
# Standardize the data

scaler = StandardScaler()

pcaData_scaled = scaler.fit_transform(pcaData)



# Create the PCA model and fit standardised data

pca = PCA(n_components=np.shape(pcaData)[1]) # Use the maximum number of component

pca.fit(pcaData_scaled)

# Update the PCA with number of components that explains 80% of the variance

varianceExplained = 0.8

pca = PCA(n_components=next(x for x, val in enumerate(pca.explained_variance_ratio_.cumsum()) if val >= varianceExplained) + 1)

pca.fit(pcaData_scaled)

pcaData_projected = pca.transform(pcaData_scaled) # for scatter plots
pd.DataFrame(pca.components_,columns=data.select_dtypes(include=[np.number]).columns,index=['PC1','PC2','PC3','PC4','PC5'])
fig = go.Figure()

[fig.add_trace(go.Scatter(x=[0, pca.components_[0,x]],y=[0,pca.components_[2,x]],name=pcaData.columns[x])) for x in range(len(pca.components_[0,:]))]

fig.update_layout(plot_bgcolor='white',height=500, width=500,

                  showlegend=False,

                  shapes=[dict(type="circle",xref="x",yref="y",x0=-1,y0=-1,x1=1,y1=1,line_color="LightSeaGreen",)],

                  xaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Principal Component Analysis", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(title=dict(text='PC1', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.update_yaxes(title=dict(text='PC3', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.show()
# Manipulate dataframe to plot heatmap

dataRaceEducationParents = data[['parental level of education_num','race/ethnicity_num']]

dataRaceEducationParents = dataRaceEducationParents.groupby(['parental level of education_num','race/ethnicity_num']).size().unstack(fill_value=0)

dataRaceEducationParents_Perc = round(dataRaceEducationParents/dataRaceEducationParents.sum(axis=0)*100,2)

matrixPercentage = []

for x in range(len(dataRaceEducationParents_Perc.index)):

       matrixPercentage.append(dataRaceEducationParents_Perc.iloc[x].values.tolist())
fig = go.Figure()

fig.add_trace(go.Heatmap(z=matrixPercentage,colorscale='gnbu'))



fig.update_layout(plot_bgcolor='white', width = 1000,

                  xaxis=dict(title='race/ethnicity',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='parental level of education',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Level of education of the parents vs race/ethnicity", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.update_yaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))



fig.show()
listUniqueEducationLevel = sorted(data['parental level of education_num'].unique())
fig = go.Figure()

[fig.add_trace(go.Box(y=pcaData['math score'][pcaData['parental level of education_num']==x],name = data['parental level of education'][data['parental level of education_num'] == x].unique()[0],boxpoints='all')) for x in listUniqueEducationLevel]

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='parental level of education',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='math score',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Math score vs level of education of the parents", font=dict(family="Verdana",size=25,color="Black")))

fig.show()
fig = go.Figure()

[fig.add_trace(go.Box(y=pcaData['reading score'][pcaData['parental level of education_num']==x],name = data['parental level of education'][data['parental level of education_num'] == x].unique()[0],boxpoints='all')) for x in listUniqueEducationLevel]

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='parental level of education',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='reading score',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Reading score vs level of education of the parents", font=dict(family="Verdana",size=25,color="Black")))

fig.show()
fig = go.Figure()

[fig.add_trace(go.Box(y=pcaData['writing score'][pcaData['parental level of education_num']==x],name = data['parental level of education'][data['parental level of education_num'] == x].unique()[0],boxpoints='all')) for x in listUniqueEducationLevel]

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='parental level of education',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='writing score',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Writing score vs level of education of the parents", font=dict(family="Verdana",size=25,color="Black")))

fig.show()
dataParentEducationTestPreparation = data.groupby(['parental level of education','parental level of education_num','test preparation course_num']).size().unstack(fill_value=0).reset_index()

dataParentEducationTestPreparation['No_perc'] = round(dataParentEducationTestPreparation[0]/(dataParentEducationTestPreparation[0]+dataParentEducationTestPreparation[1])*100,2)

dataParentEducationTestPreparation['Yes_perc'] = round(dataParentEducationTestPreparation[1]/(dataParentEducationTestPreparation[0]+dataParentEducationTestPreparation[1])*100,2)

dataParentEducationTestPreparation = dataParentEducationTestPreparation.sort_values(by='parental level of education_num')
fig = go.Figure()

fig.add_trace(go.Bar(x=dataParentEducationTestPreparation['Yes_perc'],y=dataParentEducationTestPreparation['parental level of education'],orientation='h'))

fig.update_layout(plot_bgcolor='white', height = 400,

                  xaxis=dict(title='percentage of test preparation [%]',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='parental level of education',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Level of education of the parents vs percentage of test preparation", font=dict(family="Verdana",size=25,color="Black")))

fig.show()
fig = go.Figure()

[fig.add_trace(go.Scatter(x=[0, pca.components_[0,x]],y=[0,pca.components_[3,x]],name=pcaData.columns[x])) for x in range(len(pca.components_[0,:]))]

fig.update_layout(plot_bgcolor='white',height=500, width=500,

                  showlegend=False,

                  shapes=[dict(type="circle",xref="x",yref="y",x0=-1,y0=-1,x1=1,y1=1,line_color="LightSeaGreen",)],

                  xaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Principal Component Analysis", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(title=dict(text='PC1', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.update_yaxes(title=dict(text='PC4', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.show()
fig = go.Figure()

for x in ['math score','reading score', 'writing score']:

    fig.add_trace(go.Box(y=data[x][data['test preparation course_num']==0],name=x + ' (No)'))

    fig.add_trace(go.Box(y=data[x][data['test preparation course_num']==1],name=x + ' (Yes)'))

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='exam categories',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='exam results',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Exam results with (Yes) and without (No) preparation", font=dict(family="Verdana",size=25,color="Black")))

fig.show()
fig = go.Figure()

for x in ['math score','reading score', 'writing score']:

    fig.add_trace(go.Box(y=data[x][data['lunch']=='standard'],name=x + ' (Standard)'))

    fig.add_trace(go.Box(y=data[x][data['lunch']=='free/reduced'],name=x + ' (Free/reduced)'))

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='exam categories',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='exam results',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Exam results with standard and free/reduced lunch", font=dict(family="Verdana",size=25,color="Black")))

fig.show()
fig = go.Figure()

[fig.add_trace(go.Scatter(x=[0, pca.components_[0,x]],y=[0,pca.components_[1,x]],name=pcaData.columns[x])) for x in range(len(pca.components_[0,:]))]

fig.update_layout(plot_bgcolor='white',height=500, width=500,

                  showlegend=False,

                  shapes=[dict(type="circle",xref="x",yref="y",x0=-1,y0=-1,x1=1,y1=1,line_color="LightSeaGreen",)],

                  xaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Principal Component Analysis", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(title=dict(text='PC1', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.update_yaxes(title=dict(text='PC2', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.show()
fig = go.Figure()

for x in ['math score','reading score', 'writing score']:

    fig.add_trace(go.Box(y=data[x][data['gender']=='male'],name=x + ' (Male)'))

    fig.add_trace(go.Box(y=data[x][data['gender']=='female'],name=x + ' (Female)'))

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='exam categories',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='exam results',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Exam results per gender", font=dict(family="Verdana",size=25,color="Black")))

fig.show()