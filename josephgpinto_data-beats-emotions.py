import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import  Image

%matplotlib inline

import seaborn as sns

import itertools

import warnings

warnings.filterwarnings("ignore")

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization


multiform = pd.read_csv("../input/multipleChoiceResponses.csv",skiprows=1,low_memory=False)



print(multiform.shape)

multiform.head()
gender = multiform.iloc[:,1:2]

gender.columns = ['sex']

gender['cnt'] = 1

gender = gender[1:].groupby('sex')[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - gender

trace1 = go.Bar(x = gender["sex"]  , y = gender["cnt"],

                name = "gender",

                marker = dict(color=-gender['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "Gender of respondents",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "Gender",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
age_grp = multiform.iloc[:,[1,3]]

age_grp.columns = ['gender','age group']

age_grp['cnt'] = 1

age_grp = age_grp[1:].groupby(['gender','age group'])[['cnt']].count().sort_values(by=['gender','age group']).reset_index()



age_grpf = age_grp.iloc[0:12,:]

age_grpm= age_grp.iloc[12:24,:]

age_grpo = age_grp.iloc[24:,:]

age_grpo = age_grpo[1:].groupby(['age group'])[['cnt']].sum().reset_index()

age_grpo['gender'] ="others"



#bar - male

trace1 = go.Bar(x = age_grpm["age group"]  , y = age_grpm["cnt"],

                name = "male kagglers",

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)



#bar - female

trace2 = go.Bar(x = age_grpf["age group"] , y = age_grpf["cnt"],

                name = "female kagglers",

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)





#bar - others

trace3 = go.Bar(x = age_grpo["age group"] , y = age_grpo["cnt"],

                name = "gender not known",

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)







layout = go.Layout(dict(title = "kagglers by gender and age group",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "Age group",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1,trace2,trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
country = multiform.iloc[:,[4]]

country.columns = ['country']

country['cnt'] = 1

country  = country[1:].groupby(['country'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - gender

trace1 = go.Bar(x = country["country"]  , y = country["cnt"],

                name = "country",

                marker = dict(color=-country['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "kagglers by country",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "country",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
education = multiform.iloc[:,[5]]

education.columns = ['education']

education['cnt'] = 1

education  = education[1:].groupby(['education'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - gender

trace1 = go.Bar(x = education["education"]  , y = education["cnt"],

                name = "education",

                marker = dict(color=-education['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "Kagglers by Education",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "Education",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)

jobtitle = multiform.iloc[:,[7]]

jobtitle.columns = ['job title']

jobtitle['cnt'] = 1

jobtitle  = jobtitle[1:].groupby(['job title'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - gender

trace1 = go.Bar(x = jobtitle["job title"]  , y = jobtitle["cnt"],

                name = "Job Title",

                marker = dict(color=-jobtitle['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "Kagglers by profession",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "job title",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)

industry = multiform.iloc[:,[9]]

industry.columns = ['industry']

industry['cnt'] = 1

industry  = industry[1:].groupby(['industry'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - gender

trace1 = go.Bar(x = industry["industry"]  , y = jobtitle["cnt"],

                name = "Industry",

                marker = dict(color=-jobtitle['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "Kagglers by Industry",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "Industry",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
salary =multiform[['What is your gender? - Selected Choice',

                   'What is your current yearly compensation (approximate $USD)?']]



salary.columns = ['gender',"salary"]

salary['cnt'] = 1

salary = salary.dropna()



salist=[{'id':11,'salary':'I do not wish to disclose my approximate yearly compensation'},

       {'id':12,'salary':'0-10,000'},

       {'id':13,'salary':'10-20,000'},

       {'id':14,'salary':'20-30,000'},

       {'id':15,'salary':'30-40,000'},

       {'id':16,'salary':'40-50,000'},

       {'id':17,'salary':'50-60,000'},

       {'id':18,'salary':'60-70,000'},

       {'id':19,'salary':'70-80,000'},

       {'id':20,'salary':'80-90,000'},

       {'id':21,'salary':'90-100,000'},

       {'id':22,'salary':'100-125,000'},

       {'id':23,'salary':'125-150,000'},

       {'id':24,'salary':'150-200,000'}, 

       {'id':25,'salary':'200-250,000'},

       {'id':26,'salary':'250-300,000'},

       {'id':27,'salary':'300-400,000'},

       {'id':28,'salary':'400-500,000'},

       {'id':29,'salary':'500,000'}]



salist=pd.DataFrame(salist)

salary = pd.merge(salary,salist,how='left', on ='salary' )



salary  = salary[1:].groupby(['gender','salary','id'])[['cnt']].count().sort_values(by=['gender','id']).reset_index()

                                                                                  

salaryf = salary.iloc[0:17,:]

salarym= salary.iloc[17:35,:]

salaryo = salary.iloc[35:,:]

salaryo = salaryo[1:].groupby(['salary'])[['cnt']].sum().reset_index()

salaryo['gender'] ="others"





#bar - male

trace1 = go.Bar(x = salarym["salary"]  , y = salarym["cnt"],

                name = "male kagglers",

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)



#bar - female

trace2 = go.Bar(x = salaryf["salary"] , y = salaryf["cnt"],

                name = "female kagglers",

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)





#bar - others

trace3 = go.Bar(x = salaryo["salary"] , y = salaryo["cnt"],

                name = "gender not known",

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)







layout = go.Layout(dict( barmode='stack',title = "salary of kagglers by gender",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "salary group",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1,trace2,trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig,filename='stacked-bar')
tool =multiform[['What is your gender? - Selected Choice',

                 'What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice']]



tool.columns = ['gender',"tool"]

tool['cnt'] = 1

tool  = tool[1:].groupby(['gender','tool'])[['cnt']].count().sort_values(by='gender').reset_index()



toolf = tool.iloc[0:6,:]

toolm= tool.iloc[6:12,:]

toolo = tool.iloc[12:,:]

toolo = toolo[1:].groupby(['tool'])[['cnt']].sum().reset_index()

toolo['gender'] ="others"





#bar - male

trace1 = go.Bar(y = toolm["tool"]  , x = toolm["cnt"],

                name = "male kagglers",

                orientation = 'h',

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)



#bar - female

trace2 = go.Bar(y = toolf["tool"] , x = toolf["cnt"],

                name = "female kagglers",

                orientation = 'h',

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)





#bar - others

trace3 = go.Bar(y = toolo["tool"] , x = toolo["cnt"],

                name = "gender not known",

                orientation = 'h',

                marker = dict(line = dict(width = .5,color = "black")),

                opacity = .9)







layout = go.Layout(dict(barmode='stack',title = "primary tool used by gender",

                        autosize=False,width=850,height=500,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "Primary tool group",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1,trace2,trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig,filename='marker-h-bar')
language =multiform.loc[:,['What specific programming language do you use most often? - Selected Choice']]



language.columns = ['language']

language['cnt'] = 1

language = language.dropna()

language  = language[1:].groupby(['language'])[['cnt']].count().sort_values(by='cnt').reset_index()



#bar - gender

trace1 = go.Bar(x = language["language"]  , y = language["cnt"],

                name = "language",

                marker = dict(color=-language['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "language used on regular basis",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "language",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)

framework =multiform.loc[:,['What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Scikit-Learn',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - TensorFlow',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Keras',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - PyTorch',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Spark MLlib',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - H20',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Fastai',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Mxnet',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Caret',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Xgboost',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - mlr',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Prophet',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - randomForest',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - lightgbm',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - catboost',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - CNTK',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Caffe',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - None',

 'What machine learning frameworks have you used in the past 5 years? (Select all that apply) - Selected Choice - Other']]



framework.columns = ['Scikit-Learn','TensorFlow','Keras','PyTorch','Spark MLlib','H20','Fastai','Mxnet','Caret','Xgboost','mlr','Prophet','randomForest',

                     'lightgbm','catboost','CNTK','Caffe','None','Other']





framework = pd.melt(framework, value_vars=['Scikit-Learn','TensorFlow','Keras','PyTorch','Spark MLlib','H20','Fastai','Mxnet','Caret','Xgboost','mlr','Prophet','randomForest',

                     'lightgbm','catboost','CNTK','Caffe','None','Other'],var_name='framework', value_name='fm')

framework = framework.dropna()

framework['cnt'] = 1



framework  = framework[1:].groupby(['framework'])[['cnt']].count().sort_values(by='cnt').reset_index()



#bar - gender

trace1 = go.Bar(x = framework["framework"]  , y = framework["cnt"],

                name = "machine learning frameworks",

                marker = dict(color=-framework['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "Machine learning frameworks used in 5 years",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "machine learning frameworks",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
IDE = multiform.loc[:,["Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Jupyter/IPython",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - RStudio",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - PyCharm",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Visual Studio Code",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - nteract",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Atom",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - MATLAB",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Visual Studio",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Notepad++",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Sublime Text",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Vim",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - IntelliJ",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Spyder",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - None",

 "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Other"]]





IDE.columns = ['Jupyter/IPython','RStudio','PyCharm','Visual Studio Code','nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text','Vim','IntelliJ','Spyder',

                     'None','Other']



IDE = pd.melt(IDE, value_vars=['Jupyter/IPython','RStudio','PyCharm','Visual Studio Code','nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text','Vim','IntelliJ','Spyder',

                     'None','Other'],var_name='IDE', value_name='fm')

IDE = IDE.dropna()

IDE['cnt'] = 1



IDE  = IDE[1:].groupby(['IDE'])[['cnt']].count().sort_values(by='cnt').reset_index()



#bar - gender

trace1 = go.Bar(x = IDE["IDE"]  , y = IDE["cnt"],

                name = "integrated development environments (IDE's)",

                marker = dict(color=-IDE['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "integrated development environments (IDE's)",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "integrated development environments (IDE's)",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
cloud = multiform.loc[:,['Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - AWS Elastic Compute Cloud (EC2)',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Google Compute Engine',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - AWS Elastic Beanstalk',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Google App Engine',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Google Kubernetes Engine',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - AWS Lambda',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Google Cloud Functions',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - AWS Batch',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Azure Virtual Machines',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Azure Container Service',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Azure Functions',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Azure Event Grid',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Azure Batch',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Azure Kubernetes Service',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - IBM Cloud Virtual Servers',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - IBM Cloud Container Registry',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - IBM Cloud Kubernetes Service',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - IBM Cloud Foundry',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - None',

 'Which of the following cloud computing products have you used at work or school in the last 5 years (Select all that apply)? - Selected Choice - Other']]





cloud.columns = ['AWS Elastic Compute Cloud (EC2)','Google Compute Engine','AWS Elastic Beanstalk','Google App Engine','Google Kubernetes Engine','AWS Lambda',

                 'Google Cloud Functions','AWS Batch','Azure Virtual Machines','Azure Container Service','Azure Functions','Azure Event Grid','Azure Batch',

                 'Azure Kubernetes Service','IBM Cloud Virtual Servers','IBM Cloud Container Registry','IBM Cloud Kubernetes Service','BM Cloud Foundry','None','other']



cloud = pd.melt(cloud, value_vars=['AWS Elastic Compute Cloud (EC2)','Google Compute Engine','AWS Elastic Beanstalk','Google App Engine','Google Kubernetes Engine',

                                   'AWS Lambda','Google Cloud Functions','AWS Batch','Azure Virtual Machines','Azure Container Service','Azure Functions',

                                   'Azure Event Grid','Azure Batch','Azure Kubernetes Service','IBM Cloud Virtual Servers','IBM Cloud Container Registry',

                                   'IBM Cloud Kubernetes Service','BM Cloud Foundry','None','other'],var_name='cloud', value_name='fm')

                

cloud = cloud.dropna()

cloud['cnt'] = 1



cloud  = cloud[1:].groupby(['cloud'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - gender

trace1 = go.Bar(x = cloud["cloud"]  , y = cloud["cnt"],

                name = "cloud computing products",

                marker = dict(color=-cloud['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "cloud computing products",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "cloud computing products",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
learning = multiform.loc[:,['On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Learn',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.AI',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - developers.google.com',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - TheSchool.AI',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Online University Courses',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None',

 'On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other']]





learning.columns = ['Udacity','GCoursera','edX','DataCamp','DataQuest','Kaggle Learn','Fast.AI','developers.google.com','Udemy','TheSchool.AI',

                    'Online University Courses','None','other']



learning = pd.melt(learning, value_vars=['Udacity','GCoursera','edX','DataCamp','DataQuest','Kaggle Learn','Fast.AI','developers.google.com','Udemy','TheSchool.AI',

                    'Online University Courses','None','other'],var_name='learning', value_name='fm')

                

learning = learning.dropna()

learning['cnt'] = 1



learning  = learning[1:].groupby(['learning'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - learning

trace1 = go.Bar(x = learning["learning"]  , y = cloud["cnt"],

                name = "online platforms for learning",

                marker = dict(color=-learning['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "online platforms for learning",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "online platforms for learning",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
media = multiform.loc[:,['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - r/machinelearning',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle forums',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Fastai forums',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Siraj Raval YouTube Channel',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - DataTau News Aggregator',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Linear Digressions Podcast',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Cloud AI Adventures (YouTube)',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - FiveThirtyEight.com',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - ArXiv & Preprints',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - FastML Blog',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - KDnuggets Blog',

 "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - O'Reilly Data Newsletter",

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Partially Derivative Podcast',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - The Data Skeptic Podcast',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Medium Blog Posts',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Towards Data Science Blog',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Analytics Vidhya Blog',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None/I do not know',

 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other']]







media.columns = ['Twitter','Hacker News','r/machinelearning',"Kaggle forums",'Fastai forums','Siraj Raval YouTube Channel','DataTau News Aggregator','Linear Digressions Podcast',

                 'Cloud AI Adventures (YouTube)','FiveThirtyEight.com','ArXiv & Preprints','Journal Publications','FastML Blog',"KDnuggets Blog","O'Reilly Data Newsletter",

                "Partially Derivative Podcast","The Data Skeptic Podcast","Medium Blog Posts","Towards Data Science Blog","Analytics Vidhya Blog",

                "None/I do not know","Other"]



media = pd.melt(media, value_vars=['Twitter','Hacker News','r/machinelearning',"Kaggle forums",'Fastai forums','Siraj Raval YouTube Channel','DataTau News Aggregator','Linear Digressions Podcast',

                 'Cloud AI Adventures (YouTube)','FiveThirtyEight.com','ArXiv & Preprints','Journal Publications','FastML Blog',"KDnuggets Blog","O'Reilly Data Newsletter",

                "Partially Derivative Podcast","The Data Skeptic Podcast","Medium Blog Posts","Towards Data Science Blog","Analytics Vidhya Blog",

                "None/I do not know","Other"],var_name='media', value_name='fm')

                

media= media.dropna()

media['cnt'] = 1



media  = media[1:].groupby(['media'])[['cnt']].count().sort_values(by='cnt').reset_index()





#bar - media

trace1 = go.Bar(x = media["media"]  , y = cloud["cnt"],

                name = "favorite media sources",

                marker = dict(color=-media['cnt'].values, colorscale='Portland', showscale=False),

                opacity = .9)



layout = go.Layout(dict(title = "favorite media sources",

                        autosize=False,width=850,height=700,

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "favorite media sources",automargin=True,

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "kagglers",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)