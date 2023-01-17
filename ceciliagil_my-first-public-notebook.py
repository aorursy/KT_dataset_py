# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



datos_2019=pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')
#list all countries

#datos_2019.describe()

#datos_2019.Q3.value_counts()

total_responders=datos_2019.Q1.count()



#countries to analyze

countries=['Brazil','Chile','Argentina','Bolivia','Paraguay','Uruguay']



datos_filtrados=datos_2019.loc[datos_2019.Q3.isin(countries)]

datos_filtrados.index=range(0,len(datos_filtrados),1)

#datos_filtrados

total_responders_Mercosur=datos_filtrados.Q1.count()

print ("Total respondents: ", total_responders)

print ("Total respondents Mercosur: ", total_responders_Mercosur)

print ("% respondents Mercosur: ", round((total_responders_Mercosur/total_responders),2))

#datos_filtrados.shape

#datos_filtrados.head()

#datos_filtrados.describe()

#datos_filtrados.columns.tolist()
#Colors repretentatives of countries

colors=['green', 'lightskyblue','red']                 

#datos_filtrados.Q3.value_counts()



fig=go.Figure(data=[go.Pie(labels=datos_filtrados.Q3, marker=dict(colors=colors))])

fig.update_layout(title_text="Respondents for Country")

fig.update_layout(legend_orientation="h")

fig.show()
#I Verify the data to represent

#print(datos_filtrados.Q1.value_counts())

#Encoding Age in dictionary

#age=datos_filtrados.Q1.unique()

#print(age)

ages={'18-21': 0,'22-24': 1,'25-29': 2,'30-34': 3,'35-39': 4,'40-44': 5,'45-49': 6,'50-54': 7,'55-59': 8,'60-69': 9,'70+': 10}

i=0

while (i<total_responders_Mercosur):

    datos_filtrados.loc[i,'EQ1']=ages[datos_filtrados.loc[i,'Q1']]

    i=i+1

                                      

#datos_filtrados['EQ1'].sum().plot(kind='bar')

#plt.show()

plt.figure(figsize=(14,6))

sns.distplot(a=datos_filtrados.EQ1, kde=False, color="red")

plt.title("Respondents for Age")

plt.xlabel("Years")

plt.ylabel("Count")

#show as legends in order ascendent

plt.xticks(range(len(ages)),tuple(ages.keys()))

plt.show()

#print(datos_filtrados.Q4.unique())



degree={'I prefer not to answer':0,

        'No formal education past high school':1,

        'Some college/university study without earning a bachelor’s degree':2,

       'Bachelor’s degree':3,

       'Professional degree':4,

       'Master’s degree':5,

        'Doctoral degree':6     

       }

#create a new column DQ4 to Q4 

datos_filtrados['DQ4']=datos_filtrados['Q4'].copy()

i=0

while (i<total_responders_Mercosur):

    #Cambio las filas sin respuestas a la respuesta 0 

    if pd.isna(datos_filtrados.loc[i,'Q4']):

        datos_filtrados.loc[i,'EQ4']=0

        datos_filtrados.loc[i,'DQ4']='I prefer not to answer'

    else:

        datos_filtrados.loc[i,'EQ4']=degree[datos_filtrados.loc[i,'Q4']]

    i=i+1



fig=go.Figure(data=[go.Pie(labels=datos_filtrados.DQ4)])

fig.update_layout(title_text="Degree Education")

fig.update_layout(legend_orientation="h")

#fig.update_layout(legend=dict(x=-.1, y=1.2))

#plt.legend(loc="upper center")

fig.show()

      
#Encoding Rol in dictionary

#print(datos_filtrados.Q5.unique())



rol={'Data Scientist' : 0,'Data Analyst': 1,'Student': 2,'Business Analyst': 3,

     'Statistician': 4,'Software Engineer': 5,'Product/Project Manager': 6,'Other': 7,'Not employed': 8,

     'DBA/Database Engineer': 9,'Research Scientist': 10, 'Data Engineer':11}



#create a new column DQ5 to Q5 

datos_filtrados['DQ5']=datos_filtrados['Q5'].copy()

i=0

while (i<total_responders_Mercosur):

    #Cambio las filas sin respuestas a la respuesta 7

    if pd.isna(datos_filtrados.loc[i,'Q5']):

        datos_filtrados.loc[i,'EQ5']=7

        datos_filtrados.loc[i,'DQ5']='Other'

    else:

        datos_filtrados.loc[i,'EQ5']=rol[datos_filtrados.loc[i,'Q5']]

    i=i+1





fig=go.Figure(data=[go.Pie(labels=datos_filtrados.DQ5)])

fig.update_layout(title_text="Rol")

fig.show()
#print(datos_filtrados['Q9_Part_8'].value_counts())
activities=['Q9_Part_1','Q9_Part_2', 'Q9_Part_3','Q9_Part_4',

            'Q9_Part_5', 'Q9_Part_6','Q9_Part_7','Q9_Part_8']



lact=['Analyze and understand data to influence product or business decisions',

    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

    'Build prototypes to explore applying machine learning to new areas',

    'Build and/or run a machine learning service that operationally improves my product or workflows',

    'Experimentation and iteration to improve existing ML models',

    'Do research that advances the state of the art of machine learning',

    'None of these activities are an important part of my role at work',

    'Other']



i=0

listaact=[]

while i<len(activities):

    listaact.append(datos_filtrados[activities[i]].value_counts().sum())

    i=i+1



#plt.figure(figsize=(14,6))

y_pos=np.arange(len(activities))

plt.barh(y_pos,listaact)

plt.title("Activities more important")

plt.xlabel("Count")

plt.yticks(y_pos, lact)

plt.show()
#print(datos_filtrados['Q18_Part_11'].value_counts())
programmingl=['Q18_Part_1','Q18_Part_2', 'Q18_Part_3','Q18_Part_4',

            'Q18_Part_5', 'Q18_Part_6',

          'Q18_Part_7','Q18_Part_8', 'Q18_Part_9',

            'Q18_Part_10','Q18_Part_11','Q18_Part_12']

pl=['Python ','R','SQL','C','C++','Java','Javascript','TypeScript','Bash',

   'MATLAB','None','Other']



i=0

listapl=[]

while i<len(programmingl):

    listapl.append(datos_filtrados[programmingl[i]].value_counts().sum())

    i=i+1



plt.figure(figsize=(14,6))

y_pos=np.arange(len(programmingl))

plt.barh(y_pos,listapl, color="red")

plt.title("Programming Languages")

plt.xlabel("Count")

plt.yticks(y_pos, pl)

plt.show()
IDEs=['Q16_Part_1','Q16_Part_2', 'Q16_Part_3','Q16_Part_4',

            'Q16_Part_5', 'Q16_Part_6','Q16_Part_7','Q16_Part_8', 'Q16_Part_9',

            'Q16_Part_10','Q16_Part_11','Q16_Part_12']

lides=['Jupyter (JupyterLab, Jupyter Notebooks, etc)','RStudio','PyCharm','Atom','MATLAB','Visual Studio / Visual Studio Code',

    'Spyder','Vim / Emacs','Notepad++','Sublime Text','None','Other']



i=0

listaIDEs=[]

while i<len(IDEs):

    listaIDEs.append(datos_filtrados[IDEs[i]].value_counts().sum())

    i=i+1



plt.figure(figsize=(14,6))

y_pos=np.arange(len(IDEs))

plt.barh(y_pos,listaIDEs, color="green")

plt.title("IDEs")

plt.xlabel("Count")

plt.yticks(y_pos, lides)

plt.show()





#print(datos_filtrados['Q28_Part_10'].value_counts())
MLf=['Q28_Part_1','Q28_Part_2', 'Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8', 'Q28_Part_9',

            'Q28_Part_10','Q28_Part_11','Q28_Part_12']

lmlf=['Scikit-learn','TensorFlow','Keras','RandomForest','Xgboost','PyTorch','Caret','LightGBM','Spark MLib',

   'Fast.ai','None','Other']



i=0

listaMLf=[]

while i<len(MLf):

    listaMLf.append(datos_filtrados[MLf[i]].value_counts().sum())

    i=i+1



plt.figure(figsize=(14,6))

y_pos=np.arange(len(MLf))

plt.barh(y_pos,listaMLf)

plt.title("Machine Learning Frameworks")

plt.xlabel("Count")

plt.yticks(y_pos, lmlf)

plt.show()
#print(datos_filtrados['Q13_Part_7'].value_counts())
framework=['Q13_Part_1','Q13_Part_2', 'Q13_Part_3','Q13_Part_4',

            'Q13_Part_5', 'Q13_Part_6',

          'Q13_Part_7','Q13_Part_8', 'Q13_Part_9',

            'Q13_Part_10',

          'Q13_Part_11','Q13_Part_12']

f2=['Udacity ','Coursera','edX','DataCamp','DataQuest','Kaggle Courses (i.e. Kaggle Learn)','Fast.ai','Udemy','LinkedIn Learning',

   'University Courses (resulting in a university degree)','None','Other']



i=0

lista=[]

while i<len(framework):

    lista.append(datos_filtrados[framework[i]].value_counts().sum())

    i=i+1



plt.figure(figsize=(14,6))

y_pos=np.arange(len(framework))

plt.barh(y_pos,lista, color="yellow")

plt.title("Plataform Data Science Courses")

plt.xlabel("Count")

plt.yticks(y_pos, f2)

plt.show()
#print(datos_filtrados['Q17_Part_9'].unique())
notebook=['Q17_Part_1','Q17_Part_2', 'Q17_Part_3','Q17_Part_4',

            'Q17_Part_5', 'Q17_Part_6',

          'Q17_Part_7','Q17_Part_8',

            'Q17_Part_10','Q17_Part_11','Q17_Part_12']

ln=['Kaggle Notebooks (Kernels) ','Google Colab',' Microsoft Azure Notebooks',

    'Google Cloud Notebook Products (AI Platform, Datalab, etc)','Paperspace / Gradient',

    'FloydHub',' Binder / JupyterHub','IBM Watson Studio ',

   'AWS Notebook Products (EMR Notebooks, Sagemaker Notebooks, etc)','None','Other']



i=0

listanotebook=[]

while i<len(notebook):

    listanotebook.append(datos_filtrados[notebook[i]].value_counts().sum())

    i=i+1



plt.figure(figsize=(14,6))

y_pos=np.arange(len(notebook))

plt.barh(y_pos,listanotebook, color="brown")

plt.title("Hosted Notebooks Products")

plt.xlabel("Count")

plt.yticks(y_pos, ln)

plt.show()