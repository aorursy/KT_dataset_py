import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import base64

import io

# from scipy.misc import imread

import codecs

from IPython.display import HTML

# from matplotlib_venn import venn2

# from subprocess import check_output
PATH = '/kaggle/input/kaggle-survey-2019/'

multiple_choice = pd.read_csv(PATH + 'multiple_choice_responses.csv', header=1)

questions = pd.read_csv(PATH + 'questions_only.csv')

survey_schema = pd.read_csv(PATH + 'survey_schema.csv',  header=1)

other_responses = pd.read_csv(PATH + 'other_text_responses.csv', header=1)
data=questions.transpose().values

for x in range(len(data)):

    print(x,data[x])
df_19=multiple_choice

df_19.head()
#How many people participated?

len(df_19)
# #How many feature we have ?

# df_19.columns
# show head of the file 

df_19.head()
# col=df_19.columns

# for x in range(len(col)):

#     print(x,col[x])
 

data_filter19=df_19[['What is your age (# years)?',

                     'What is your gender? - Selected Choice',

                     'In which country do you currently reside?',

                     'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',

                     'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',

                     'What is your current yearly compensation (approximate $USD)?',

                     'Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?',

                     'How long have you been writing code to analyze data (at work or at school)?',

                     'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice']].drop(0)#

# #Replace title

col_replace={

             'What is your age (# years)?':'age',

             'What is your gender? - Selected Choice':'gender',

             'In which country do you currently reside?':'country',

             'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'FormalEducation',

             'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'CurrentJobTitleSelect',

             'What is your current yearly compensation (approximate $USD)?':'CompensationAmount',

             'Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?':"money spent ML" ,  

             'How long have you been writing code to analyze data (at work or at school)?':'Tenure',

             'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice':'LanguageRecommendationSelect',

             

                

            }





data_filter19=data_filter19.rename(columns=col_replace)
# FormalEducation

f,ax=plt.subplots(1,2,figsize=(25,12))

indexes=data_filter19['FormalEducation'].value_counts().index

sns.countplot(y=data_filter19['FormalEducation'],order=indexes,ax=ax[0])

sns.set(font_scale=3) 

ax[0].set_title('FormalEducation',size=40)

ax[0].set_xlabel('')









data_filter19["FormalEducation"].value_counts().plot.pie(autopct='%3.0f%%',fontsize=25,colors=sns.color_palette("Paired",10),ax=ax[1])

ax[1].set_title('FormalEducation',size=40)

my_circle=plt.Circle( (0,0), 0.3, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()

data_filter19['FormalEducation'].value_counts()
f,ax=plt.subplots(1,2,figsize=(25,12))

indexes=data_filter19['CurrentJobTitleSelect'].value_counts().index

sns.countplot(y=data_filter19['CurrentJobTitleSelect'],order=indexes,ax=ax[0])

sns.set(font_scale=3) 

ax[0].set_title('CurrentJobTitleSelect',size=40)

ax[0].set_xlabel('')









data_filter19["CurrentJobTitleSelect"].value_counts().plot.pie(autopct='%3.0f%%',fontsize=25,colors=sns.color_palette("Paired",10),ax=ax[1])

ax[1].set_title('CurrentJobTitleSelect',size=40)

my_circle=plt.Circle( (0,0), 0.3, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()

data_filter19['CurrentJobTitleSelect'].value_counts()
# CompensationAmount

f,ax=plt.subplots(1,2,figsize=(25,12))

indexes=data_filter19['CompensationAmount'].value_counts().index

sns.countplot(y=data_filter19['CompensationAmount'],order=indexes,ax=ax[0])

sns.set(font_scale=3) 

ax[0].set_title('CompensationAmount',size=40)

ax[0].set_xlabel('')









data_filter19["CompensationAmount"].value_counts().plot.pie(autopct='%3.0f%%',fontsize=25,colors=sns.color_palette("Paired",10),ax=ax[1])

ax[1].set_title('CompensationAmount',size=40)

my_circle=plt.Circle( (0,0), 0.3, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()

data_filter19['CompensationAmount'].value_counts()
#Experiance 

f,ax=plt.subplots(1,2,figsize=(25,12))

indexes=data_filter19['Tenure'].value_counts().index

sns.countplot(y=data_filter19['Tenure'],order=indexes,ax=ax[0])

sns.set(font_scale=3) 

ax[0].set_title('Tenure',size=40)

ax[0].set_xlabel('')









data_filter19["Tenure"].value_counts().plot.pie(autopct='%3.0f%%',fontsize=25,colors=sns.color_palette("Paired",10),ax=ax[1])

ax[1].set_title('Tenure',size=40)

my_circle=plt.Circle( (0,0), 0.3, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()

data_filter19['Tenure'].value_counts()
#i splite "FormalEducation" into two catagory . degree vs no_degree

#Using groupyby and filter helps me to filter the data filter out degrees participant aside and other aside

degree=data_filter19.groupby(['CompensationAmount','FormalEducation','CurrentJobTitleSelect']).filter(lambda x: ((x['FormalEducation']=='Master’s degree' ).any() or (x['FormalEducation']=='Bachelor’s degree').any() or x['FormalEducation']=='Doctoral degree' ).any()).reset_index()





no_degree=data_filter19.groupby(['CompensationAmount','FormalEducation','CurrentJobTitleSelect']).filter(lambda x: ((x['FormalEducation']=='Some college/university study without earning a bachelor’s degree' ).any() or (x['FormalEducation']=='Professional degree').any()  or x['FormalEducation']=='No formal education past high school' ).any()).reset_index()





"WIth degree",len(degree),"No degree",len(no_degree)
# CompensationAmount

f,ax=plt.subplots(figsize=(40,20))

indexes=data_filter19['FormalEducation'].value_counts().index

sns.countplot(y=data_filter19['FormalEducation'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('FormalEducation',size=40)

ax.set_xlabel('')



data_filter19['FormalEducation'].value_counts()






f,ax=plt.subplots(figsize=(10,10))

indexes=degree['FormalEducation'].value_counts().index

sns.countplot(y=degree['FormalEducation'],order=indexes,ax=ax)

sns.set(font_scale=2) 

ax.set_title('Degree ',size=20)

ax.set_xlabel('')
f,ax=plt.subplots(figsize=(25,12))

indexes=no_degree['FormalEducation'].value_counts().index

sns.countplot(y=no_degree['FormalEducation'],order=indexes,ax=ax)

sns.set(font_scale=4) 

ax.set_title('No Degree',size=50)

ax.set_xlabel('')
degree['CurrentJobTitleSelect'].value_counts()
# CompensationAmount

f,ax=plt.subplots(figsize=(40,10))

indexes=degree['CurrentJobTitleSelect'].value_counts().index

sns.countplot(y=degree['CurrentJobTitleSelect'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('Jobs Filter by by Degrees',size=40)

ax.set_xlabel('')



degree['CurrentJobTitleSelect'].value_counts()
# CompensationAmount

f,ax=plt.subplots(figsize=(40,10))

indexes=no_degree['CurrentJobTitleSelect'].value_counts().index

sns.countplot(y=no_degree['CurrentJobTitleSelect'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('Jobs Filter by Non Degrees',size=40)

ax.set_xlabel('')



no_degree['CurrentJobTitleSelect'].value_counts()
# temper_data=data_filter19.groupby(['CompensationAmount','FormalEducation','Tenure']).filter(lambda x: ((( x['FormalEducation']=='No formal education past high school').any())       and (x['Tenure']=='< 1 years').any())).reset_index()



                                        #Data scientist vs student

f,ax=plt.subplots(1,2,figsize=(20,24))

sns.set(font_scale=1.5)

r_vs_py=degree.groupby(['FormalEducation','CurrentJobTitleSelect'])['gender'].count().reset_index()

r_vs_py

r_vs_py.pivot('FormalEducation','CurrentJobTitleSelect','gender').plot.barh(width=0.3,ax=ax[0])

ax[0].set_title('Dataset filter by Degree')

ax[0].set_ylabel(" ")

ax[0].set_xlabel("People ")

plt.subplots_adjust(wspace=0.5)









r_vs_py=degree.groupby(['CurrentJobTitleSelect','FormalEducation'])['gender'].count().reset_index()

r_vs_py

r_vs_py.pivot('CurrentJobTitleSelect','FormalEducation','gender').plot.barh(width=0.5,ax=ax[1])

ax[1].set_title('Dataset filter by Degree')

ax[1].set_ylabel(" ")

ax[1].set_xlabel("People ")

plt.subplots_adjust(wspace=0.5)

# temper_data
# temper_data=data_filter19.groupby(['CompensationAmount','FormalEducation','Tenure']).filter(lambda x: ((( x['FormalEducation']=='No formal education past high school').any())       and (x['Tenure']=='< 1 years').any())).reset_index()



                                        #Data scientist vs student

f,ax=plt.subplots(1,2,figsize=(20,24))

sns.set(font_scale=1.5)

r_vs_py=no_degree.groupby(['FormalEducation','CurrentJobTitleSelect'])['gender'].count().reset_index()

r_vs_py

r_vs_py.pivot('FormalEducation','CurrentJobTitleSelect','gender').plot.barh(width=0.3,ax=ax[0])

ax[0].set_title('Dataset filter by Degree')

ax[0].set_ylabel(" ")

ax[0].set_xlabel("People ")

plt.subplots_adjust(wspace=0.5)









r_vs_py=no_degree.groupby(['CurrentJobTitleSelect','FormalEducation'])['gender'].count().reset_index()

r_vs_py

r_vs_py.pivot('CurrentJobTitleSelect','FormalEducation','gender').plot.barh(width=0.5,ax=ax[1])

ax[1].set_title('Dataset filter by Degree')

ax[1].set_ylabel(" ")

ax[1].set_xlabel("People ")

plt.subplots_adjust(wspace=0.5)

# temper_data
# CompensationAmount

f,ax=plt.subplots(figsize=(40,24))

indexes=data_filter19['CompensationAmount'].value_counts().index

sns.countplot(y=data_filter19['CompensationAmount'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('CompensationAmount',size=40)

ax.set_xlabel('')



data_filter19['CompensationAmount'].value_counts()[:5]
# CompensationAmount

f,ax=plt.subplots(figsize=(40,20))

indexes=degree['CompensationAmount'].value_counts().index

sns.countplot(y=degree['CompensationAmount'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('CompensationAmount',size=40)

ax.set_xlabel('')



degree['CompensationAmount'].value_counts()[:5]
# CompensationAmount

f,ax=plt.subplots(figsize=(40,20))

indexes=no_degree['CompensationAmount'].value_counts().index

sns.countplot(y=no_degree['CompensationAmount'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('No Degree CompensationAmount',size=40)

ax.set_xlabel('')



no_degree['CompensationAmount'].value_counts()[:5]
f,ax=plt.subplots(figsize=(20,20))

r_vs_py=degree.groupby(['CompensationAmount','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



r_vs_py

r_vs_py.pivot('CompensationAmount','FormalEducation','gender').plot.barh(width=0.5,ax=ax)

ax.fig=plt.gcf()

ax.set_title('Degrees holder Employees  ')

ax.set_ylabel(" ")

# plt.subplots_adjust(wspace=0.5)

plt.show()

r_vs_py[:10]




f,ax=plt.subplots(figsize=(20,20))





r_vs_py=no_degree.groupby(['CompensationAmount','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



r_vs_py

r_vs_py.pivot('CompensationAmount','FormalEducation','gender').plot.barh(width=0.5,ax=ax)

ax.fig=plt.gcf()

ax.set_title('Degrees holder Employees  ')

ax.set_ylabel(" ")

# plt.subplots_adjust(wspace=0.5)

plt.show()

r_vs_py[:10]
# CompensationAmount

f,ax=plt.subplots(figsize=(40,24))

indexes=data_filter19['Tenure'].value_counts().index

sns.countplot(y=data_filter19['Tenure'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('Tenure',size=40)

ax.set_xlabel('')



data_filter19['Tenure'].value_counts()[:5]
# CompensationAmount

f,ax=plt.subplots(figsize=(40,24))

indexes=degree['Tenure'].value_counts().index

sns.countplot(y=degree['Tenure'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('Tenure',size=40)

ax.set_xlabel('')



degree['Tenure'].value_counts()
# CompensationAmount

f,ax=plt.subplots(figsize=(40,20))

indexes=no_degree['Tenure'].value_counts().index

sns.countplot(y=no_degree['Tenure'],order=indexes,ax=ax)

sns.set(font_scale=3) 



ax.set_title('Tenure',size=40)

ax.set_xlabel('')



no_degree['Tenure'].value_counts()
f,ax=plt.subplots(figsize=(20,20))

r_vs_py=degree.groupby(['Tenure','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



r_vs_py

r_vs_py.pivot('Tenure','FormalEducation','gender').plot.barh(width=0.5,ax=ax)

ax.fig=plt.gcf()

ax.set_title('Degrees holder Employees  ')

ax.set_ylabel(" ")

# plt.subplots_adjust(wspace=0.5)

plt.show()

r_vs_py[:10]
f,ax=plt.subplots(figsize=(20,24))

r_vs_py=no_degree.groupby(['Tenure','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



r_vs_py

r_vs_py.pivot('Tenure','FormalEducation','gender').plot.barh(width=0.5,ax=ax)

ax.fig=plt.gcf()

ax.set_title('Degrees holder Employees  ')

ax.set_ylabel(" ")

# plt.subplots_adjust(wspace=0.5)

plt.show()

r_vs_py[:10]
degree_vd_exper=data_filter19.groupby(['CompensationAmount','FormalEducation','CurrentJobTitleSelect','Tenure']).filter(lambda x: (((x['FormalEducation']=='Master’s degree' ).any() or (x['FormalEducation']=='Bachelor’s degree').any() or x['FormalEducation']=='Doctoral degree' ).any()) and (x['Tenure']=='< 1 years').any()) .reset_index()



no_degree_vd_exper=data_filter19.groupby(['CompensationAmount','FormalEducation','CurrentJobTitleSelect',"Tenure"]).filter(lambda x: ((x['FormalEducation']=='Some college/university study without earning a bachelor’s degree' ).any() or (x['FormalEducation']=='Professional degree').any() or x['FormalEducation']=='No formal education past high school' ).any() and (x['Tenure']=='< 1 years').any()).reset_index()
f,ax=plt.subplots(1,2,figsize=(20,20))

sns.set(font_scale=2) 

r_vs_py=degree_vd_exper.groupby(['CompensationAmount','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



r_vs_py

r_vs_py.pivot('CompensationAmount','FormalEducation','gender').plot.barh(width=0.5,ax=ax[0])

ax[0].fig=plt.gcf()

ax[0].set_title('employes with "Degree" of  1< year Experiance ')

ax[0].set_ylabel(" ")

temp_data=no_degree_vd_exper.groupby(['CompensationAmount','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



temp_data

temp_data.pivot('CompensationAmount','FormalEducation','gender').plot.barh(width=0.5,ax=ax[1])

ax[1].fig=plt.gcf()

ax[1].set_title('employes with " No Degree" of  1< year Experiance ')

ax[1].set_ylabel(" ")

plt.subplots_adjust(wspace=0.5)

plt.show()

temp_data[:10]
degree_vd_exper_10_20yr=data_filter19.groupby(['CompensationAmount','FormalEducation','CurrentJobTitleSelect','Tenure']).filter(lambda x: (((x['FormalEducation']=='Master’s degree' ).any() or (x['FormalEducation']=='Bachelor’s degree').any() or x['FormalEducation']=='Doctoral degree' ).any()) and (x['Tenure']=='10-20 years').any()) .reset_index()



no_degree_vd_exper_10_20yr=data_filter19.groupby(['CompensationAmount','FormalEducation','CurrentJobTitleSelect',"Tenure"]).filter(lambda x: ((x['FormalEducation']=='Some college/university study without earning a bachelor’s degree' ).any() or (x['FormalEducation']=='Professional degree').any() or x['FormalEducation']=='No formal education past high school' ).any() and (x['Tenure']=='10-20 years').any()).reset_index()
f,ax=plt.subplots(1,2,figsize=(20,20))

r_vs_py=degree_vd_exper_10_20yr.groupby(['CompensationAmount','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



r_vs_py

r_vs_py.pivot('CompensationAmount','FormalEducation','gender').plot.barh(width=0.5,ax=ax[0])

ax[0].fig=plt.gcf()

ax[0].set_title('employes with "Degree" of  "10-20 years" Experiance ')

ax[0].set_ylabel(" ")

temp_data=no_degree_vd_exper_10_20yr.groupby(['CompensationAmount','FormalEducation'])['gender'].count().sort_values(ascending=False).reset_index()



temp_data

temp_data.pivot('CompensationAmount','FormalEducation','gender').plot.barh(width=0.5,ax=ax[1])

ax[1].fig=plt.gcf()

ax[1].set_title('employes with " No Degree" of  "10-20 years" Experiance ')

ax[1].set_ylabel(" ")

plt.subplots_adjust(wspace=0.5)

plt.show()

r_vs_py[:10]