import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np 

import warnings

warnings.filterwarnings("ignore")



import os

mcr= pd.read_csv("../input/multipleChoiceResponses.csv")

mcr.head()
mcr.Q1[1:].value_counts().plot(kind='bar',title='What is your gender?\n',figsize=(10,5)) #returns matplotlib object

plt.xticks(rotation=30) #use matplotlib functions

plt.ylabel("Number of respondents")

plt.show()
labels = '18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+'

sizes = [3037,5154,6159,3776,2253,1360,858,582,328,273,53,39]

plt.bar(height=sizes,x=labels) 

plt.xticks(rotation=40)

plt.xlabel("Age range (years)")

plt.ylabel("Number of respondents")

plt.title('What is your age (# years)?\n\n')

plt.show()
mcr.Q3[1:].value_counts().plot(kind='bar',figsize=(20,10),title='In which country do you currently reside?\n')

plt.ylabel("Number of respondents")

plt.show()
mcr.Q4[1:].value_counts().plot(kind='pie',figsize=(8,10),autopct='%2.1f%%',title='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?\n')

plt.ylabel(" ")



plt.show()
labels = 'Computer science (software engineering, etc.)',\

'Engineering (non-computer focused)',\

'Mathematics or statistics',\

'A business discipline (accounting, economics, finance, etc.)',\

'Physics or astronomy',\

'Information technology, networking, or system administration',\

'Medical or life sciences (biology, chemistry, medicine, etc.)',\

'Other'

sizes = [9430, 3705, 2950, 1791,1110,1029,871,2061]

explode = (0.2,0,0,0,0,0,0,0)  # explode 1st slice

 

plt.pie(sizes, explode=explode,autopct='%1.1f%%', startangle=140) #labels=labels,rotatelabels = True

#plt.axis('equal')

plt.title('Which best describes your undergraduate major?\n')

plt.legend(labels=labels,loc='right',bbox_to_anchor=(2, 1, 1, -1))

plt.show()
mcr.Q8[1:].value_counts().plot(kind='bar',title='How many years of experience do you have in your current role?\n',figsize=(10,5)) #returns matplotlib object

plt.xticks(rotation=330) #use matplotlib functions

plt.ylabel("Number of respondents")

plt.xlabel("Experience (in years)")

plt.show()
df1 = mcr.loc[1:, 'Q11_Part_1':'Q11_Part_7']  #column slicing

df1 = df1.melt(var_name='Question',value_name='important_role')

df1 = df1.dropna()

df1=df1.groupby(['Question','important_role']).size().reset_index(name="Number of respondents")

df1.set_index('important_role',drop=True,inplace=True)

ax=df1.sort_values(by="Number of respondents",ascending=False).plot(kind='barh',figsize=(5,5),\

                                                  title="Select any activities that make up an important part of your role at work\n\n")



plt.ylabel(" ")

plt.show()
mcr.Q12_MULTIPLE_CHOICE[1:].value_counts().plot(kind='barh',title='What is the primary tool that you use at work or school to analyze data? \n',figsize=(10,5)) #returns matplotlib object

plt.xticks(rotation=330) #use matplotlib functions

plt.xlabel("Number of respondents")

plt.show()
df1 = mcr.loc[1:, 'Q13_Part_1':'Q13_Part_15']  #column slicing

df1 = df1.melt(var_name='Question',value_name='IDE')

df1 = df1.dropna()

df1=df1.groupby(['Question','IDE']).size().reset_index(name='Number of respondents')

df1.set_index("IDE",drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='barh',figsize=(5,5),\

                                                  title="Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years?\n\n")

                                            



plt.ylabel("")

plt.show()
df1 = mcr.loc[1:, 'Q14_Part_1':'Q14_Part_11']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Hosted notebooks')

df1 = df1.dropna()



df1=df1.groupby(['Question','Hosted notebooks']).size().reset_index(name='Number of respondents')

df1.set_index("Hosted notebooks",drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(5,5),\

                                                  title="Which of the following hosted notebooks have you used at work or school in the last 5 years?\n\n",\

                                                  colors = 'lightgreen')



plt.xticks(rotation=90)

plt.xlabel("")

plt.show()
df1 = mcr.loc[1:, 'Q15_Part_1':'Q15_Part_7']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Cloud computing services')

df1 = df1.dropna()



df1=df1.groupby(['Question','Cloud computing services']).size().reset_index(name='Number of respondents')

df1.set_index('Cloud computing services',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(5,5),\

                                                  title="Which of the following cloud computing services have you used at work or school in the last 5 years?\n\n",\

                                                  colors = 'brown')



plt.xticks(rotation=90)

plt.xlabel("")

plt.show()
df1 = mcr.loc[1:, 'Q16_Part_1':'Q16_Part_18']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Programming languages')

df1 = df1.dropna()



df1=df1.groupby(['Question','Programming languages']).size().reset_index(name='Number of respondents')

df1.set_index('Programming languages',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(8,6),\

                                                  title="What programming languages do you use on a regular basis?\n\n",\

                                                  color='darkcyan')



plt.xticks(rotation=75)

plt.xlabel("")

plt.show()
mcr.Q17[1:].value_counts().plot(kind='bar',figsize=(8,6),title="What specific programming language do you use most often? \n\n")

plt.xticks(rotation=75)

plt.ylabel("Number of respondents")

plt.show()
mcr.Q18[1:].value_counts().plot(kind='bar',figsize=(8,6),title="What programming language would you recommend an aspiring data scientist to learn first? \n\n")

plt.xticks(rotation=75)

plt.ylabel("Number of respondents")

plt.show()
df1 = mcr.loc[1:, 'Q19_Part_1':'Q19_Part_19']  #column slicing

df1 = df1.melt(var_name='Question',value_name='ML frameworks')

df1 = df1.dropna()



df1=df1.groupby(['Question','ML frameworks']).size().reset_index(name='Number of respondents')

df1.set_index('ML frameworks',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(8,6),\

                                                  title="What machine learning frameworks have you used in the past 5 years? \n\n",\

                                                  color='cyan')



plt.xticks(rotation=75)

plt.xlabel("")

plt.show()
mcr.Q20[1:].value_counts().plot(kind='bar',figsize=(8,6),title="Which ML library have you used the most?  \n\n")

plt.xticks(rotation=75)

plt.ylabel("Number of respondents")

plt.show()
df1 = mcr.loc[1:, 'Q21_Part_1':'Q21_Part_13']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Visualization libraries')

df1 = df1.dropna()



df1=df1.groupby(['Question','Visualization libraries']).size().reset_index(name='Number of respondents')

df1.set_index('Visualization libraries',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(8,6),\

                                                  title="What data visualization libraries or tools have you used in the past 5 years?\n\n",\

                                                  color='darkcyan')



plt.xticks(rotation=75)

plt.xlabel("")

plt.show()
mcr.Q22[1:].value_counts().plot(kind='bar',figsize=(8,6),title="Which specific data visualization library or tool have you used the most?\n\n")

plt.xticks(rotation=75)

plt.ylabel("Number of respondents")

plt.show()
mcr.Q23[1:].value_counts().plot(kind='bar',figsize=(10,8),title="Approximately what percent of your time at work or school is spent actively coding?\n\n")

plt.xticks(rotation=90)

plt.ylabel("Number of respondents")

plt.show()
mcr.Q25[1:].value_counts().plot(kind='barh',figsize=(10,5),title="For how many years have you used machine learning methods (at work or in school)?\n\n")

plt.xticks(rotation=60)

plt.xlabel("Number of respondents")

plt.show()
df1 = mcr.loc[1:, 'Q27_Part_1':'Q27_Part_20']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Cloud computing products')

df1 = df1.dropna()



df1=df1.groupby(['Question','Cloud computing products']).size().reset_index(name='Number of respondents')

df1.set_index('Cloud computing products',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='barh',figsize=(8,6),\

                                                  title="Which of the following cloud computing products have you used at work or school in the last 5 years?\n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=50)

plt.show()
df1 = mcr.loc[1:, 'Q29_Part_1':'Q29_Part_28']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Database products')

df1 = df1.dropna()



df1=df1.groupby(['Question','Database products']).size().reset_index(name='Number of respondents')

df1.set_index('Database products',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),\

                                                  title="Which of the following relational database products have you used at work or school in the last 5 years? \n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=90)

plt.xlabel("")

plt.show()
df1 = mcr.loc[1:, 'Q30_Part_1':'Q30_Part_25']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Big data and analytics products')

df1 = df1.dropna()



df1=df1.groupby(['Question','Big data and analytics products']).size().reset_index(name='Number of respondents')

df1.set_index('Big data and analytics products',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),\

                                                  title="Which of the following big data and analytics products have you used at work or school in the last 5 years?\n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=90)

plt.xlabel("")

plt.show()
df1 = mcr.loc[1:, 'Q31_Part_1':'Q31_Part_12']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Types of data')

df1 = df1.dropna()



df1=df1.groupby(['Question','Types of data']).size().reset_index(name='Number of respondents')

df1.set_index('Types of data',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),\

                                                  title="Which types of data do you currently interact with most often at work or school?\n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=60)

plt.xlabel("")

plt.show()
df1 = mcr.loc[1:, 'Q33_Part_1':'Q33_Part_11']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Datasets')

df1 = df1.dropna()



df1=df1.groupby(['Question','Datasets']).size().reset_index(name='Number of respondents')

df1.set_index('Datasets',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='barh',figsize=(12,6),\

                                                  title="Where do you find public datasets? \n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=50)

plt.ylabel("")

plt.show()
df1 = mcr.loc[1:, 'Q34_Part_1':'Q34_Part_6']

df1=df1.dropna()



#df1=df1.reset_index()

df1=df1.rename(columns={'Q34_Part_1':'Gathering','Q34_Part_2':'Cleaning','Q34_Part_3':'Visualizing','Q34_Part_4':'Model','Q34_Part_5':'Putting the model into production','Q34_Part_6':'Finding'})

df1=df1.astype(float)  #V.V.I. step

ax=df1.plot(kind='box',figsize=(12,8),title="During a typical data science project at work or school, approximately what proportion of your time is devoted to the following?\n\n")

ax.set_ylabel("Percentage")

plt.xticks(rotation=55)

plt.show()
df1 = mcr.loc[1:, 'Q35_Part_1':'Q35_Part_6']

df1=df1.dropna()



#df1=df1.reset_index()

df1=df1.rename(columns={'Q35_Part_1':'Self-taught','Q35_Part_2':'Online courses (Coursera, Udemy, edX, etc.)','Q35_Part_3':'Work','Q35_Part_4':'University','Q35_Part_5':'Kaggle competitions','Q35_Part_6':'Other'})



df1=df1.astype(float)  #V.V.I. step

ax=df1.plot(kind='box',figsize=(12,10),title="What percentage of your current machine learning/data science training falls under each category?\n\n")

ax.set_ylabel("Percentage")

plt.xticks(rotation=55)

plt.show()
df1 = mcr.loc[1:, 'Q36_Part_1':'Q36_Part_13']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Online platforms')

df1 = df1.dropna()



df1=df1.groupby(['Question','Online platforms']).size().reset_index(name='Number of respondents')

df1.set_index('Online platforms',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),\

                                                  title="On which online platforms have you begun or completed data science courses?\n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=60)

plt.xlabel("")

plt.show()
mcr.Q37[1:].value_counts().plot(kind='bar',figsize=(8,6),title="On which online platform have you spent the most amount of time?\n\n")

plt.xticks(rotation=75)

plt.ylabel("Number of respondents")

plt.show()
df1 = mcr.loc[1:, 'Q38_Part_1':'Q38_Part_22']  #column slicing

df1 = df1.melt(var_name='Question',value_name='Favorite media sources')

df1 = df1.dropna()



df1=df1.groupby(['Question','Favorite media sources']).size().reset_index(name='Number of respondents')

df1.set_index('Favorite media sources',drop=True,inplace=True)

df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),\

                                                  title="Who/what are your favorite media sources that report on data science topics? \n\n",\

                                                  color='darkmagenta')



plt.xticks(rotation=70)

plt.xlabel("")

plt.show()
mcr.Q39_Part_1[1:].value_counts().plot(kind='bar',figsize=(8,6),title="How do you perceive the quality of online learning platforms and MOOCs as compared to the quality of the education provided by traditional brick and mortar institutions?\n\n")

plt.xticks(rotation=60)

plt.ylabel("Number of respondents")

plt.show()
mcr.Q39_Part_2[1:].value_counts().plot(kind='bar',figsize=(8,6),title="How do you perceive the quality of In-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions?\n\n")

plt.xticks(rotation=60)

plt.ylabel("Number of respondents")

plt.show()
labels = 'Independent projects are much more \nimportant than academic achievements',\

'Independent projects are slightly more \nimportant than academic achievements',\

'Independent projects are equally \nimportant as academic achievements',\

'No opinion; I do not know',\

'Independent projects are slightly less important than academic achievements',\

'Independent projects are much less important than academic achievements'

sizes = [4990,4473,4343,936,831,306]

explode = (0.2,0,0,0,0,0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True)

plt.title("Which better demonstrates expertise in data science: academic achievements or independent projects?\n\n")

#plt.savefig('output.png', dpi=300, bbox_inches='tight')

plt.show()
labels = 'We are exploring ML methods \n (and may one day put a model into production)',\

'No (we do not use ML methods)',\

'We recently started using ML methods \n (i.e., models in production for less than 2 years)',\

'I do not know',\

'We have well established ML methods \n (i.e., models in production for more than 2 years)',\

'We use ML methods for generating insights \n (but do not put working models into production)'

sizes = [4688,4411,3790,2893,2782,2105]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red', 'green']

explode = (0.2,0,0,0,0,0)  # explode 1st slice



plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140,rotatelabels = False)

plt.title("Does your current employer incorporate machine learning methods into their business?\n\n")

plt.show()