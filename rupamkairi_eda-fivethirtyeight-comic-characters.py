

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%pip show plotly
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from sklearn.utils import shuffle



plt.style.use('ggplot')
dc = pd.read_csv("../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv")

marvel = pd.read_csv("../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv")
dc.sample(5)
print("All unique genders are in the data: ", dc['SEX'].unique(), "\nnan is just for missing data.\n")



sns.countplot(y=dc['SEX'])



dc_male = dc[dc['SEX'] == "Male Characters"]

print("Number of Male characters:", len(dc_male))

dc_female = dc[dc['SEX'] == "Female Characters"]

print("Number of Female characters:", len(dc_female))



print("Number of missing values:", len(dc[dc['SEX'].isna() == True]))

print("Number of Genderless characters:",len(dc[dc['SEX'] == "Genderless Characters"]))

print("Number of Transgender Characters:",len(dc[dc['SEX'] == "Transgender Characters"]))



print("Gender ratio for male to female {rM:.2f} : {rF:.2f}.".format(rM = len(dc_male)/len(dc_male), rF = len(dc_female)/len(dc_male)))
print("All unique Identity Types in the data: ", dc['ID'].unique(), "\nnan is just for missing data.\n")



sns.countplot(y=dc['ID'])


fig = px.scatter(dc, x='page_id', y='APPEARANCES', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of All characters.")

fig.show()



print("No of character they never appeared. ", len(dc[dc['APPEARANCES'] == 0]), ". Expectected, otherwise why would they exist, Right?")

print("No of character they appeared once. ", len(dc[dc['APPEARANCES'] == 1]), ". Wonder why >D?")

print("No of character they appeared twice. ", len(dc[dc['APPEARANCES'] == 2]), ". Just to see how the trend drops.")



print("No of character they appeared more than 10 times. ", len(dc[dc['APPEARANCES'] >= 10]))

print("No of character they appeared more than 100 times. ", len(dc[dc['APPEARANCES'] >= 100]))

print("No of character they appeared more than 500 times. ", len(dc[dc['APPEARANCES'] >= 500]))



print("No of characters gone out of trend in a year for monthly comics:", len(dc[dc['APPEARANCES'] <= 12]))

print("No of characters gone out of trend in a year for bi-weekly comics: ", len(dc[dc['APPEARANCES'] <= 24]))



print("No of characters appeared over than a year for monthly comics:", len(dc[dc['APPEARANCES'] > 12]))

print("No of characters appeared over than a year for bi-weekly comics:", len(dc[dc['APPEARANCES'] > 24]))



dc_pop_appear = dc[dc['APPEARANCES'] > 24]



fig = px.scatter(dc_pop_appear, x='page_id', y='APPEARANCES', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters.")

fig.show()



fig = px.scatter(dc_pop_appear, x='page_id', y='APPEARANCES', color='ALIVE', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters categorized by Alive or Deceased.")

fig.show()

dc_pop_appear_m_f = dc_pop_appear[dc_pop_appear['SEX'].isna() == False]



fig = px.scatter(dc_pop_appear_m_f, x='page_id', y='APPEARANCES', color='SEX', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters categorized by Alive or Deceased.")

fig.show()



print("Gender ratio for male to female for popular characters {rM:.2f} : {rF:.2f}.".format(

    rM = 1.00, 

    rF = len(dc_pop_appear_m_f[dc_pop_appear_m_f['SEX'] == 'Female Characters']) / len(dc_pop_appear_m_f[dc_pop_appear_m_f['SEX'] == 'Male Characters'])

))



print("Number of Male characters:", len(dc_pop_appear_m_f[dc_pop_appear_m_f['SEX']=='Male Characters']))

print("Number of Female characters:", len(dc_pop_appear_m_f[dc_pop_appear_m_f['SEX'] == 'Female Characters']))

print("\nNumber of Genderless characters:", len(dc_pop_appear_m_f[dc_pop_appear_m_f['SEX'] == 'Genderless Characters']), ", They really are something else")

print("They are: \n", dc_pop_appear_m_f[dc_pop_appear_m_f['SEX'] == 'Genderless Characters']['name'].tolist())
dc_pop_appear_g_b = dc_pop_appear[dc_pop_appear['ALIGN'].isna() == False]



fig = px.scatter(dc_pop_appear_g_b, x='page_id', y='APPEARANCES', color='ALIGN', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters categorized by Alive or Deceased.")

fig.show()



print("Ratio for good to bad for popular characters {rG:.2f} : {rB:.2f}.".format(

    rG = 1.00, 

    rB = len(dc_pop_appear_g_b[dc_pop_appear_g_b['ALIGN'] == 'Bad Characters']) / len(dc_pop_appear_g_b[dc_pop_appear_g_b['ALIGN'] == 'Good Characters'])

))



print("Number of Good characters:", len(dc_pop_appear_g_b[dc_pop_appear_g_b['ALIGN']=='Good Characters']))

print("Number of Bad characters:", len(dc_pop_appear_g_b[dc_pop_appear_g_b['ALIGN'] == 'Bad Characters']))



print("Bonus Characters")

print("Number of Neutral characters:", len(dc_pop_appear_g_b[dc_pop_appear_g_b['ALIGN']=='Neutral Characters']))

print("Number of Reformed Criminals:", len(dc_pop_appear_g_b[dc_pop_appear_g_b['ALIGN'] == 'Reformed Criminals']))

marvel.sample(5)
print("All unique genders are in the data: ", marvel['SEX'].unique(), "\nnan is just for missing data.\n")



sns.countplot(y=marvel['SEX'])



marvel_male = marvel[marvel['SEX'] == "Male Characters"]

print("Number of Male characters:", len(marvel_male))

marvel_female = marvel[marvel['SEX'] == "Female Characters"]

print("Number of Female characters:", len(marvel_female))



print("Number of missing values:", len(marvel[marvel['SEX'].isna() == True]))

print("Number of Genderfluid Characters:",len(marvel[marvel['SEX'] == "Genderfluid Characters"]))

print("Number of Agender Characters:",len(marvel[marvel['SEX'] == "Agender Characters"]))



print("Gender ratio for male to female {rM:.2f} : {rF:.2f}.".format(rM = len(marvel_male)/len(marvel_male), rF = len(marvel_female)/len(marvel_male)))
marvel[marvel['SEX'] == "Genderfluid Characters"]['name'].values
print("All unique Identity Types in the data: ", marvel['ID'].unique(), "\nnan is just for missing data.\n")



sns.countplot(y=marvel['ID'])
fig = px.scatter(marvel, x='page_id', y='APPEARANCES', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of All characters.")

fig.show()



print("No of character they appeared once: ", len(marvel[marvel['APPEARANCES'] == 1]), ". Wonder why >D?")

print("No of character they appeared twice: ", len(marvel[marvel['APPEARANCES'] == 2]), ". Just to see how the trend drops.")



print("No of character they appeared more than 10 times: ", len(marvel[marvel['APPEARANCES'] >= 10]))

print("No of character they appeared more than 100 times: ", len(marvel[marvel['APPEARANCES'] >= 100]))

print("No of character they appeared more than 500 times: ", len(marvel[marvel['APPEARANCES'] >= 500]))



print("No of characters gone out of trend in a year for monthly comics:", len(marvel[marvel['APPEARANCES'] <= 12]))

print("No of characters gone out of trend in a year for bi-weekly comics: ", len(marvel[marvel['APPEARANCES'] <= 24]))



print("No of characters appeared over than a year for monthly comics:", len(marvel[marvel['APPEARANCES'] > 12]))

print("No of characters appeared over than a year for bi-weekly comics:", len(marvel[marvel['APPEARANCES'] > 24]))



marvel_pop_appear = marvel[marvel['APPEARANCES'] > 24]


fig = px.scatter(marvel_pop_appear, x='page_id', y='APPEARANCES', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters.")

fig.show()



fig = px.scatter(marvel_pop_appear, x='page_id', y='APPEARANCES', color='ALIVE', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters categorized by Alive or Deceased.")

fig.show()

marvel_pop_appear_m_f = marvel_pop_appear[marvel_pop_appear['SEX'].isna() == False]



fig = px.scatter(marvel_pop_appear_m_f, x='page_id', y='APPEARANCES', color='SEX', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters categorized by Alive or Deceased.")

fig.show()



print("Gender ratio for male to female for popular characters {rM:.2f} : {rF:.2f}.".format(

    rM = 1.00, 

    rF = len(marvel_pop_appear_m_f[marvel_pop_appear_m_f['SEX'] == 'Female Characters']) / len(marvel_pop_appear_m_f[marvel_pop_appear_m_f['SEX'] == 'Male Characters'])

))



print("Number of Male characters:", len(marvel_pop_appear_m_f[marvel_pop_appear_m_f['SEX']=='Male Characters']))

print("Number of Female characters:", len(marvel_pop_appear_m_f[marvel_pop_appear_m_f['SEX'] == 'Female Characters']))

marvel_pop_appear_g_b = marvel_pop_appear[marvel_pop_appear['ALIGN'].isna() == False]



fig = px.scatter(marvel_pop_appear_g_b, x='page_id', y='APPEARANCES', color='ALIGN', hover_data=['name'],

                 labels = {'APPEARANCES': 'No. of Appearence', 'page_id': 'Page id(just a number)'},

                 title = "Appearance of Popular characters categorized by Alive or Deceased.")

fig.show()



print("Ratio for good to bad for popular characters {rG:.2f} : {rB:.2f}.".format(

    rG = 1.00, 

    rB = len(marvel_pop_appear_g_b[marvel_pop_appear_g_b['ALIGN'] == 'Bad Characters']) / len(marvel_pop_appear_g_b[marvel_pop_appear_g_b['ALIGN'] == 'Good Characters'])

))



print("Number of Good characters:", len(marvel_pop_appear_g_b[marvel_pop_appear_g_b['ALIGN']=='Good Characters']))

print("Number of Bad characters:", len(marvel_pop_appear_g_b[marvel_pop_appear_g_b['ALIGN'] == 'Bad Characters']))



print("Bonus Characters")

print("Number of Neutral characters:", len(marvel_pop_appear_g_b[marvel_pop_appear_g_b['ALIGN']=='Neutral Characters']))
