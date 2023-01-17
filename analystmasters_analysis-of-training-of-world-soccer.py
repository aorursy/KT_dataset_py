import pandas as pd

import plotly.plotly as py

import plotly.graph_objs as go

import numpy as np

import os 

import sys

import plotly.plotly as py

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import time

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

Mode4 = pd.read_csv('../input/analystm_mode_4_v1.csv')

MV=0.7



data_dict = dict()

mylist=[]



result=(Mode4['Results_1'])-(Mode4['Results_2'])

for index in range(Mode4.shape[0]):

    # check if majority of votes (MV%+1) is given to home or away

    if (Mode4.iloc[index,0]>Mode4.iloc[index,0:3].sum()*MV+1 or Mode4.iloc[index,2]>Mode4.iloc[index,0:3].sum()*MV+1 ) and (Mode4['Country_1'].iloc[index]==Mode4['Country_2'].iloc[index]):



        if (Mode4.iloc[index,0]>Mode4.iloc[index,2] and result[index]>0.5) or (Mode4.iloc[index,2]>Mode4.iloc[index,0] and result[index]<0.5):

            mylist.append([Mode4['Country_1'].iloc[index],1])



        else:

            mylist.append([Mode4['Country_1'].iloc[index],-1])



Data=pd.DataFrame(mylist, columns=['Country', 'Outcome'])

setData=set(Data['Country'])



stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=10,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(Data['Country']))



figsize=(20,10)

fig = plt.figure(1,figsize=(10, 8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

data_dict = dict.fromkeys(setData,0 )

data_dict_cor = dict.fromkeys(setData,0 )

data_dict_wrong = dict.fromkeys(setData,0 )



for keys in data_dict.keys():

    data_dict[keys]=list(Data['Country']==keys).count(True)



for keys in data_dict_cor.keys():

    try:

        data_dict_cor[keys]=list((Data['Country']==keys) & (Data['Outcome']==1)).count(True)

    except ValueError:

        data_dict_cor[keys]=0

    

    

for keys in data_dict_wrong.keys():

    try:

        data_dict_wrong[keys]=list((Data['Country']==keys) & (Data['Outcome']==(-1))).count(True)

    except ValueError:

        data_dict_wrong[keys]=0



sorted_by_value = sorted(data_dict.items(), key=lambda kv: kv[1],reverse=True)

b=10

ind = np.arange(b)    # the x locations for the groups

width = 0.25       # the width of the bars: can also be len(x) sequence

country_cor = [data_dict_cor[sorted_by_value[item][0]] for item in range(0,b)]

country_wrong = [data_dict_wrong[sorted_by_value[item][0]] for item in range(0,b)]

clist=[sorted_by_value[item][0] for item in range(0,b)]



fig, ax = plt.subplots(figsize=(20,10))



rects1 = ax.bar(ind, country_cor, width, color='y')

rects2 = ax.bar(ind + width, country_wrong, width, color='r')

plt.ylim(0,100)

plt.ylabel('Countries')

plt.title('Predictability')



ax.set_xticks(ind + width / 2)

ax.set_xticklabels(clist)

ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'))





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)



plt.show()
Mode3 = pd.read_csv('../input/analystm_mode_3_v1.csv')

MV=0.7



data_dict = dict()

mylist=[]

result=(Mode3['Results_1'])-(Mode3['Results_2'])

for index in range(Mode3.shape[0]):

    # check if majority of votes (MV%+1) is given to home or away

    if (Mode3.iloc[index,0]>Mode3.iloc[index,0:3].sum()*MV+1 or Mode3.iloc[index,2]>Mode3.iloc[index,0:3].sum()*MV+1 ) and (Mode3['Country_1'].iloc[index]==Mode3['Country_2'].iloc[index]):

        if (Mode3.iloc[index,0]>Mode3.iloc[index,2] and result[index]>0.5) or (Mode3.iloc[index,2]>Mode3.iloc[index,0] and result[index]<0.5):

            mylist.append([Mode3['Country_1'].iloc[index],1])

        else:

            mylist.append([Mode3['Country_1'].iloc[index],-1])



Data=pd.DataFrame(mylist, columns=['Country', 'Outcome'])

setData=set(Data['Country'])

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=10,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(Data['Country']))



figsize=(20,10)

fig = plt.figure(1,figsize=(10, 8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

data_dict = dict.fromkeys(setData,0 )

data_dict_cor = dict.fromkeys(setData,0 )

data_dict_wrong = dict.fromkeys(setData,0 )



for keys in data_dict.keys():

    data_dict[keys]=list(Data['Country']==keys).count(True)



for keys in data_dict_cor.keys():

    try:

        data_dict_cor[keys]=list((Data['Country']==keys) & (Data['Outcome']==1)).count(True)

    except ValueError:

        data_dict_cor[keys]=0

    

for keys in data_dict_wrong.keys():

    try:

        data_dict_wrong[keys]=list((Data['Country']==keys) & (Data['Outcome']==(-1))).count(True)

    except ValueError:

        data_dict_wrong[keys]=0



sorted_by_value = sorted(data_dict.items(), key=lambda kv: kv[1],reverse=True)

b=10

ind = np.arange(b)    # the x locations for the groups

width = 0.25       # the width of the bars: can also be len(x) sequence

country_cor = [data_dict_cor[sorted_by_value[item][0]] for item in range(0,b)]

country_wrong = [data_dict_wrong[sorted_by_value[item][0]] for item in range(0,b)]

clist=[sorted_by_value[item][0] for item in range(0,b)]



fig, ax = plt.subplots(figsize=(20,10))



rects1 = ax.bar(ind, country_cor, width, color='y')

rects2 = ax.bar(ind + width, country_wrong, width, color='r')

plt.ylim(0,50)

plt.ylabel('Countries')

plt.title('Predictability')



ax.set_xticks(ind + width / 2)

ax.set_xticklabels(clist)



ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'))



autolabel(rects1)

autolabel(rects2)



plt.show()
Mode2 = pd.read_csv('../input/analystm_mode_2_v1.csv')

MV=0.7



data_dict = dict()

mylist=[]



result=(Mode2['Results_1'])-(Mode2['Results_2'])

for index in range(Mode2.shape[0]):

    # check if majority of votes (MV%+1) is given to home or away

    if (Mode2.iloc[index,0]>Mode2.iloc[index,0:3].sum()*MV+1 or Mode2.iloc[index,2]>Mode2.iloc[index,0:3].sum()*MV+1 ) and (Mode2['Country_1'].iloc[index]==Mode2['Country_2'].iloc[index]):



        if (Mode2.iloc[index,0]>Mode2.iloc[index,2] and result[index]>0.5) or (Mode2.iloc[index,2]>Mode2.iloc[index,0] and result[index]<0.5):

            mylist.append([Mode2['Country_1'].iloc[index],1])

            

        else:

            mylist.append([Mode2['Country_1'].iloc[index],-1])

            

Data=pd.DataFrame(mylist, columns=['Country', 'Outcome'])

setData=set(Data['Country'])



stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=10,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(Data['Country']))



figsize=(20,10)

fig = plt.figure(1,figsize=(10, 8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

data_dict = dict.fromkeys(setData,0 )

data_dict_cor = dict.fromkeys(setData,0 )

data_dict_wrong = dict.fromkeys(setData,0 )



for keys in data_dict.keys():

    data_dict[keys]=list(Data['Country']==keys).count(True)



for keys in data_dict_cor.keys():

    try:

        data_dict_cor[keys]=list((Data['Country']==keys) & (Data['Outcome']==1)).count(True)

    except ValueError:

        data_dict_cor[keys]=0

        

for keys in data_dict_wrong.keys():

    try:

        data_dict_wrong[keys]=list((Data['Country']==keys) & (Data['Outcome']==(-1))).count(True)

    except ValueError:

        data_dict_wrong[keys]=0



sorted_by_value = sorted(data_dict.items(), key=lambda kv: kv[1],reverse=True)

b=10

ind = np.arange(b)    

width = 0.25      

country_cor = [data_dict_cor[sorted_by_value[item][0]] for item in range(0,b)]

country_wrong = [data_dict_wrong[sorted_by_value[item][0]] for item in range(0,b)]

clist=[sorted_by_value[item][0] for item in range(0,b)]



fig, ax = plt.subplots(figsize=(20,10))



rects1 = ax.bar(ind, country_cor, width, color='y')

rects2 = ax.bar(ind + width, country_wrong, width, color='r')

plt.ylim(0,50)

plt.ylabel('Countries')

plt.title('Predictability')



ax.set_xticks(ind + width / 2)

ax.set_xticklabels(clist)

ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'))

       

autolabel(rects1)

autolabel(rects2)

plt.show()
Mode1 = pd.read_csv('../input/analystm_mode_1_v1.csv')

MV=0.7



data_dict = dict()

mylist=[]

result=(Mode1['Results_1'])-(Mode1['Results_2'])

for index in range(Mode1.shape[0]):

    # check if majority of votes (MV%+1) is given to home or away

    if (Mode1.iloc[index,0]>Mode1.iloc[index,0:3].sum()*MV+1 or Mode1.iloc[index,2]>Mode1.iloc[index,0:3].sum()*MV+1 ) and (Mode1['Country_1'].iloc[index]==Mode1['Country_2'].iloc[index]):



        if (Mode1.iloc[index,0]>Mode1.iloc[index,2] and result[index]>0.5) or (Mode1.iloc[index,2]>Mode1.iloc[index,0] and result[index]<0.5):

            mylist.append([Mode1['Country_1'].iloc[index],1])

        else:

            mylist.append([Mode1['Country_1'].iloc[index],-1])



Data=pd.DataFrame(mylist, columns=['Country', 'Outcome'])

setData=set(Data['Country'])



stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=10,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(Data['Country']))



figsize=(20,10)

fig = plt.figure(1,figsize=(10, 8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

data_dict = dict.fromkeys(setData,0 )

data_dict_cor = dict.fromkeys(setData,0 )

data_dict_wrong = dict.fromkeys(setData,0 )



for keys in data_dict.keys():

    data_dict[keys]=list(Data['Country']==keys).count(True)



for keys in data_dict_cor.keys():

    try:

        data_dict_cor[keys]=list((Data['Country']==keys) & (Data['Outcome']==1)).count(True)

    except ValueError:

        data_dict_cor[keys]=0

        

for keys in data_dict_wrong.keys():

    try:

        data_dict_wrong[keys]=list((Data['Country']==keys) & (Data['Outcome']==(-1))).count(True)

    except ValueError:

        data_dict_wrong[keys]=0



sorted_by_value = sorted(data_dict.items(), key=lambda kv: kv[1],reverse=True)

b=10

ind = np.arange(b)    

width = 0.25      

country_cor = [data_dict_cor[sorted_by_value[item][0]] for item in range(0,b)]

country_wrong = [data_dict_wrong[sorted_by_value[item][0]] for item in range(0,b)]

clist=[sorted_by_value[item][0] for item in range(0,b)]



fig, ax = plt.subplots(figsize=(20,10))



rects1 = ax.bar(ind, country_cor, width, color='y')

rects2 = ax.bar(ind + width, country_wrong, width, color='r')

plt.ylim(0,150)

plt.ylabel('Countries')

plt.title('Predictability')



ax.set_xticks(ind + width / 2)

ax.set_xticklabels(clist)

ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'))

       

autolabel(rects1)

autolabel(rects2)

plt.show()