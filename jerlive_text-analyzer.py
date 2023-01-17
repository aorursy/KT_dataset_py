# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print("Export Chats from WhatsApp and add the file by Adding Data to this kernel.\n\nFiles Available :\n")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import pandas as pd

import numpy as np

import regex

import emoji

import re

import matplotlib.pyplot as plotter

print("\n\nCopy the required directory, paste it into the input box above and then press Enter. ")

expfile=input("PASTE HERE >>>")

file = open(expfile,mode='r',encoding="utf8")

data = file.read()

file.close()

#print(data)

pattern = re.compile('\d+:\d+\s+-\s+([a-zA-Z0-9]+\s?[a-zA-Z0-9]+\s?[a-zA-Z0-9]+\s?):\s+')

messengers = re.findall(pattern,data)

if(len(messengers)==0):

    pattern = re.compile('\d+:\d+\s+[a-z][a-z]\s+-\s+([a-zA-Z0-9]+\s?[a-zA-Z0-9]+\s?[a-zA-Z0-9]+\s?):\s+')

messengers = re.findall(pattern,data)

senders = (np.unique(messengers))

countdict={}

for sender in senders:

	countdict.update({sender:messengers.count(sender)})

countlist = sorted(countdict.items(), key=lambda kv: kv[1])

#print(countlist)





messages_split = pattern.split(data)



sep_msgs=[]

for each in countdict.keys():

    for msg in range(len(messages_split)):

        if each == messages_split[msg]:

            sep_msgs.append(messages_split[msg+1])



cleaned_sep_msg = []

for each in sep_msgs:

    if '\n0' in each:

        cleaned_sep_msg.append(each.split('\n0'))

    elif '\n1' in each:

        cleaned_sep_msg.append(each.split('\n1'))

    elif '\n2' in each:

        cleaned_sep_msg.append(each.split('\n2'))

    elif '\n3' in each:

        cleaned_sep_msg.append(each.split('\n3'))

my_msg = []

for each in cleaned_sep_msg:

    my_msg.append(each[0])



for each in countdict.keys():

    if messages_split[-2] == each:

        my_msg.insert(countdict[each]-1,messages_split[-1])



        who_sent_what = []

prev = 0

for each in countdict.keys():

    num = countdict[each]

    

    nex = num+prev

    messages = my_msg[prev:nex]

    who_sent_what.append(messages)

    prev = nex
count=0

countofeach=[]

col2=[]

for entry in who_sent_what:

    countofeach.append(len(entry))

    for text in entry:

        col2.append(text)





col1=[]

for i in range(len(who_sent_what)):

    send=list(countdict.keys())

    for j in range(countofeach[i]):

        col1.append(send[i])



df=pd.DataFrame(list(zip(col2,col1)),columns=['message','sender'])

df.groupby(df.sender).count().plot(kind='bar',title="Number of Messages Sent")
result=(df[df['message']=="<Media omitted>"])

result['sender'].groupby(df.sender).count().plot(kind='bar',title="Number of Media Messages sent")
result=(df[df['message']=="This message was deleted"])

result['sender'].groupby(df.sender).count().plot(kind='bar',title="Number of Messages Deleted")
result=df[df['message'].apply(lambda x:len(x)<=4)]

result['sender'].groupby(df.sender).count().plot(kind='bar',title="Number of short replies sent")
def text_has_emoji(text):

    

    for character in text:

        if character not in emoji.UNICODE_EMOJI:

            return False

    return True



result=df[df['message'].apply(lambda x:text_has_emoji(x)==True)]
result=result.groupby(['sender'])['message'].apply(lambda x: x.value_counts().index[0]).reset_index()

print("Most Used Emoji\n")

print(result)
df=pd.DataFrame(list(zip(col2,col1)),columns=['message','sender'])
temp=df[df['message']!="<Media omitted>"]

temp2=temp[temp['message']!="This message was deleted"]

temp2=temp2[temp2['message']!="You deleted this message"]

temp3=temp2[temp2['message']!="."]

temp4=temp3[temp3['message'].apply(lambda x:text_has_emoji(x)==False)]

result=temp4.groupby(['sender'])['message'].apply(lambda x: x.value_counts().index[0]).reset_index()

print("Most Used Messages\n")

print(result)
dupes=pd.concat(g for _, g in temp4.groupby("message") if len(g) > 5)

dupes2=dupes[temp['message'].apply(lambda x:x[0]!='@')]

dupes2.loc['message'] = dupes2['message'].str.replace(r'[.]$', '')

data=dupes2.groupby(['message']).count()



data=data.reset_index()

report=data.sort_values(by=['sender'],ascending=False)

report=report[0:10].reset_index()

report=report.drop(['index'],axis=1)

report.columns=["Most Sent Text","Freq"]

print(report)
def text_has_emoji2(text):

    

    for character in text:

        if character in emoji.UNICODE_EMOJI:

            return False

    return True



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

import matplotlib as mpl



mpl.rcParams['figure.figsize']=(12,8)    #(6.0,4.0)

mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=300            #72 

mpl.rcParams['figure.subplot.bottom']=.1 





stopwords = set(STOPWORDS)



d = {}

for a, x in data.values:

    if(text_has_emoji2(a)==True):

        d[a] = x



import matplotlib.pyplot as plt

from wordcloud import WordCloud



wordcloud = WordCloud(background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42)

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()