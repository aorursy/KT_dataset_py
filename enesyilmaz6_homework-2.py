# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

data3 = pd.read_csv('../input/yonelimfinal.csv')
data3.columns # we want know column name
data3.head()
data3['Age Range'] = ['Relevant' if i != '0-18' else 'Irrelevant' for i in data3['Yas']]
data3.loc[0:len(data3),["Age Range","Yas"]] # we see two columns values
data3['Education Situation'] = ['On Lisans ve Üstü' if i == 'Lisans' or i == 'Ön Lisans' or i == 'Lisans Üstü' else 'Lise ve Altı' for i in data3['Egitim']]
data3.tail(10)
data3['Egitim'].unique() # see education situation
datax = pd.DataFrame(data3,columns = ['Cinsiyet','Yas','Bolge','Egitim','parti','Age Range','Education Situation'])

datax = datax[np.logical_and(data3['Age Range'] == 'Relevant',data3['Education Situation'] == 'On Lisans ve Üstü' )]

datax = datax.reset_index(drop = True)
datax.info()
datax.head()
datax.columns
parti = list(datax['parti'].unique())
hexalist = datax.groupby('parti').size()
parti
counter_list = [] #global list

def partisayaci():

    for a in range(len(parti)):

        counter = 0 #local variable

        for b in range(len(datax)):

            if parti[a] == datax['parti'][b]:

                counter += 1

            else:

                continue

        counter_list.append(counter)

    return print(counter_list)

partisayaci()

print(parti)

counter_list
dictionary = {} # we will create dict in that party name : number of votes

for w in range(len(parti)):

    dictionary[parti[w]] = counter_list[w]

print(dictionary)

dfparty_new = pd.DataFrame.from_dict(dictionary,orient='index',columns=['Voter']) # we created dataframe from dictionary 
dfparty_new # This is number of voters by party
print(parti) # party names

print(counter_list) # party votes
print(dictionary) # parti-counter list unified
dfparty_new['Voter']
#partyname['IYI',       'AKP'    'DIGER',  'hdp',   'CHP'  'MHP']

colors = ['turquoise','yellow','orange','purple','red','green']

fig = plt.figure(figsize=(8,7))

datax['parti'].value_counts().plot(kind = 'pie', autopct='%.1f%%')

plt.title('Party Ratio')

plt.legend(labels=['IYI','AKP','OTH','HDP','CHP','MHP']) # colors --> party name

plt.tight_layout() #regularity

plt.show()
datax2 = pd.DataFrame(data3,columns = ['Cinsiyet','Yas','Bolge','Egitim','parti','Age Range','Education Situation'])

datax2 = datax2[np.logical_and(data3['Age Range'] == 'Relevant',data3['Education Situation'] == 'Lise ve Altı' )]

datax2 = datax2.reset_index(drop = True)
datax2.head()
#partyname['IYI',       'AKP'    'DIGER',  'hdp',   'CHP'  'MHP']

colors = ['turquoise','yellow','orange','purple','red','green']

fig = plt.figure(figsize=(8,7))

datax2['parti'].value_counts().plot(kind = 'pie', autopct='%.1f%%')

plt.title('Party Ratio')

plt.legend(labels=['IYI','AKP','OTH','HDP','CHP','MHP']) # colors --> party name

plt.tight_layout() #regularity

plt.show()