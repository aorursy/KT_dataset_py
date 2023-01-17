import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import random as random



e = pd.read_csv("../input/survey.csv")

e.info()





e = e[e['Country']=='United States']

ye = e[e['treatment']=='Yes']

nah = e[e['treatment']=='No']

yes = ye.treatment.value_counts()

no = nah.treatment.value_counts()

kokku = yes['Yes'].tolist()+no['No'].tolist()

print("Vaimse seisundi parandamiseks on abi otsinud "+str(round(yes['Yes'].tolist()*100/kokku,1))+" protsenti "+str(kokku)+"st küsitletud inimesest")



kt = e[e['remote_work']=='Yes']

ot = kt[kt['treatment']=='Yes']

ise = e[e['self_employed']=='Yes']

isea = ise[ise['treatment']=='Yes']

jahkokku = yes.tolist()[0]

arv = kt.remote_work.value_counts().tolist()[0]

arv2 = ot.remote_work.value_counts().tolist()[0]

arv3 = ise.self_employed.value_counts().tolist()[0]

arv4 = isea.self_employed.value_counts().tolist()[0]

protsent = round(arv2*100/arv)

protsent2 = round(arv4*100/arv3)

print(str(arv)+"st kaugtööd teinud küsitletust otsis vaimset abi "+str(arv2)+" inimest ehk "+str(protsent)+"%")

print(str(arv3)+"st isetöötajast otsis vaimset abi "+str(arv4)+" inimest ehk "+str(protsent2)+"%")

def age_process(age): #loome meetodi vanuste filtreerimiseks

    if age>=17 and age<=80:

        return age

    else:

        return np.nan

e['Age'] = e['Age'].apply(age_process) #filtreerime vanused 17-80
n = e[e['treatment']=='No']

n.Age.plot.hist(subplots=True, bins=53, grid=False, rwidth=0.95, color='Red')

m = e[e['treatment']=='Yes']

m.Age.plot.hist(subplots=True, bins=53, grid=False, rwidth=0.95)
def noemployees_process(no_employees): #meetod töötajate arvu muutmiseks arvudeks

    if no_employees == '6-25':

        return random.randint(6, 25)

    elif no_employees == '26-100':

        return random.randint(26,100)

    elif no_employees == '100-500':

        return random.randint(100, 500)

    elif no_employees == '500-1000':

        return random.randint(500, 1000)

    elif no_employees == 'More than 1000':

        return random.randint(1000, 5000)

    elif no_employees == '1-5':

        return random.randint(1,5)

    else:

        return 0



e['no_employees'] = e['no_employees'].apply(noemployees_process) #filtreerime
e = e[e["Country"]=='United States']

e.plot.scatter(x='Age', y='no_employees', alpha=0.2, subplots=True)

e['Vanusegrupp'] = pd.cut(e['Age'].dropna(), #

                         [18,25,35,45,99],

                         labels=['18-24','25-34','35-44','45+'])

fig,ax = plt.subplots(figsize=(8,6))

sb.countplot(data=e,x = 'seek_help',ax=ax)

plt.title('Info kättesaadavus')
e['Töötajate suurusgrupp'] = pd.cut(e['no_employees'].dropna(), #

                         [1,5,25,100,500,1000, 5000],

                         labels=['1-5','5-25','25-100','100-500', '500-1000', '1000+'])

fig,ax = plt.subplots(figsize=(8,6))

sb.countplot(data=e,x = 'Töötajate suurusgrupp', hue= 'seek_help',ax=ax)

plt.title('Info kättesaadavus')
def seekhelp_process(seek_help): #loome meetodi Jah/Ei vastuste filtreerimiseks

    if seek_help == 'Yes' or seek_help == 'No':

        return seek_help

    else:

        return np.nan

    

e['seek_help'] = e['seek_help'].apply(seekhelp_process) #filtreerime



fig,ax = plt.subplots(figsize=(8,6))

sb.countplot(data=e,x = 'treatment',hue= 'seek_help',ax=ax)

plt.title('Info kättesaadavus vs Abi otsimine')

fig,ax = plt.subplots(figsize=(8,6))

sb.countplot(data=e,x = 'Töötajate suurusgrupp', hue= 'work_interfere',ax=ax)

plt.title('Vaimne tervis segab tööd?')
e['Vanusegrupp'] = pd.cut(e['Age'].dropna(), #

                         [18,25,35,45,99],

                         labels=['18-24','25-34','35-44','45+'])
tr = e[e['treatment']=='Yes']

fig,ax = plt.subplots(figsize=(8,6))

sb.countplot(data=tr,x = 'Vanusegrupp',hue= 'family_history',ax=ax)

plt.title('Vanus vs Perekondlik Ajalugu')
fig,ax = plt.subplots(figsize=(20,30))

sb.countplot(data=e,x = 'state',hue= 'treatment',ax=ax)
#e.groupby(['state'])['phys_health_consequence','mental_health_consequence']



df = e[['treatment','phys_health_consequence','mental_health_consequence']].groupby(['treatment','phys_health_consequence','mental_health_consequence']).size().reset_index(name='count')

df1 = df[df['mental_health_consequence']=='Yes']

df1.sort_values("count", ascending=False)
fig,ax = plt.subplots(figsize=(8,6))

sb.countplot(data=e,x = 'mental_vs_physical',hue= 'treatment',ax=ax)

plt.title('Kas tööandjad võtavad vaimset tervist sama tõsiselt kui füüsilist?')
fig,ax = plt.subplots(figsize=(8,6))

#interfere = e[e['treatment']=='Yes']

sb.countplot(data=e,x = 'phys_health_interview',ax=ax)
fig,ax = plt.subplots(figsize=(8,6))

#interfere = e[e['treatment']=='Yes']

sb.countplot(data=e,x = 'mental_health_interview',ax=ax)