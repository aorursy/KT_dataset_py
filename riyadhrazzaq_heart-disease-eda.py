# imports

import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

import seaborn as sns

## models

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn import svm
# uci = pd.read_csv(f"{DATA}heart-disease-eda/heart-UCI.csv")

fhs = pd.read_csv('/kaggle/input/framingham-heart-study-dataset/framingham.csv')
fhs.head()
fhs.describe()
impMost = SimpleImputer(strategy='most_frequent')

impMean = SimpleImputer(strategy='mean')

fhs.isnull().sum()
# education

fhs['education'] = impMost.fit_transform(fhs['education'].values.reshape(-1,1))
# cigsperday

rows = fhs['cigsPerDay'].isnull()

fhs.loc[rows,'cigsPerDay'] = int(fhs['cigsPerDay'].mean())
# bpmeds

fhs['BPMeds'] = impMost.fit_transform(fhs['BPMeds'].values.reshape(-1,1))
# totChol

fhs['totChol'] = impMean.fit_transform(fhs['totChol'].values.reshape(-1,1))
fhs['BMI'] = impMean.fit_transform(fhs['BMI'].values.reshape(-1,1))
row = fhs['heartRate'].isnull()

fhs.loc[row,'heartRate'] = int(fhs['heartRate'].mean())
row = fhs['glucose'].isnull()

fhs.loc[row,'glucose'] = int(fhs['glucose'].mean())
fhs.isnull().sum()
## rename TenYearCHD column name

fhs = fhs.rename(columns={'TenYearCHD':'target'})
# male vs target

fig, ax = plt.subplots(nrows=1,ncols=2,facecolor='white',figsize=(10,5))

gender = fhs['male'].value_counts()

ax[0].bar(gender.index.values, gender.values, align='center', \

          tick_label=['female','male'],width=0.2)

ax[0].set_title('Distribution of Gender Data')

gender = gender.index.values

count = [0,0]

for t,g in zip(fhs['target'],fhs['male']):

    count[g] += t



ax[1].bar(gender,count,align='center',tick_label=['female','male'],width=0.2)

ax[1].set_title('Distribution of Target in Different Sex')

plt.show()
# age vs target

fig, ax = plt.subplots(1,1,facecolor='white')

x = sorted(fhs['age'].unique())

y = fhs.groupby(['age']).mean()['target'].values

plt.scatter(x,y)

plt.show()
# education 

fix, ax = plt.subplots(1,2,facecolor='white',figsize=(15,5))

edu = fhs['education'].value_counts().index

freq = fhs['education'].value_counts()

ax[1].barh(y=edu,width=freq,tick_label=edu,height=0.1)

ax[1].set_xlabel('number of people')

ax[1].set_title('Distribution of Education level')

ax[1].set_ylabel('Education level')



countOrig = [0,0,0,0]

for t,e in zip(fhs['target'],fhs['education']):

    countOrig[int(e-1)] += t

count = list(map(lambda x: -x,countOrig))

ax[0].barh(y=edu,width=count,height=0.1,color='red')

ax[0].set_xticklabels(list(map(lambda x: str(abs(x)),ax[0].get_xticks())))

ax[0].set_title('Target true for each education levels')

ax[0].set_xlabel('number of people with Target = True')

ax[0].tick_params(axis='y',labelsize=0,which='major', \

                 bottom=False,top=False,left=False)

plt.show()
fig, ax = plt.subplots(figsize=(15,5),facecolor='white')

xlab = fhs['education'].value_counts().index

freq = fhs['education'].value_counts().values

rates = [(c/float(f))*100 for f,c in zip(freq,countOrig)] 

counts = countOrig

print(rates,counts,freq)





x = np.arange(len(xlab))

width = 0.3

rect1  = ax.bar(x-width/2, rates,width=width,label='Rates of CHD in 10 years',color='blue')

rect2 = ax.bar(x+width/2,counts,width=width,label='CHD in 10 years')



ax.set_ylabel('CHD Prediction for 10 years')

ax.set_title('CHD Prediction in Education Level')

ax.set_xticks(x)

ax.set_xticklabels(edu)

ax.legend()



def autoLabel(rect):

    for r in rect:

        heights = r.get_height()

        ax.annotate("{:.2f}".format(heights),

                   xy=( r.get_x()+ r.get_width()/2, heights),

                   xytext=(0,0),

                   textcoords='offset points',

                   ha='center',

                   va='bottom')



autoLabel(rect1)

autoLabel(rect2)

plt.show()
fig, ax = plt.subplots(figsize=(8,5),facecolor='white')

xlab = ['Non-smoker','Smoker']

x = np.arange(len(xlab))

freq = fhs['currentSmoker'].value_counts().values

dist = [0 for w in xlab]





for t,s in zip(fhs['target'],fhs['currentSmoker']):

    dist[s] += t

rates = [(d/float(f))*100 for d,f in zip(dist,freq)]

width = 0.2

rect1 = ax.bar(x - width/2,freq, width=width/2,color='green',label='Total people')

rect2 = ax.bar(x+width/2, dist, width= width/2,color='black',label='Total people pre. w/ CHD')

rect3 = ax.bar(x,rates,width=width/2,color='purple',label='Total people pre. w/ CHD per 100')

ax.legend()

ax.set_xticks(x)

ax.set_xticklabels(xlab)



## define autolabel to remember

def autoLabel(rect,k=None):

    for b in rect:

        height = b.get_height()

        if k is None:

            ax.annotate("%.2f"%(height),

                       xy=(b.get_x(),height),

                        xytext=(0,0),

                       xycoords='data',

                       textcoords='offset points',

                       ha='left',

                       va='bottom')

        else:

                ax[k].annotate("%.2f"%(height),

                       xy=(b.get_x(),height),

                        xytext=(0,0),

                       xycoords='data',

                       textcoords='offset points',

                       ha='left',

                       va='bottom',

                        rotation=45)



autoLabel(rect1)

autoLabel(rect2)

autoLabel(rect3)

plt.show()
# cigsperday vs target

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,10),facecolor='white')

dist = fhs['cigsPerDay'].value_counts()

cigs = [int(x) for x in sorted(dist.index)] 

chdCount = dict(map(lambda x: [x,0],cigs))



for t,c in zip(fhs['target'],fhs['cigsPerDay']):

    chdCount[c] += t

rates = [(chdCount[c] / float(dist.loc[c]))*100 for c in cigs]



x = np.arange(len(cigs))



bar1 = ax[0].bar(x,chdCount.values())

ax[0].set_xticks(x)

ax[0].set_xticklabels(cigs)

ax[0].set_ylabel('Number of People')

ax[1].bar(x,rates)

ax[1].set_xticks(x)

ax[1].set_xticklabels(cigs)

ax[1].set_xlabel('Cig Per Day')

ax[1].set_ylabel('People w/ CHD (%)')



autoLabel(bar1,0)



plt.show()

print(chdCount[12],dist.loc[12])
# BPMeds



fig, ax = plt.subplots(figsize=(8,5),facecolor='white')

xlab = ['No','Yes']

x = np.arange(len(xlab))

dist = fhs['BPMeds'].value_counts().values

chd = [0 for w in xlab]

for t,c in zip(fhs['target'],fhs['BPMeds']):

    chd[int(c)] += t

rates = [(c/float(d))*100 for c,d in zip(chd,dist)]



width = 0.2

rect1 = ax.bar(x - width/2,dist, width=width/2,color='black',label='Total people')

rect2 = ax.bar(x+width/2, chd, width= width/2,color='red',label='Total people pre. w/ CHD')

rect3 = ax.bar(x,rates,width=width/2,color='blue',label='Total people pre. w/ CHD per 100')

ax.legend()

ax.set_xticks(x)

ax.set_xticklabels(xlab)

ax.set_xlabel('Takes Medicine')

autoLabel(rect1)

autoLabel(rect2)

autoLabel(rect3)
# prevalentStroke



fig, ax = plt.subplots(figsize=(8,5),facecolor='white')

xlab = ['No','Yes']

x = np.arange(len(xlab))

dist = fhs['prevalentStroke'].value_counts().values

chd = [0 for w in xlab]

for t,c in zip(fhs['target'],fhs['prevalentStroke']):

    chd[int(c)] += t

rates = [(c/float(d))*100 for c,d in zip(chd,dist)]



width = 0.2

rect1 = ax.bar(x - width/2,dist, width=width/2,color='black',label='Total people')

rect2 = ax.bar(x+width/2, chd, width= width/2,color='red',label='Total people pre. w/ CHD')

rect3 = ax.bar(x,rates,width=width/2,color='blue',label='Total people pre. w/ CHD per 100')

ax.legend()

ax.set_xticks(x)

ax.set_xticklabels(xlab)

ax.set_xlabel('Prevalent Stroke')

autoLabel(rect1)

autoLabel(rect2)

autoLabel(rect3)
# prevalent Hypertension



fig, ax = plt.subplots(figsize=(8,5),facecolor='white')

xlab = ['No','Yes']

x = np.arange(len(xlab))

dist = fhs['prevalentHyp'].value_counts().values

chd = [0 for w in xlab]

for t,c in zip(fhs['target'],fhs['prevalentHyp']):

    chd[int(c)] += t

rates = [(c/float(d))*100 for c,d in zip(chd,dist)]



width = 0.2

rect1 = ax.bar(x - width/2,dist, width=width/2,color='black',label='Total people')

rect2 = ax.bar(x+width/2, chd, width= width/2,color='red',label='Total people pre. w/ CHD')

rect3 = ax.bar(x,rates,width=width/2,color='blue',label='Total people pre. w/ CHD per 100')



ax.legend()

ax.set_xticks(x)

ax.set_xticklabels(xlab)

ax.set_xlabel('Prevalent Hyp')

autoLabel(rect1)

autoLabel(rect2)

autoLabel(rect3)

plt.show()
# diabetes

fig, ax = plt.subplots(figsize=(8,5),facecolor='white')

xlab = ['No','Yes']

x = np.arange(len(xlab))

dist = fhs['diabetes'].value_counts().values

chd = [0 for w in xlab]

for t,c in zip(fhs['target'],fhs['diabetes']):

    chd[int(c)] += t

rates = [(c/float(d))*100 for c,d in zip(chd,dist)]



width = 0.2

rect1 = ax.bar(x - width/2,dist, width=width/2,color='black',label='Total people')

rect2 = ax.bar(x+width/2, chd, width= width/2,color='red',label='Total people pre. w/ CHD')

rect3 = ax.bar(x,rates,width=width/2,color='blue',label='Total people pre. w/ CHD per 100')



ax.legend()

ax.set_xticks(x)

ax.set_xticklabels(xlab)

ax.set_xlabel('diabetes')

autoLabel(rect1)

autoLabel(rect2)

autoLabel(rect3)

plt.show()
# totchol

chol0 = []

chol1 = []

for c,t in zip(fhs['totChol'],fhs['target']):

    if t==1:

        chol1.append(c)

    else:

        chol0.append(c)



fig, ax = plt.subplots(figsize=(10,8),facecolor='white')

ax.boxplot([chol0,chol1],notch=True)

ax.set_xticklabels(['w/o CHD','w/ CHD'])

ax.grid(True)
# sysBP

bp0 = []

bp1 = []

for c,t in zip(fhs['sysBP'],fhs['target']):

    if t==1:

        bp1.append(c)

    else:

        bp0.append(c)



fig, ax = plt.subplots(figsize=(10,8),facecolor='white')

ax.boxplot([bp0,bp1],notch=True)

ax.set_xticklabels(['w/o CHD','w/ CHD'])

ax.grid(True)
# diaBP

diabp0 = []

diabp1 = []

for c,t in zip(fhs['diaBP'],fhs['target']):

    if t==1:

        diabp1.append(c)

    else:

        diabp0.append(c)



fig, ax = plt.subplots(figsize=(10,8),facecolor='white')

ax.boxplot([diabp0,diabp1],notch=True)

ax.set_xticklabels(['w/o CHD','w/ CHD'])

ax.grid(True)
# totchol

bmi0 = []

bmi1 = []

for c,t in zip(fhs['BMI'],fhs['target']):

    if t==1:

        bmi1.append(c)

    else:

        bmi0.append(c)



fig, ax = plt.subplots(figsize=(10,8),facecolor='white')

ax.boxplot([bmi0,bmi1],notch=True,meanline=True,showmeans=True)

ax.set_xticklabels(['w/o CHD','w/ CHD'])

ax.grid(True)
# heartrate

hr0 = []

hr1 = []

for c,t in zip(fhs['heartRate'],fhs['target']):

    if t==1:

        hr1.append(c)

    else:

        hr0.append(c)



fig, ax = plt.subplots(figsize=(10,8),facecolor='white')

ax.boxplot([hr0,hr1],notch=True)

ax.set_xticklabels(['w/o CHD','w/ CHD'])

ax.grid(True)
# glucose

glu0 = []

glu1 = []

for c,t in zip(fhs['glucose'],fhs['target']):

    if t==1:

        glu1.append(c)

    else:

        glu0.append(c)



fig, ax = plt.subplots(figsize=(10,8),facecolor='white')

ax.boxplot([glu0,glu1],notch=True)

ax.set_xticklabels(['w/o CHD','w/ CHD'])

ax.grid(True)
## correlation matrix & plot without sns :v 

corr = fhs.corr()

data = corr.values

labels = corr.index



fig, ax = plt.subplots(facecolor='white',figsize=(10,10))

sns.heatmap(corr,vmin=-1,vmax=1,center=0)

plt.show()

cols = ['age', 'prevalentHyp', 'sysBP', 'diaBP', 'age', 'glucose']

trainX, testX,trainy,testy = train_test_split(fhs[cols],fhs['target'],test_size=0.3)
clf = svm.SVC(decision_function_shape='ovo',probability=True)

clf.fit(trainX,trainy)

print(clf.score(testX,testy))

print(clf.score(trainX,trainy))