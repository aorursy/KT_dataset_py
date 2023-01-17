# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/stack-overflow-developer-survey-2020/developer_survey_2020/survey_results_public.csv')
def prettify(ax):
  ax.grid(False)
  ax.set_frame_on(False)
  ax.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False)
plt.style.use('ggplot')

dft = df.groupby(['Country']).count().sort_values('Respondent', ascending=False).head(15)
cntrs = list(dft.index)
cntrs[0], cntrs[2], cntrs[-3] = "USA", "UK", "Russia"
cntrs = list(reversed(cntrs))

fig1, ax1 = plt.subplots(figsize=(9,6))
ax1.barh(cntrs, list(reversed(dft['Respondent'])))
ax1.set_xticks(np.arange(0,13001,1000))
ax1.grid(axis='y')
# ax1.set_title('\nTop 15 Countries of Respondents',fontdict={'fontsize':20})
fig1.tight_layout()

plt.style.use('default')
plt.style.use('bmh')

dft = df.groupby(['Ethnicity']).count().sort_values('Respondent', ascending=False)
races = [str(x) for x in dft.index[0:10]]
pies = [dft.Respondent[dft.index==races[x]][0] for x in range(10)]

fig, ax = plt.subplots(figsize=(10,10))
ax.pie(
    pies,
    labels=races,
    autopct='%.1f%%',
    explode=[0.01, 0.02, 0.03, 0.04, 0.04, 0.04, 0.05, 0.08, 0.1, 0.25],
    startangle=-150, pctdistance=0.7, labeldistance=1.05
)
# ax.set_title('\nTop 10 Ethnicities of Programmers',fontdict={'fontsize':20})
fig.tight_layout()
# fig.savefig('1.png', dpi=200, bbox_inches='tight')
plt.show()
plt.style.use('default')
plt.style.use('seaborn')

dft = df.groupby(['Gender']).Respondent.count()

fig, ax = plt.subplots()
pies = []
temp = 0
for i in dft.index:
    if i=='Man':
        pies.append(dft[i])
    elif i=='Woman':
        pies.append(dft[i])
    else:
        temp+=dft[i]
pies.append(temp)
labels = ['Man', 'Woman', 'Non-conforming']
_ = ax.pie(pies, labels=labels, autopct="%.1f%%" ,explode=[0.03,0.01,0.1], startangle=180)
# ax.set_title('Gender Of Programmers', fontdict={'fontsize':20}, horizontalalignment='right')
# fig.savefig('3.png',dpi=200, bbox_inches='tight')
# plt.style.use('dark_background')
plt.style.use('seaborn')


fig, ax = plt.subplots()

#Trans or not
ax.pie(
    [
        df.Respondent[df.Trans=='Yes'].count(),
        df.Respondent[df.Trans=='No'].count(),
        df.Respondent[df.Trans.isna()].count()
    ],
    autopct='%.2f%%',
    labels=['Transgender', 'Not A Transgender', 'Didn\'t tell'],
    explode=[0.05, 0.01, 0.01]
)
# ax.set_title('Is Respondent Transgender Or Not?',fontdict={'fontsize':20}, horizontalalignment='right')

#Gender

plt.style.use('default')
plt.style.use('classic')

#Sexuality
dft = df.groupby(['Sexuality']).Respondent.count()

fig, ax = plt.subplots()
pies = []
temp = 0
for i in dft.index:
    if i=='Straight / Heterosexual':
        pies.append(dft[i])
    elif i=='Gay or Lesbian':
        pies.append(dft[i])
    elif i=='Bisexual':
        pies.append(dft[i])
    else:
        temp+=dft[i]
pies.append(temp)
labels = ['Bisexual', 'Gay', 'Straight', 'Others']
ax.pie(pies, labels=labels, autopct="%.1f%%", startangle=150, labeldistance=1.05)
# ax.set_title('Sexual Orientation Of Programmers', fontdict={'fontsize':20}, horizontalalignment='right')

# fig.savefig('3.png',dpi=200, bbox_inches='tight')

plt.style.use('default')
plt.style.use('fast')

dft = df.groupby(['EdLevel']).count()
EdLvl = [
    x[:x.find('(')-1] if x.find('(')!=-1
    else "No Formal Education" if x.find('formal')!=-1
    else "College Dropout" if x.find('without earning')!=-1
    else x
    for x in dft.index 
]
pies = [dft.Respondent[dft.index==x][0] for x in dft.index]

fig, ax = plt.subplots()
ax.pie(pies, labels=EdLvl, startangle=-120, autopct="%.1f%%", pctdistance=0.75,
      explode=[0,0,0.05,0,0,0.05,0,0,0], labeldistance=1.03)
# ax.set_title('Education Levels',fontdict={'fontsize':30})

plt.savefig('4.png', dpi=200, bbox_inches='tight')
plt.style.use('default')
plt.style.use('ggplot')

dft = df.drop(df[df.Age>90].index, axis=0).drop(df[df.Age<10].index, axis=0)

fig, ax = plt.subplots(figsize=(8,4))
ax = sns.violinplot(dft.Age, inner='quartile')
_ = ax.set_xticks(np.arange(0,101,10))
# ax.set_title('Age Of Programmers')
ax.grid(axis='y')

plt.style.use('default')
plt.style.use('seaborn')

fig,ax = plt.subplots()
dft = df.groupby(['EdLevel']).mean()
ax.barh(EdLvl, dft.Age)
# ax.set_title('Mean Ages Of Various Education Levels',fontdict={'fontsize':30})
ax.set_xlabel('Age')
ax.grid(axis='y')

plt.style.use('default')
plt.style.use('bmh')

dft = df.groupby(['Employment']).count()#.sort_values(by='Respondent',ascending=False)

fig, ax = plt.subplots( tight_layout=True, figsize=(10,16))
ax.pie(dft.Respondent, labels=dft.index, startangle=-130,
       explode=[0.01, 0.03, 0.02, 0.06, 0.02, 0.06, 0.02],
       autopct="%.1f%%", pctdistance=0.7, labeldistance=1.05)
# ax.set_title('Employment Status',fontdict={'fontsize':20}, horizontalalignment='center')

fig.savefig('5.png')
plt.show()

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots()

dft = df.groupby(['Employment']).mean()
ax.barh(dft.index, dft.Age)
# ax.set_title('Mean Age Of Various Employments',fontdict={'fontsize':20}, horizontalalignment='right')
ax.set_xlabel('Age')
ax.grid(axis='x')
plt.style.use('default')
plt.style.use('dark_background')

df.Age1stCode.replace('Younger than 5 years', '4', inplace=True)
df.Age1stCode.replace('Older than 85', '86', inplace=True)
df.Age1stCode.fillna(np.nan)
df.Age1stCode = df.Age1stCode.astype('float')

bins = [4,8,10,11,12,13,14,15,16,17,18,19,20,22,25,30,50,86]
names = [x for x in bins[1:]]
names[-1] = 21
df['fc'] = pd.cut(df.Age1stCode.dropna(), bins=bins, labels=names, right=False)
labels=['50 years or older ', '30 to 49 years old ', '25 to 29 years old ',
                         '22 to 24 years old ','20 to 21 years old ','19 years old ','18 years old ',
                          '17 years old ','16 years old ','15 years old ','14 years old ','13 years old ',
                         '12 years old ', '11 years old ', '10 years old ', '8 to 10 years old ', 
                          'less than 8 years old ']

dft = df.groupby(['fc']).Respondent.count()
nums = [dft[x] for x,_ in enumerate(dft.index)]
ticks = []
for x,_ in enumerate(nums):
    ticks.append(x)

vals = [round(100*x/sum(nums),2) for x in reversed(nums)]


fig, ax = plt.subplots(figsize=(10,12))
ax.barh(ticks, [x for x in reversed(nums)], edgecolor='b', tick_label=labels)
ax.grid(True)
ax.set_yticks(ticks) 

for (x, v), y in zip(enumerate(vals), reversed(nums) ) :
    ax.annotate("{}%".format(v), (y+50,x-0.2), fontsize=15)

fig.tight_layout()
prettify(ax)
# ax.set_title('\n\nWritten Their First Code' ,fontdict={'fontsize':30}, horizontalalignment='right')
fig.savefig('0.png', dpi=200, bbox_inches='tight')
plt.style.use('default')

plt.style.use('seaborn')

fig, ax = plt.subplots(figsize=(12,6))
sns.violinplot(df.Age1stCode, inner='quartile')
ax.set_xticks(np.arange(0,90,5))

plt.style.use('default')
plt.style.use('ggplot')
dft = df.groupby(['Country']).Age1stCode.mean()
cntr =['Pakistan','India','Turkey','France','Germany','Canada','United States','United Kingdom']
m_age = [dft[x] for x in cntr]

fig, ax = plt.subplots()
ax.barh(cntr, m_age)
ax.set_xticks(np.arange(0,19,2))
# ax.set_title('Average Age Of Writing 1st Code Categorized by Country\n' ,fontdict={'fontsize':15}, horizontalalignment='center')
ax.grid(axis='y')

plt.style.use('default')
plt.style.use('ggplot')
# plt.style.use('dark_background')

df.YearsCode.replace('Less than 1 year', '0.5', inplace=True)
df.YearsCode.replace('More than 50 years', '52', inplace=True)
df.YearsCode = df.YearsCode.astype('float')
dft = df.groupby(['Country']).YearsCode.mean()
cntr =['Pakistan','India','Turkey','France','Germany','Canada','United States','United Kingdom']
age = [dft[x] for x in cntr]

fig, ax = plt.subplots(nrows=2, figsize=(10,8),sharex=True)
fig.subplots_adjust(hspace=0)
ax[0].barh(cntr, age, color='g', edgecolor='w')
ax[0].grid(axis='y')
# ax[0].set_title('\nAverage Years of Coding Experience and Professional Coding Experience\n' ,fontdict={'fontsize':20}, horizontalalignment='center')
# ax[0].set_xticks()

df.YearsCodePro.replace('Less than 1 year', '0.5', inplace=True)
df.YearsCodePro.replace('More than 50 years', '52', inplace=True)
df.YearsCodePro = df.YearsCodePro.astype('float')
dft = df.groupby(['Country']).YearsCodePro.mean()
cntr =['Pakistan','India','Turkey','France','Germany','Canada','United Kingdom','United States']
age = [dft[x] for x in cntr]

ax[1].barh(cntr, age, color='red', edgecolor='w')
ax[1].set_xlabel('Years')
ax[1].grid(axis='y')
plt.style.use('default')
plt.style.use('default')
dft = df.groupby(['OpSys']).Respondent.count()
types = [x for x in dft.index]
pcnts = [round(100*dft[x]/sum(nums),1) for x in dft.index]

fig, ax = plt.subplots()
ax.barh(types, pcnts)

for y, p in enumerate(pcnts):
  ax.annotate("%.1f%%"%p, (p+0.5, y))
prettify(ax)
# ax.set_title('Primary Operating System For Work')

plt.style.use('default')
plt.style.use('seaborn-dark')

# current lw users
lw = {}
for x in df.LanguageWorkedWith.dropna():
    for d in x.split(';'):
        if d not in lw:
            lw[d] = 1   # create new instance
        else:
            lw[d] += 1  # increments that instance           
lw = {k:v for k,v in sorted(lw.items(), key=lambda x:x[1])}
pos = np.arange(0,len(lw.keys()), 1)
labels = [x for x in lw.keys()]
tot = df.LanguageWorkedWith.count()
pcnts = [round(100*x/tot, 2) for x in lw.values()]

fig, ax = plt.subplots(figsize=(10,10))
fig.subplots_adjust(hspace=0.01)

ax.barh(labels, lw.values(), color='y', edgecolor='k')
ax.set_xticks(np.arange(0, max(lw.values())+600, 100))
ax.xaxis.set_ticklabels([])
for (y, p), x  in zip(enumerate(pcnts), lw.values()):
    ax.annotate("%.2f%%"%p, (x+100 ,y-0.125))
ax.xaxis.set_ticks_position('none')
# ax[0].set_title('Languages Worked With' ,fontdict={'fontsize':25}, horizontalalignment='center')
ax.grid(False)
ax.set_frame_on(False)
prettify(ax)
# fig.tight_layout()


# # ld that users desire
# ld = {}
# for x in df.LanguageDesireNextYear.dropna():
#     for d in x.split(';'):
#         if d not in ld:
#             ld[d] = 1   # create new instance
#         else:
#             ld[d] += 1  # increments that instance           
# ld = {k:v for k,v in sorted(ld.items(), key=lambda x:x[1])}
# pos = np.arange(0,len(ld.keys()), 1)
# labels = [x for x in ld.keys()]
# tot = df.LanguageDesireNextYear.count()
# pcnts = [round(100*x/tot,2) for x in ld.values()]

# ax[1].barh(labels, ld.values(), edgecolor='k')
# ax[1].set_xticks(np.arange(0, max(ld.values())+600, 100))
# ax[1].xaxis.set_ticklabels([])
# for (y, p), x  in zip(enumerate(pcnts), ld.values()):
#     ax[1].annotate("%.2f%%"%p, (x+80 ,y-0.125))
# ax[1].xaxis.set_ticks_position('none')
# ax[1].grid(False)
# # fig.tight_layout()
# ax[1].set_title('Languages Desired(wants to work with)' ,fontdict={'fontsize':25}, horizontalalignment='center')
# ax[1].set_frame_on(False)

# plt.style.use('default')
fig , ax = plt.subplots(figsize=(10,10))
# ld that users desire
ld = {}
for x in df.LanguageDesireNextYear.dropna():
    for d in x.split(';'):
        if d not in ld:
            ld[d] = 1   # create new instance
        else:
            ld[d] += 1  # increments that instance           
ld = {k:v for k,v in sorted(ld.items(), key=lambda x:x[1])}
pos = np.arange(0,len(ld.keys()), 1)
labels = [x for x in ld.keys()]
tot = df.LanguageDesireNextYear.count()
pcnts = [round(100*x/tot,2) for x in ld.values()]

ax.barh(labels, ld.values(), edgecolor='k')
ax.set_xticks(np.arange(0, max(ld.values())+600, 100))
ax.xaxis.set_ticklabels([])
for (y, p), x  in zip(enumerate(pcnts), ld.values()):
    ax.annotate("%.2f%%"%p, (x+80 ,y-0.125))
ax.xaxis.set_ticks_position('none')
ax.grid(False)
# fig.tight_layout()
# ax.set_title('Languages Desired(wants to work with)' ,fontdict={'fontsize':25}, horizontalalignment='center')
ax.set_frame_on(False)
prettify(ax)
plt.style.use('default')
plt.style.use('seaborn-dark')

# current DB users
db = {}
for x in df.DatabaseWorkedWith.dropna():
    for d in x.split(';'):
        if d not in db:
            db[d] = 1   # create new instance
        else:
            db[d] += 1  # increments that instance           
db = {k:v for k,v in sorted(db.items(), key=lambda x:x[1])}
pos = np.arange(0,len(db.keys()), 1)
labels = [x for x in db.keys()]
tot = df.DatabaseWorkedWith.count()
pcnts = [round(100*x/tot,2) for x in db.values()]

fig, ax = plt.subplots(figsize=(10,8))
fig.subplots_adjust(hspace=0.01)

ax.barh(labels, db.values(), color='y', edgecolor='k')
ax.set_xticks(np.arange(0, max(db.values())+600, 100))
ax.xaxis.set_ticklabels([])
for (y, p), x  in zip(enumerate(pcnts), db.values()):
    ax.annotate("%.2f%%"%p, (x+100 ,y-0.125))
ax.xaxis.set_ticks_position('none')
# ax.set_title('Databases Worked With' ,fontdict={'fontsize':20}, horizontalalignment='center')
ax.grid(False)
prettify(ax)
# fig.tight_layout()


# # db that users desire
# dbd = {}
# for x in df.DatabaseDesireNextYear.dropna():
#     for d in x.split(';'):
#         if d not in dbd:
#             dbd[d] = 1   # create new instance
#         else:
#             dbd[d] += 1  # increments that instance           
# dbd = {k:v for k,v in sorted(dbd.items(), key=lambda x:x[1])}
# pos = np.arange(0,len(dbd.keys()), 1)
# labels = [x for x in dbd.keys()]
# tot = df.DatabaseDesireNextYear.count()
# pcnts = [round(100*x/tot,2) for x in dbd.values()]

# ax[1].barh(labels, dbd.values(), edgecolor='k')
# ax[1].set_xticks(np.arange(0, max(dbd.values())+600, 100))
# ax[1].xaxis.set_ticklabels([])
# for (y, p), x  in zip(enumerate(pcnts), dbd.values()):
#     ax[1].annotate("%.2f%%"%p, (x+80 ,y-0.125))
# ax[1].xaxis.set_ticks_position('none')
# ax[1].grid(False)
# ax[1].set_title('Databases Desired(wants to work with)' ,fontdict={'fontsize':20}, horizontalalignment='center')

# fig.tight_layout()


# db that users desire
fig , ax = plt.subplots(figsize=(10,8))

dbd = {}
for x in df.DatabaseDesireNextYear.dropna():
    for d in x.split(';'):
        if d not in dbd:
            dbd[d] = 1   # create new instance
        else:
            dbd[d] += 1  # increments that instance           
dbd = {k:v for k,v in sorted(dbd.items(), key=lambda x:x[1])}
pos = np.arange(0,len(dbd.keys()), 1)
labels = [x for x in dbd.keys()]
tot = df.DatabaseDesireNextYear.count()
pcnts = [round(100*x/tot,1) for x in dbd.values()]

ax.barh(labels, dbd.values(), edgecolor='k')
ax.set_xticks(np.arange(0, max(dbd.values())+600, 100))
ax.xaxis.set_ticklabels([])
for (y, p), x  in zip(enumerate(pcnts), dbd.values()):
    ax.annotate("%.1f%%"%p, (x+80 ,y-0.125))
ax.xaxis.set_ticks_position('none')
ax.grid(False)
prettify(ax)
# ax.set_title('Databases Desired(wants to work with)' ,fontdict={'fontsize':20}, horizontalalignment='center')

plt.style.use('default')
plt.style.use('ggplot')
dev = {}
for x in df.DevType.dropna():
    for d in x.split(';'):
        if d!='nan':
            if d not in dev:
                dev[d] = 1   # create new instance
            else:
                dev[d] += 1  # increments that instance  
            
dev = {k:v for k,v in sorted(dev.items(), key=lambda x:x[1])}
pos = np.arange(0,len(dev.keys()), 1)
labels = [x for x in dev.keys()]
tot = df.DevType.count()
pcnts = [round(100*x/tot, 1) for x in dev.values()]

fig, ax = plt.subplots(figsize=(10,8))
ax.barh(labels, dev.values(), edgecolor='k')
ax.set_xticks(np.arange(0, max(dev.values())+900, 100))
ax.xaxis.set_ticklabels([])
for (y, p), x  in zip(enumerate(pcnts), dev.values()):
    ax.annotate("%.1f%%"%p, (x+50 ,y-0.25))
ax.xaxis.set_ticks_position('none')
ax.grid(False)
# ax.set_title('Developer\'s Type Of Work' ,fontdict={'fontsize':25}, horizontalalignment='right')
# fig.tight_layout()
prettify(ax)
plt.style.use('default')
plt.style.use('default')
dev_age, dev_num = {}, {}
df.DevType = df.DevType.astype('str')
for i in range(0,len(df)):
  age = df.loc[i].Age
  devt = df.loc[i].DevType
  if (not np.isnan(age)) and devt!='nan':
    for d in devt.split(';'):
      if d not in dev_age:
        dev_age[d] = 0
        dev_num[d] = 1
      else:
        dev_age[d] += age
        dev_num[d] += 1
for k in dev_age.keys():
  dev_age[k] = dev_age[k]/dev_num[k]

dev_age = {k:v for k,v in sorted(dev_age.items(), key=lambda x:x[1])}
bar = [float(x) for x in dev_age.values()]
dev = [str(x) for x in dev_age.keys()]

fig, ax = plt.subplots(figsize=(10,10))

ax.barh(dev, bar, edgecolor='k')
ax.grid(axis='x')
# ax.set_title('Mean Age by Developer Type',fontdict={'fontsize':20}, horizontalalignment='right')
plt.style.use('default')
plt.style.use('ggplot')

df.YearsCodePro.replace('Less than 1 year', '0.6', inplace=True)
df.YearsCodePro.replace('More than 50 years', '55', inplace=True)

for d in dev_num:
  dev_num[d] = 0
dev_yc = dev_num.copy()
df.YearsCodePro = df.YearsCodePro.astype('float')

for i in range(len(df.YearsCodePro)):
  yc = df.loc[i].YearsCodePro
  devt = df.loc[i].DevType
  if devt!='nan' and (not np.isnan(yc)):
    for d in devt.split(';'):
        dev_yc[d] += yc
        dev_num[d] += 1
for k in dev_yc.keys():
  dev_yc[k] = dev_yc[k] / dev_num[k]

dev_yc = {k:v for k,v in sorted(dev_yc.items(), key=lambda x:x[1])}
yc = [float(x) for x in dev_yc.values()]
dev = [x for x in dev_yc.keys()]

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(dev, yc, edgecolor='b' )
ax.grid(axis='y')
# ax.set_title('Years of Professional Coding BY Developer Type',fontdict={'fontsize':20}, horizontalalignment='right')
plt.style.use('default')
plt.style.use('ggplot')

dft = df[['LanguageWorkedWith', 'DevType']].dropna()
dft.LanguageWorkedWith.loc[0]
dl = {}

i = 0
for dev, lang in zip(dft.DevType, dft.LanguageWorkedWith):
  i+=1
  for d in dev.split(';'):
    if d not in dl and d!='nan':
      dl[d] = dict()
    if d!='nan':
      for l in lang.split(';'):
        if l not in dl[d]:
          dl[d][l] = 0
        else:
          dl[d][l] += 1
for d in dl.keys():
  temp = dl[d]
  temp = {k:v for k,v in sorted(temp.items(), key=lambda x:x[1])}
  temp = {k:round(100*v/sum(temp.values()),1) for k,v in temp.items()}
  dl[d] = temp


title = [n for n in dl.keys()]
nm1, nm2, nm3, n1, n2, n3 = [], [], [], [], [], []
for t in title:
    nm = [x for x in dl[t].values()]
    nam = [x for x in dl[t].keys()]
    nm1.append(nam[-1])
    nm2.append(nam[-2])
    nm3.append(nam[-3])
    n1.append(nm[-1])
    n2.append(nm[-2])
    n3.append(nm[-3])
n = np.arange(len(title))

fig, ax = plt.subplots(figsize=(10,19))
ax.barh(n, n1, height=0.25, align='edge')
ax.barh(n-0.2, n2, height=0.25, align='edge')
ax.barh(n-0.4, n3, height=0.25, align='edge')
ax.set_yticks(np.arange(len(title)))
ax.set_yticklabels([x for x in title])
for (y,_),x1, x2, x3, a1, a2, a3 in zip(enumerate(title), n1, n2, n3, nm1, nm2, nm3):
  ax.annotate(a1, (x1+0.1, y+0.05), fontsize=8.5)
  ax.annotate(a2, (x2+0.1, y-0.2), fontsize=8.5)
  ax.annotate(a3, (x3+0.1, y-0.4), fontsize=8.5)
ax.set_title('\nTop 3 Languages Worked With BY Developer Type')
prettify(ax)

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8,6))
sns.violinplot(df[df.WorkWeekHrs<100].WorkWeekHrs, inner='quartile',)
ax.set_xticks(np.arange(0,101,10))
# ax.set_title('Distribution of Hours Worked Weekly')
ax.grid(axis='x')

plt.style.use('default')
plt.style.use('seaborn')

dft = df.groupby(['Country']).WorkWeekHrs.mean()
cntr =['Pakistan','India','Turkey','France','Germany','Canada', 'China','United States','United Kingdom']
age = [dft[x] for x in cntr]

fig, ax = plt.subplots()
ax.barh(cntr, age)
ax.set_xlabel('Years')
ax.grid(axis='y')
# ax.set_title('Average Weekly Work Hours' ,fontdict={'fontsize':20}, horizontalalignment='center')
fig.tight_layout()

plt.style.use('default')
plt.style.use('ggplot')

dft = df.JobFactors.astype('str')
jf = {}
for x in dft.unique():
  for d in x.split(';'):
    if d not in jf and d!='nan':
      jf[d] = 0

for x in df.JobFactors.dropna():
  for d in x.split(';'):
    jf[d] += 1

tot = df.JobFactors.notna().sum()
jf = {k:v for k,v in sorted(jf.items(), key=lambda x:x[1])}
jfnum = [round(100*jf[k]/tot,1) for k in jf.keys()]
jfname = [k for k in jf.keys()]

fig, ax = plt.subplots(figsize=(10,10))
ax.barh(jfname, jfnum, edgecolor='k')
# ax.set_title('Most Important Job Factors While Looking For A Job',fontdict={'fontsize':20}, horizontalalignment='right')
ax.xaxis.set_ticklabels([])
ax.grid(False)
for y, p in enumerate(jfnum):
  ax.annotate("%.1f%%"%p, (p+0.2 , y), fontsize=12)
prettify(ax)
plt.style.use('default')
plt.style.use('ggplot')

dft = df.groupby(['NEWOvertime']).Respondent.count()
names = list(map(lambda x: x, dft.index))
nums = list(map(lambda x: dft[x], range(len(dft.index))))
pcnts = [round(100*x/sum(nums),1) for x in nums]

fig, ax = plt.subplots()
ax.barh(names, pcnts, edgecolor='k')
for y, p in enumerate(pcnts):
  ax.annotate("{}%".format(p), (p+0.5, y))
prettify(ax)
# ax.set_title('Overtime Routine')
plt.style.use('default')
plt.style.use('ggplot')

uns = {}
for d in df.NEWStuck.dropna():
  for x in d.split(';'):
    if x not in uns:
      uns[x] = 1
    else:
      uns[x] += 1
uns = {k:v for k,v in sorted(uns.items(), key=lambda x:x[1])}

names = [x[:x.find('(')-1]if x=='Visit another developer community (please name):' else x for x in uns.keys()]
pcnts = [round(100*x/df.NEWStuck.dropna().count(),1) for x in uns.values()]

fig, ax = plt.subplots()
ax.barh(names, pcnts)
for y, p in enumerate(pcnts):
  ax.annotate("%.1f%%"%p, (p+0.5, y-0.1))
prettify(ax)
# ax.set_title('What do you do when you get stuck?')
plt.style.use('default')
plt.style.use('ggplot')

dft = df.groupby(['NEWEdImpt']).Respondent.count()
pies = [dft[x] for x in dft.index]
labels = [dft.index[x] for x in range(dft.shape[0])]

fig, ax = plt.subplots()
ax.pie(pies, labels=labels)
# ax.set_title('Importance of Formal Education for career\n')
plt.style.use('default')
