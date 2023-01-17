import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir('../input'))
df = pd.read_csv('../input/Salaries.csv')
df.head()
df.info()
df.describe()
toCorrect = list(df.columns[3:7])
# status columns has no float type, lets skip it
# toCorrect.append(df.columns[12])
toCorrect
ldict = {}
for i in toCorrect:
    ldict[i] = set()
    for j in df[i].unique():
        ldict[i].add(type(j))
ldict
ldict = {}
temp = {}
for i in toCorrect:
    ldict[i] = set()
    count = 0
    temp[i] = []
    for j in df[i].unique():
        if(type(j) == str):
            ldict[i].add(j)
            # limit as list would go long as in version 3
            if count < 5:
                temp[i].append(j)
                count += 1
temp
import re
patt = re.compile("[A-Z]+.*",re.IGNORECASE)
for i in list(ldict.keys()):
    print(i,end=" = [")
    for j in ldict[i]:
        if patt.match(j):
            print(j,end=", ")
    print("]")
df[(df['BasePay'] == 'Not Provided') | (df['OvertimePay'] == 'Not Provided') | (df['OtherPay'] == 'Not Provided') | 
   (df['Benefits'] == 'Not Provided')]
df = df[~df['Id'].isin([148647,148651,148652,148653])]
df.head()
for i in toCorrect:
    df[i] = pd.Series(map(lambda l:np.float64(l), df[i]))
# running previous piece of code to check type
ldict = {}
for i in toCorrect:
    ldict[i] = set()
    for j in df[i].unique():
        ldict[i].add(type(j))
ldict
df.isnull().sum()
del df['Notes']
del df['Status']
df[df['BasePay'].isnull()].head(2)
df[df['Benefits'].isnull()].head(2)
df.fillna(value=0,inplace=True)
df.describe()
toCorrect.extend(['TotalPay', 'TotalPayBenefits'])
for i in toCorrect:
    df[i] = df[i].apply(lambda l: np.float64(0) if l < 0 else l)
df.describe()
df[df['JobTitle'].apply(lambda l: ((l.upper().find('POLICE DEPARTMENT') != -1)) | (l.upper().find('POLICE OFFICER') != -1) | (l.upper() == 'CHIEF OF POLICE'))]['JobTitle'].unique()
df[df['JobTitle'].apply(lambda l: l.upper() == 'TRANSIT OPERATOR')]['JobTitle'].unique()
# RE to match string ending with <any text><space><numbers> 
patt = re.compile(".* [0-9]+$")

# replace numbers with roman equivalent
def i2r(n):
    roman = ''
    d = {1000 : 'M', 900 : 'CM', 500 : 'D', 400 : 'CD', 100 : 'C', 90 : 'XC', 50 : 'L', 40 : 'XL', 10 : 'X', 9 : 'IX', 5 : 'V', 4 : 'IV', 1 : 'I'}
    while n > 0:
        for i in d.keys():
            while n >= i:
                roman += d[i]
                n -= i
    return roman

def norm(l):
    # convert to uppercase
    l = l.upper()
    # to convert to roman
    if patt.match(l):
        i = 1
        while True:
            if l[-i:].isdecimal():
                i += 1
            else:
                break
        l = l[:-i] + ' ' + i2r(int(l[-i:]))
    return l 

print(norm('Transit Operator 12'))
df['JobTitle'] = df['JobTitle'].apply(norm)
# check for previous duplication
df[df['JobTitle'].apply(lambda l: ((l.upper().find('POLICE DEPARTMENT') != -1)) | (l.upper().find('POLICE OFFICER') != -1) | (l.upper() == 'CHIEF OF POLICE'))]['JobTitle'].unique()
df['EmployeeName'] = df['EmployeeName'].apply(str.upper)
df.head()
sns.countplot(df['Year'], palette='magma')
jobcount = df['JobTitle'].value_counts()[:20]
sns.barplot(x=jobcount, y=jobcount.keys())
fig, ax = plt.subplots(4, figsize = (8, 13))
for i in range(4):
    jcount = df[df['Year'] == (2011 + i)]['JobTitle'].value_counts()[:10]
    sns.barplot(x=jcount, y = jcount.keys(),ax = ax[i])
    ax[i].set_title(str(2011+i))
    ax[i].set_xlabel(' ')
    ax[i].set_xlim(0,2500)
param = ['BasePay', 'Benefits', 'TotalPay']
def by_year(emp_list):
    d = df[df['JobTitle'].isin(emp_list)].groupby(['JobTitle', 'Year']).mean().reset_index()
    for i in range(3):
        splot = sns.factorplot(data = d, x = param[i], y = 'JobTitle', hue = 'Year', kind = 'bar', size = len(emp_list) * 2).set(title = param[i])
        #splot = sns.catplot(data = d, x = param[i], y = 'JobTitle', hue = 'Year', kind = 'bar', aspect = len(emp_list) / 2.5, height = len(emp_list) * 1.5).set(title = param[i])
top5s = df['JobTitle'].value_counts().keys()[:5]
by_year(top5s)
def dist_by_year(emp):
    fig, ax = plt.subplots(3, 1, figsize = (15,13))
    for i in range(3):
        sns.violinplot(data = df[df['JobTitle'] == emp], x = 'Year', y = param[i], ax = ax[i]).set(title = param[i])

dist_by_year('TRANSIT OPERATOR')
def dist_among_job(emp_list):
    fig1, ax1 = plt.subplots(4, 1, figsize = (16,13))
    fig2, ax2 = plt.subplots(4, 1, figsize = (16,13))
    fig3, ax3 = plt.subplots(4, 1, figsize = (16,13))
    for i in range(4):
        sns.violinplot(data = df[(df['JobTitle'].isin(emp_list)) & (df['Year'] == (2011 + i))], x = 'JobTitle', y = 'BasePay', ax = ax1[i])
    for i in range(4):
        sns.violinplot(data = df[(df['JobTitle'].isin(emp_list)) & (df['Year'] == (2011 + i))], x = 'JobTitle', y = 'Benefits', ax = ax2[i])
    for i in range(4):
        sns.violinplot(data = df[(df['JobTitle'].isin(emp_list)) & (df['Year'] == (2011 + i))], x = 'JobTitle', y = 'TotalPay', ax = ax3[i])
    ax1[0].set(title='BasePay - 2011-14')
    ax2[0].set(title='Benefits - 2011-14')
    ax3[0].set(title='TotalPay - 2011-14')
    
dist_among_job(top5s)
def large_dist_among_job(emp_list):
    fig1, ax1 = plt.subplots(4, 1, figsize = (16,13))
    fig2, ax2 = plt.subplots(4, 1, figsize = (16,13))
    fig3, ax3 = plt.subplots(4, 1, figsize = (16,13))
    
    for i in range(4):
        for j in range(len(emp_list)):
            sns.distplot(df[df['JobTitle'] == emp_list[j]]['BasePay'], hist = False, label = emp_list[j], ax = ax1[i])
            
    for i in range(4):
        for j in range(len(emp_list)):
            sns.distplot(df[df['JobTitle'] == emp_list[j]]['Benefits'], hist = False, label = emp_list[j], ax = ax2[i])
            
    for i in range(4):
        for j in range(len(emp_list)):
            sns.distplot(df[df['JobTitle'] == emp_list[j]]['TotalPay'], hist = False, label = emp_list[j], ax = ax3[i])
            
    ax1[0].set(title='BasePay - 2011-14')
    ax2[0].set(title='Benefits - 2011-14')
    ax3[0].set(title='TotalPay - 2011-14')
    
large_dist_among_job(df[df['JobTitle'].apply(lambda l: ((l.upper().find('POLICE OFFICER') != -1)) | (l.upper().find('CHIEF OF POLICE') != -1))]['JobTitle'].unique()[:10])
