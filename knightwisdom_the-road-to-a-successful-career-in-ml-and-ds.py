# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 24})



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings(action="ignore")
mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

qs = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
print(mcr.shape, qs.shape)
mcr.head()
for i in qs.columns:

    print(i,' : ',qs[i][0])
mcr['Q10'].value_counts()[:-1].plot(kind='bar', figsize=(15, 6), rot=90)
q10_ranking = []

q10_values = mcr['Q10']

print("Percentage of missing values: {} %".format(round(100*q10_values.isna().sum()/len(q10_values),2)))

q10_values = q10_values.fillna('0')

for i in range(1,len(mcr)):

    t=q10_values[i]

    if True:

        x10 = int(''.join(t.split('$')[-1].split('-')[-1].split(',')))

        q10_ranking.append(x10)

        #print(x10)

mcr_q10_ranking = mcr[1:].copy()

mcr_q10_ranking['max_income_level'] = q10_ranking

value_cnts = mcr_q10_ranking['max_income_level'].value_counts()

axx = (value_cnts.sort_index(ascending = True)[1:]).plot(kind='bar', figsize=(15, 6), rot=90)

axx.set_xlabel("Income group maximum: USD $")

axx.set_ylabel("# of Respondents")
q10_group = []

for i in range(len(q10_ranking)):

    if q10_ranking[i]>=150000:

        q10_group.append(3)

    elif q10_ranking[i]>=50000 and q10_ranking[i]<=149999:

        q10_group.append(2)

    elif q10_ranking[i]>=1 and q10_ranking[i]<=49999:

        q10_group.append(1)

    else:

        q10_group.append(0)
mcr_q10_ranking.head()
mcr_q10_ranking['q10_group'] = q10_group
mcr_q10_ranking.head()
value_cnts = mcr_q10_ranking['q10_group'].value_counts()

axx = (value_cnts.sort_index(ascending = True)[1:]).plot(kind='bar', figsize=(15, 6), rot=90)

axx.set_xlabel("Income group: USD $")

axx.set_ylabel("# of Respondents")

print('Group 3 is the highest earning group')
print('There are {} % of respondents who are earning more than 150,000$'.format(round(100*value_cnts[3]/len(mcr_q10_ranking),2)))

print('{} % of respondents are earning between 50,000$ and 149,999$'.format(round(100*value_cnts[2]/len(mcr_q10_ranking),2)))

print('{} % of respondents are earning < 50,000$'.format(round(100*value_cnts[1]/len(mcr_q10_ranking),2)))

print('{} % of respondents are not earning'.format(round(100*value_cnts[0]/len(mcr_q10_ranking),2)))
mcr_g3 = mcr_q10_ranking[mcr_q10_ranking['q10_group']==3]

len(mcr_g3)/len(mcr_q10_ranking)
#Reset_index

mcr_g3 = mcr_g3.reset_index().drop('index', axis=1)

mcr_g3.head()
for i in mcr_g3.isna().sum().index:

    print(i,' : ', mcr_g3[i].isna().sum())
# Time taken by the G3 people

duration_only = mcr_g3[mcr_g3.columns[0]]

import seaborn as sns

lst = pd.to_numeric(duration_only, errors='coerce')/60

lst = pd.DataFrame(lst)

lst.columns = ['Time from Start to Finish (minutes)']

lst = lst['Time from Start to Finish (minutes)']

sns.distplot(lst, bins = 5000).set(xlim=(0, 60))
print('There are {} total G3 respondents out of whom {} respondents have spent more than 10 minutes taking this survey and {} respondents really took more than an hour.'.format(lst.shape[0],lst[lst>10].shape[0],lst[lst>60].shape[0]))
def plotMcqHist(i):

    plt.rcParams.update({'font.size': 24})

    print(mcr_g3.columns[i], '  : ', mcr[mcr_g3.columns[i]][0])

    temp = mcr_g3[mcr_g3.columns[i]].value_counts()

    temp = temp.fillna(-10)

    plt.figure(figsize=(20,5))

    #plt.xlabel(mcr_g3.columns[i])

    plt.ylabel('# Respondents')

    plt.bar(list(temp.index),list(temp))

    

    plt.xticks(rotation=90)

    plt.show()

    

    print('='*50)
for i in enumerate(mcr_g3.columns):

    print(i)
lst1 = [1,2,4,5,6,8,9,10,20,21,48,55,95,116,117]
plotMcqHist(lst1[0])
# Reordering the plot about X axis

temp = mcr_g3[mcr_g3.columns[1]].value_counts()

#temp = temp.fillna(-10)

temp_list=[]

for i in range(len(temp)):

    temp_list.append(int(temp.index[i].split('-')[0].split('+')[0]))

tmp = pd.DataFrame(temp)

tmp['index'] = temp_list

tmp = tmp.sort_values(by=['index'], ascending = True)

plt.figure(figsize=(20,5))

#plt.xlabel(mcr_g3.columns[1])

plt.ylabel('# Respondents')

plt.bar(list(tmp.index),list(tmp['Q1']))



plt.xticks(rotation=90)

plt.show()

print('='*50)
print("There are {} % of G3 respondents who belong to 18 to 21 Age group".format(round((((mcr_g3[mcr_g3.columns[lst1[0]]]=='18-21')).sum()/len(mcr_g3))*100,2)))

print("There are {} % of G3 respondents who belong to 22 to 24 Age group".format(round((((mcr_g3[mcr_g3.columns[lst1[0]]]=='22-24')).sum()/len(mcr_g3))*100,2)))

print("There are {} % of G3 respondents who belong to 25 to 29 Age group".format(round((((mcr_g3[mcr_g3.columns[lst1[0]]]=='25-29')).sum()/len(mcr_g3))*100,2)))

print("There are {} % of G3 respondents who belong to 30 to 34 Age group".format(round((((mcr_g3[mcr_g3.columns[lst1[0]]]=='30-34')).sum()/len(mcr_g3))*100,2)))
plotMcqHist(lst1[1])
print("{} % of G3 respondents are Male".format(round((((mcr_g3[mcr_g3.columns[lst1[1]]]=='Male')).sum()/len(mcr_g3))*100,2)))

print("{} % of G3 respondents are Female".format(round((((mcr_g3[mcr_g3.columns[lst1[1]]]=='Female')).sum()/len(mcr_g3))*100,2)))
print("{} % of Total Male respondents belong to G3".format(round((((mcr_g3[mcr_g3.columns[lst1[1]]]=='Male')).sum()/((mcr[mcr.columns[lst1[1]]]=='Male')).sum())*100,2)))

print("{} % of Total Female respondents belong to G3".format(round((((mcr_g3[mcr_g3.columns[lst1[1]]]=='Female')).sum()/((mcr[mcr.columns[lst1[1]]]=='Female')).sum())*100,2)))
plotMcqHist(lst1[2])
print("{} % of high income group (Group 3) respondents belong to United States of America".format(round((mcr_g3[mcr_g3.columns[lst1[2]]]=='United States of America').sum()/(len(mcr_g3))*100,2)))

print("While {} % of high income group (Group 3) respondents belong to India".format(round((mcr_g3[mcr_g3.columns[lst1[2]]]=='India').sum()/(len(mcr_g3))*100,2)))
plotMcqHist(lst1[3])
print("{} % of high income group (Group 3) respondents have Bachelors as Highest level of Formal education".format(round((mcr_g3[mcr_g3.columns[lst1[3]]]=='Bachelor’s degree').sum()/(len(mcr_g3))*100,2)))

print("{} % of high income group (Group 3) respondents have Masters as Highest level of Formal education".format(round((mcr_g3[mcr_g3.columns[lst1[3]]]=='Master’s degree').sum()/(len(mcr_g3))*100,2)))

print("{} % of high income group (Group 3) respondents have Doctoral as Highest level of Formal education".format(round((mcr_g3[mcr_g3.columns[lst1[3]]]=='Doctoral degree').sum()/(len(mcr_g3))*100,2)))
plotMcqHist(lst1[4])
print("{} % of G3 respondents identify themselves as Data Scientists".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')).sum()/len(mcr_g3))*100,2)))

print("While {} % of total respondents identify themselves as Data Scientists".format(round((((mcr[mcr.columns[lst1[4]]]=='Data Scientist')).sum()/len(mcr))*100,2)))
print("{} % of G3 respondents identify themselves as Data Scientists and have Master's Degree".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')&(mcr_g3[mcr_g3.columns[lst1[3]]]=='Master’s degree')).sum()/len(mcr_g3))*100,2)))

print("{} % of G3 respondents identify themselves as Data Scientists and have Doctoral Degree".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')&(mcr_g3[mcr_g3.columns[lst1[3]]]=='Doctoral degree')).sum()/len(mcr_g3))*100,2)))



print("{} % of G3 respondents identify themselves as Data Scientists, have Master's Degree and belong to United States of America".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')&(mcr_g3[mcr_g3.columns[lst1[3]]]=='Master’s degree') & (mcr_g3[mcr_g3.columns[lst1[2]]]=='United States of America')).sum()/len(mcr_g3))*100,2)))

print("{} % of G3 respondents identify themselves as Data Scientists, have Doctoral Degree and belong to United States of America".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')&(mcr_g3[mcr_g3.columns[lst1[3]]]=='Doctoral degree') & (mcr_g3[mcr_g3.columns[lst1[2]]]=='United States of America')).sum()/len(mcr_g3))*100,2)))



print("{} % of G3 Male respondents identify themselves as Data Scientists, have Master's Degree and belong to United States of America".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')&(mcr_g3[mcr_g3.columns[lst1[3]]]=='Master’s degree') & (mcr_g3[mcr_g3.columns[lst1[2]]]=='United States of America')&(mcr_g3[mcr_g3.columns[lst1[1]]]=='Male')).sum()/len(mcr_g3))*100,2)))

print("{} % of G3 Male respondents identify themselves as Data Scientists, have Doctoral Degree and belong to United States of America".format(round((((mcr_g3[mcr_g3.columns[lst1[4]]]=='Data Scientist')&(mcr_g3[mcr_g3.columns[lst1[3]]]=='Doctoral degree') & (mcr_g3[mcr_g3.columns[lst1[2]]]=='United States of America')&(mcr_g3[mcr_g3.columns[lst1[1]]]=='Male')).sum()/len(mcr_g3))*100,2)))



plotMcqHist(lst1[5])
plotMcqHist(lst1[6])
plotMcqHist(lst1[7])
plotMcqHist(lst1[8])
print("If anyone follows the recommendations from this notebook, they have a {} % probability of earning an income >$500,000 and {} % of propbability of earning an income greater than $250,000 ".format(round(((mcr_g3[mcr_g3.columns[lst1[8]]]=='> $500,000').sum()/len(mcr_g3))*100,2), round((((mcr_g3[mcr_g3.columns[lst1[8]]]=='250,000-299,999')|(mcr_g3[mcr_g3.columns[lst1[8]]]=='300,000-500,000')|(mcr_g3[mcr_g3.columns[lst1[8]]]=='> $500,000')).sum()/len(mcr_g3))*100,2)))
plotMcqHist(lst1[9])
print("You are {}% likely to spend more than 100,000$ for cloud computing products.".format(round(((mcr_g3[mcr_g3.columns[lst1[9]]]=='> $100,000 ($USD)').sum()/len(mcr_g3))*100,2)))
plotMcqHist(lst1[10])
print("You are {}% likely to use JupyterLab or RStudio type local development environments at work or school".format(round(((mcr_g3[mcr_g3.columns[lst1[10]]]=='Local development environments (RStudio, JupyterLab, etc.)').sum()/len(mcr_g3))*100,2)))
plotMcqHist(lst1[11])
# Reordering the plot about X axis

temp = mcr_g3[mcr_g3.columns[lst1[11]]].value_counts()

temp_list=[]



tmp = pd.DataFrame(temp)

tmp['index'] = [4,5,6,3,2,1,0]

print(tmp)

tmp = tmp.sort_values(by=['index'], ascending = True)

plt.figure(figsize=(20,5))

#plt.xlabel(mcr_g3.columns[1])

plt.ylabel("# Respondents")

plt.bar(list(tmp.index),list(tmp['Q15']))



plt.xticks(rotation=90)

plt.show()

print('='*50)
print("You are {}% likely to have 10+ years of data analysis related coding experience if you are earning more than 150,000$.".format(round((((mcr_g3[mcr_g3.columns[lst1[11]]]=='10-20 years')|(mcr_g3[mcr_g3.columns[lst1[11]]]=='20+ years')).sum()/len(mcr_g3))*100,2)))
plotMcqHist(lst1[12])
print("Hahaa! {}% respondents earning more than 150,000$ recommend to learn Python first.".format(round((((mcr_g3[mcr_g3.columns[lst1[12]]]=='Python')).sum()/len(mcr_g3))*100,2)))

print("{}% respondents recommending to learn Python first are earning more than 150,000$.".format(round((((mcr_g3[mcr_g3.columns[lst1[12]]]=='Python')).sum()/(mcr[mcr.columns[lst1[12]]]=='Python').sum())*100,2)))
plotMcqHist(lst1[14])
print('Q13', '  : ', mcr[mcr_g3.columns[35]][0].split('-')[0])

temp=[]

ind=[]

for i in [i for i in range(35,45)]:

    temp.append(mcr_g3[mcr_g3.columns[i]].value_counts()[0])

    ind.append(mcr_g3[mcr_g3.columns[i]].value_counts().index[0])

#print(ind,temp)

lst_tmp = pd.DataFrame(temp)

lst_tmp.index=ind

#print(lst_tmp)

plt.figure(figsize=(20,5))

#plt.xlabel('Q13')

plt.ylabel('# Respondents')

plt.bar(list(lst_tmp.index),list(lst_tmp[0]))

plt.xticks(rotation=90)

plt.show()

print('='*50)
print('Q25', '  : ', mcr[mcr_g3.columns[118]][0].split('-')[0])

temp=[]

ind=[]

for i in [i for i in range(118,129)]:

    temp.append(mcr_g3[mcr_g3.columns[i]].value_counts()[0])

    ind.append(mcr_g3[mcr_g3.columns[i]].value_counts().index[0])

#print(ind,temp)

lst_tmp = pd.DataFrame(temp)

lst_tmp.index=ind

#print(lst_tmp)

plt.figure(figsize=(20,5))

#plt.xlabel('Q25')

plt.ylabel('# Respondents')

plt.bar(list(lst_tmp.index),list(lst_tmp[0]))

plt.xticks(rotation=90)

plt.show()

print('='*50)
print('Top 3 ML algorithms you need to be equiped with are: \n',ind[:3])
print('Q29', '  : ', mcr[mcr_g3.columns[168]][0].split('-')[0])

temp=[]

ind=[]

for i in [i for i in range(169,178)]:

    temp.append(mcr_g3[mcr_g3.columns[i]].value_counts()[0])

    ind.append(mcr_g3[mcr_g3.columns[i]].value_counts().index[0])

#print(ind,temp)

lst_tmp = pd.DataFrame(temp)

lst_tmp.index=ind

#print(lst_tmp)

plt.figure(figsize=(20,5))

#plt.xlabel('Q29')

plt.ylabel('# Respondents')

plt.bar(list(lst_tmp.index),list(lst_tmp[0]))

plt.xticks(rotation=90)

plt.show()

print('='*50)
print('Q30', '  : ', mcr[mcr_g3.columns[168]][0].split('-')[0])

temp=[]

ind=[]

for i in [i for i in range(181,192)]:

    temp.append(mcr_g3[mcr_g3.columns[i]].value_counts()[0])

    ind.append(mcr_g3[mcr_g3.columns[i]].value_counts().index[0])

#print(ind,temp)

lst_tmp = pd.DataFrame(temp)

lst_tmp.index=ind

#print(lst_tmp)

plt.figure(figsize=(20,5))

#plt.xlabel('Q30')

plt.ylabel('# Respondents')

plt.bar(list(lst_tmp.index),list(lst_tmp[0]))

plt.xticks(rotation=90)

plt.show()

print('='*50)
mcr
mcr_q10_ranking.head()
# If respondent took more than 10 minutes: Category 0, else category 1

time_encoding = []

for i in mcr_q10_ranking[mcr_q10_ranking.columns[0]]:

    if int(i) >600:

        time_encoding.append(0)

    elif int(i)<=600:

        time_encoding.append(1)

X = mcr_q10_ranking.copy()



X = X.reset_index()

Y = mcr_q10_ranking['q10_group']

X = X.drop([mcr_q10_ranking.columns[0],mcr_q10_ranking.columns[-2], 'index', 'q10_group', 'Q10'], axis=1)

X.head()
# As there will be no new category we will proceed with Label Encoder. Data Leakage is not a problem here.

# All columns are categorical



from sklearn import preprocessing

# Label Encoding

for f in X.columns:

    lbl = preprocessing.LabelEncoder()

    X[f] = lbl.fit_transform(list(X[f].values))
X.head()
Y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y)

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size = 0.2, stratify = y_train)

print(X_train.shape, y_train.shape)

print(X_cv.shape, y_cv.shape)

print(X_test.shape, y_test.shape)
import xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=500,

                        n_jobs=4,

                        max_depth=10,

                        learning_rate=0.05,

                        subsample=0.9,

                        colsample_bytree=0.9)



clf.fit(X_train, y_train)
predict_train = clf.predict(X_train)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_train,predict_train))

print(accuracy_score(y_train,predict_train))
predict_cv = clf.predict(X_cv)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_cv,predict_cv))

print(accuracy_score(y_cv,predict_cv))
predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test,predict))

print(accuracy_score(y_test,predict))
# plot feature importance

from xgboost import plot_importance

plot_importance(clf, max_num_features=10)

plt.show()
print(mcr.columns[4], mcr[mcr.columns[4]][0])

print(mcr.columns[1], mcr[mcr.columns[1]][0])

print(mcr.columns[6], mcr[mcr.columns[6]][0])

print(mcr.columns[9], mcr[mcr.columns[9]][0])

print(mcr.columns[8], mcr[mcr.columns[8]][0])

print(mcr.columns[10], mcr[mcr.columns[10]][0])