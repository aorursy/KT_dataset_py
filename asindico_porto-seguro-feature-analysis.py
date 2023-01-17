# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
tdf = pd.read_csv('../input/test.csv')
import re

def istype(name,_type):

    match = re.search('^.*'+_type+'.*$',name)

    if match:

        return True

    else:

        return False

    

def notBinOrCat(name):

    match = re.search('^.*bin.*$',name)

    if match:

        return False

    else:

        match = re.search('^.*cat.*$',name)

        if match:

            return False

        else:

            return True
ind_cols = [col for col in df.columns if istype(col,'ind')]

reg_cols = [col for col in df.columns if istype(col,'reg')]

car_cols = [col for col in df.columns if istype(col,'car')]

calc_cols= [col for col in df.columns if istype(col,'calc')]



ind_cat = [col for col in ind_cols if istype(col,'cat')]

reg_cat = [col for col in reg_cols if istype(col,'cat')]

car_cat = [col for col in car_cols if istype(col,'cat')]

calc_cat= [col for col in calc_cols if istype(col,'cat')]



ind_bin = [col for col in ind_cols if istype(col,'bin')]

reg_bin = [col for col in reg_cols if istype(col,'bin')]

car_bin = [col for col in car_cols if istype(col,'bin')]

calc_bin= [col for col in calc_cols if istype(col,'bin')]



ind_con = [col for col in ind_cols if not (istype(col,'bin') or istype(col,'cat'))]

reg_con = [col for col in reg_cols if not (istype(col,'bin') or istype(col,'cat'))]

car_con = [col for col in car_cols if not (istype(col,'bin') or istype(col,'cat'))]

calc_con= [col for col in calc_cols if not (istype(col,'bin') or istype(col,'cat'))]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

rdf = df.sample(100000)

clf = RandomForestClassifier(n_estimators = 50,random_state=0)

clf.fit(rdf[car_con],rdf['target'])

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

for i in range(len(indices)):

    print (car_con[indices[i]],importances[indices[i]])
clf = RandomForestClassifier(n_estimators = 50,random_state=0)

clf.fit(rdf[reg_con],rdf['target'])

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

for i in range(len(indices)):

    print (reg_con[indices[i]],importances[indices[i]])
clf = RandomForestClassifier(n_estimators = 50,random_state=0)

clf.fit(rdf[calc_con],rdf['target'])

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

for i in range(len(indices)):

    print (calc_con[indices[i]],importances[indices[i]])
clf = RandomForestClassifier(n_estimators = 50,random_state=0)

cf  = reg_con+car_con+calc_con

clf.fit(rdf[cf],rdf['target'])

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

for i in range(len(indices)):

    print (cf[indices[i]],importances[indices[i]])
f,axarr = plt.subplots(3,2,figsize=(15,24))

hist, bin_edges = np.histogram(df[df['ps_reg_03']!=-1]['ps_reg_03'], density=False)

bin_edges = [float("{0:.2f}".format(x)) for x in bin_edges]

sns.barplot(x=bin_edges[0:len(hist)],y=hist,ax=axarr[0][0])

axarr[0][0].set_title('ps_reg_03 histogram')

df['ps_reg_03_cat'] = np.digitize(df['ps_reg_03'], bin_edges)

for tick in axarr[0][0].get_xticklabels():

        tick.set_rotation(90)

        

hist, bin_edges = np.histogram(df[df['ps_car_13']!=-1]['ps_car_13'], density=False)

bin_edges = [float("{0:.2f}".format(x)) for x in bin_edges]

sns.barplot(x=bin_edges[0:len(hist)],y=hist,ax=axarr[0][1])

axarr[0][1].set_title('ps_car_13 histogram')

df['ps_car_13_cat'] = np.digitize(df['ps_car_13'], bin_edges)

for tick in axarr[0][1].get_xticklabels():

        tick.set_rotation(90)

        

hist, bin_edges = np.histogram(df[df['ps_car_14']!=-1]['ps_car_14'], density=False)

bin_edges = [float("{0:.2f}".format(x)) for x in bin_edges]

sns.barplot(x=bin_edges[0:len(hist)],y=hist,ax=axarr[1][1])

axarr[1][1].set_title('ps_car_14 histogram')

df['ps_car_14_cat'] = np.digitize(df['ps_car_14'], bin_edges)

for tick in axarr[1][1].get_xticklabels():

        tick.set_rotation(90)



hist, bin_edges = np.histogram(df[df['ps_calc_14']!=-1]['ps_calc_14'], density=False)

bin_edges = [float("{0:.2f}".format(x)) for x in bin_edges]

sns.barplot(x=bin_edges[0:len(hist)],y=hist,ax=axarr[1][0])

axarr[1][0].set_title('ps_calc_14 histogram')

df['ps_calc_14_cat'] = np.digitize(df['ps_calc_14'], bin_edges)

for tick in axarr[1][0].get_xticklabels():

        tick.set_rotation(90)



hist, bin_edges = np.histogram(df[df['ps_calc_10']!=-1]['ps_calc_10'], density=False)

bin_edges = [float("{0:.2f}".format(x)) for x in bin_edges]

sns.barplot(x=bin_edges[0:len(hist)],y=hist,ax=axarr[2][0])

axarr[2][0].set_title('ps_calc_10 histogram')

df['ps_calc_10_cat'] = np.digitize(df['ps_calc_10'], bin_edges)

for tick in axarr[2][0].get_xticklabels():

        tick.set_rotation(90)

        

hist, bin_edges = np.histogram(df[df['ps_calc_11']!=-1]['ps_calc_11'], density=False)

bin_edges = [float("{0:.2f}".format(x)) for x in bin_edges]

sns.barplot(x=bin_edges[0:len(hist)],y=hist,ax=axarr[2][1])

axarr[2][1].set_title('ps_calc_11 histogram')

df['ps_calc_11_cat'] = np.digitize(df['ps_calc_11'], bin_edges)

for tick in axarr[2][1].get_xticklabels():

        tick.set_rotation(90)

        

plt.show()
df['ps_reg_03_cat'].values[0:20]
f,axarr = plt.subplots(1,2,figsize=(15,6))

sns.distplot(df['ps_car_13'],ax=axarr[0])

sns.distplot(np.log(df['ps_car_13']),ax=axarr[1])

plt.show()


f,axarr = plt.subplots(1,2,figsize=(15,6))

null_ind_hist = (df[ind_cat]==-1).sum()

sns.barplot(x=null_ind_hist.index, y= null_ind_hist.values,ax=axarr[0])

null_car_hist = (df[car_cat]==-1).sum()

sns.barplot(x=null_car_hist.index, y= null_car_hist.values,ax=axarr[1])

plt.xticks(rotation=90)

plt.show()
f,axarr = plt.subplots(1,2,figsize=(15,6))

null_ind_hist = (tdf[ind_cat]==-1).sum()

sns.barplot(x=null_ind_hist.index, y= null_ind_hist.values,ax=axarr[0])

null_car_hist = (tdf[car_cat]==-1).sum()

sns.barplot(x=null_car_hist.index, y= null_car_hist.values,ax=axarr[1])

plt.xticks(rotation=90)

plt.show()
print('train is ',len(df),'while test is',len(tdf))
from sklearn.feature_selection import chi2

for i in ind_cat:

    valid = df[df[i]!=-1]

    print(i,':',chi2(np.asarray(valid[i]).reshape(-1,1), valid['target'])[0][0])
from sklearn.feature_selection import chi2

for i in car_cat:

    valid = df[df[i]!=-1]

    print(i,':',chi2(np.asarray(valid[i]).reshape(-1,1), valid['target'])[0][0])
for i in ind_cat:

    for j in ind_cat:

        valid = df[df[i]!=-1]

        valid2 = df[df[j]!=-1]

        print(i,'|',j,':',chi2(np.asarray(valid[i]).reshape(-1,1),

                               np.asarray(valid[j]))[0])
for i in car_cat:

    for j in car_cat:

        valid = df[df[i]!=-1]

        valid2 = df[df[j]!=-1]

        print(i,'|',j,':',chi2(np.asarray(valid[i]).reshape(-1,1),

                               np.asarray(valid[j]))[0])