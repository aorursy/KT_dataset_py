                                                                                                                # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df=pd.read_csv("../input/bouts_out_new.csv")

# Any results you write to the current directory are saved as output.
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df1=df.drop(['judge1_B','judge1_A','judge3_B','judge3_A','judge2_B','judge2_A'],axis=1)
df2= df1[np.isfinite(df1['reach_A'])&np.isfinite(df1['reach_B'])
         &np.isfinite(df1['weight_B'])&np.isfinite(df1['weight_A'])
         &np.isfinite(df1['age_A'])&np.isfinite(df1['age_B'])
        &np.isfinite(df1['height_A'])&np.isfinite(df1['height_B'])
        &pd.notnull(df1.stance_A)]
print(df2.columns.values)
print(df.stance_A.head())
total_not_null = df2.notnull().sum()
total = df2.isnull().sum()
percent = (df2.isnull().sum()*100/df2.isnull().count())
missing_data = pd.concat([total, percent,total_not_null], axis=1, keys=['Total', 'Percent','Total_nn'])
missing_data.head(20)
print(df2.age_A.head())
from matplotlib import pyplot
from matplotlib.pyplot import figure



x_age = df2.loc[:,'age_A'].values
y_age = df2.loc[:,'age_B'].values

bins = np.linspace(15, 40,100)
pyplot.subplots_adjust(hspace=1.4)

f, axs = pyplot.subplots(figsize=(15,15))
pyplot.subplot(411)
pyplot.title('Age')
pyplot.hist(x_age, bins, alpha=0.5, label='age A', color='red')
pyplot.hist(y_age, bins, alpha=0.5, label='age B', color='green')
pyplot.legend(loc='upper left')

pyplot.subplot(412)
diff_age=x_age-y_age
bins = np.linspace(-25, 25,100)
pyplot.title('Age Diff')
pyplot.hist(diff_age, bins, alpha=0.5, label='age A - age B', color='red')
pyplot.legend(loc='upper left')
pyplot.subplot(413)
pyplot.title('Height')
x_height = df2.loc[:,'height_A'].values
y_height = df2.loc[:,'height_B'].values
bins = np.linspace(150, 200, 40)
pyplot.hist(x_height, bins, alpha=0.5, label='height A', color='red')
pyplot.hist(y_height, bins, alpha=0.5, label='height B', color='green')
pyplot.legend(loc='upper left')
diff_height=x_height-y_height
bins = np.linspace(-25, 25,100)
pyplot.subplot(414)
pyplot.title('Height Diff')
pyplot.hist(diff_height, bins, alpha=0.5, label='height A - height B', color='red')
pyplot.legend(loc='upper left')
pyplot.show()

print('Mean value of age A ',df2['age_A'].mean())
print('Mean value of age A ',df2['age_B'].mean())
print('Mean value of height ',df2['height_A'].mean())
print('Mean value of height ',df2['height_B'].mean())
df2['reach_A'].head()
x_reach = df2.loc[:,'reach_A'].values
y_reach = df2.loc[:,'reach_B'].values

bins = np.linspace(150, 200,100)
pyplot.subplots_adjust(hspace=.4)
f, axs = pyplot.subplots(figsize=(15,15))
pyplot.subplot(211)
pyplot.title('reach')
pyplot.hist(x_reach, bins, alpha=0.5, label='reach A', color='red')
pyplot.hist(y_reach, bins, alpha=0.5, label='reach B', color='green')
pyplot.legend(loc='upper left')
diff_reach=x_reach-y_reach
bins = np.linspace(-25, 25,100)
pyplot.subplot(212)
pyplot.title('reach Diff')
pyplot.hist(diff_reach, bins, alpha=0.5, label='age A - age B', color='red')
pyplot.legend(loc='upper left')
pyplot.show()
df2 = pd.get_dummies(df2, prefix='stance_A_', columns=['stance_A'])
df2 = pd.get_dummies(df2, prefix='stance_B_', columns=['stance_B'])
print(df2.columns.values)
print(df2.columns.values)
df2['result'].head(20)
winA=df2[df2['result']=='win_A']
winB=df2[df2['result']=='win_B']
x_age = winA.loc[:,'age_A'].values
y_age = winA.loc[:,'age_B'].values

bins = np.linspace(15, 40,100)
pyplot.subplots_adjust(hspace=.4)
f, axs = pyplot.subplots(figsize=(15,15))
pyplot.subplot(411)
pyplot.title('Age')
pyplot.hist(x_age, bins, alpha=0.5, label='age A', color='red')
pyplot.hist(y_age, bins, alpha=0.5, label='age B', color='green')
pyplot.legend(loc='upper left')
diff_age=x_age-y_age
bins = np.linspace(-30, 30,100)
pyplot.subplot(412)
pyplot.title('Age Diff')
pyplot.hist(diff_age, bins, alpha=0.5, label='age A - age B', color='red')
pyplot.legend(loc='upper left')
pyplot.subplot(413)
pyplot.title('Height')
x_height = winA.loc[:,'height_A'].values
y_height = winA.loc[:,'height_B'].values
bins = np.linspace(150, 200, 40)
pyplot.hist(x_height, bins, alpha=0.5, label='height A', color='red')
pyplot.hist(y_height, bins, alpha=0.5, label='height B', color='green')
pyplot.legend(loc='upper left')
diff_height=x_height-y_height
bins = np.linspace(-30, 30,100)
pyplot.subplot(414)
pyplot.title('Height Diff')
pyplot.hist(diff_height, bins, alpha=0.5, label='HD', color='red')
pyplot.legend(loc='upper left')
pyplot.show()
x_reach = winA.loc[:,'reach_A'].values
y_reach = winA.loc[:,'reach_B'].values

bins = np.linspace(150, 200,100)
pyplot.subplots_adjust(hspace=.4)
f, axs = pyplot.subplots(figsize=(15,15))
pyplot.subplot(211)
pyplot.title('reach')
pyplot.hist(x_reach, bins, alpha=0.5, label='reach A', color='red')
pyplot.hist(y_reach, bins, alpha=0.5, label='reach B', color='green')
pyplot.legend(loc='upper left')
diff_reach=x_reach-y_reach
bins = np.linspace(-25, 25,100)
pyplot.subplot(212)
pyplot.title('reach Diff')
pyplot.hist(diff_reach, bins, alpha=0.5, label='age A - age B', color='red')
pyplot.legend(loc='upper left')
pyplot.show()
print('Mean value of age A ',winA['age_A'].mean())
print('Mean value of age B ',winA['age_B'].mean())
print('Mean value of height A',winA['height_A'].mean())
print('Mean value of height B',winA['height_B'].mean())
print('Mean value of reach A',winA['reach_A'].mean())
print('Mean value of reach B',winA['reach_B'].mean())
x_age = winB.loc[:,'age_A'].values
y_age = winB.loc[:,'age_B'].values

bins = np.linspace(15, 40,20)
pyplot.subplots_adjust(hspace=.4)
f, axs = pyplot.subplots(figsize=(15,15))
pyplot.subplot(411)
pyplot.title('Age')
pyplot.hist(x_age, bins, alpha=0.5, label='age A', color='red')
pyplot.hist(y_age, bins, alpha=0.5, label='age B', color='green')
pyplot.legend(loc='upper left')
diff_age=x_age-y_age
bins = np.linspace(-30, 30,60)
pyplot.subplot(412)
pyplot.title('Age Diff')
pyplot.hist(diff_age, bins, alpha=0.5, label='age A - age B', color='red')
pyplot.legend(loc='upper left')
pyplot.subplot(413)
pyplot.title('Height')
x_height = winB.loc[:,'height_A'].values
y_height = winB.loc[:,'height_B'].values
bins = np.linspace(150, 200, 40)
pyplot.hist(x_height, bins, alpha=0.5, label='height A', color='red')
pyplot.hist(y_height, bins, alpha=0.5, label='height B', color='green')
pyplot.legend(loc='upper left')
diff_height=x_height-y_height
bins = np.linspace(-30, 30,50)
pyplot.subplot(414)
pyplot.title('Height Diff')
pyplot.hist(diff_height, bins, alpha=0.5, label='HD', color='red')
pyplot.legend(loc='upper left')
pyplot.show()
print('Mean value of age A ',winB['age_A'].mean())
print('Mean value of age B ',winB['age_B'].mean())
print('Mean value of height A',winB['height_A'].mean())
print('Mean value of height B',winB['height_B'].mean())
print('Mean value of reach A',winB['reach_A'].mean())
print('Mean value of reach B',winB['reach_B'].mean())
KO_A=winA[winA['decision']=='KO']
KO_B=winB[winB['decision']=='KO']
print('Mean value of age A ',KO_A['age_A'].mean())
print('Mean value of age B ',KO_A['age_B'].mean())
print('Mean value of height A',KO_A['height_A'].mean())
print('Mean value of height B',KO_A['height_B'].mean())
print('Mean value of reach A',KO_A['reach_A'].mean())
print('Mean value of reach B',KO_A['reach_B'].mean())
print('Mean value of age A ',KO_B['age_A'].mean())
print('Mean value of age B ',KO_B['age_B'].mean())
print('Mean value of height A',KO_B['height_A'].mean())
print('Mean value of height B',KO_B['height_B'].mean())
print('Mean value of reach A',KO_B['reach_A'].mean())
print('Mean value of reach B',KO_B['reach_B'].mean())
UD_A=winA[winA['decision']=='UD']
UD_B=winB[winB['decision']=='UD']
print('Mean value of age A ',UD_A['age_A'].mean())
print('Mean value of age B ',UD_A['age_B'].mean())
print('Mean value of height A',UD_A['height_A'].mean())
print('Mean value of height B',UD_A['height_B'].mean())
print('Mean value of reach A',UD_A['reach_A'].mean())
print('Mean value of reach B',UD_A['reach_B'].mean())
print('Mean value of age A ',UD_B['age_A'].mean())
print('Mean value of age B ',UD_B['age_B'].mean())
print('Mean value of height A',UD_B['height_A'].mean())
print('Mean value of height B',UD_B['height_B'].mean())
print('Mean value of reach A',UD_B['reach_A'].mean())
print('Mean value of reach B',UD_B['reach_B'].mean())
A1=KO_A['age_A']-KO_A['age_B']
A2=KO_B['age_B']-KO_B['age_A']
H1=KO_A['height_A']-KO_A['height_B']
H2=KO_B['height_B']-KO_B['height_A']
R1=KO_A['reach_A']-KO_A['reach_B']
R2=KO_B['reach_B']-KO_B['reach_A']
print('Age difference ',A1.mean())
print('Age difference ',A2.mean())
print('Height difference ',H1.mean())
print('Height difference ',H2.mean())
print('Reach difference ',R1.mean())
print('Reach difference ',R1.mean())
x_age = A1.values
y_age = A2.values

bins = np.linspace(-20, 20,40)
pyplot.subplots_adjust(hspace=.4)
f, axs = pyplot.subplots(figsize=(15,15))
pyplot.subplot(311)
pyplot.title('Age')
pyplot.hist(x_age, bins, alpha=0.5, label='A1', color='red')
pyplot.hist(y_age, bins, alpha=0.5, label='A2', color='green')
pyplot.legend(loc='upper left')

bins = np.linspace(-30, 30,20)
x_h = H1.values
y_h = H2.values

pyplot.subplot(312)
pyplot.title('Height')
pyplot.hist(x_h, bins, alpha=0.5, label='H1', color='red')
pyplot.hist(y_h, bins, alpha=0.5, label='H2', color='green')
pyplot.legend(loc='upper left')

pyplot.subplot(313)
pyplot.title('Reach')
x_R = R1.values
y_R = R2.values
bins = np.linspace(-30, 30, 15)
pyplot.hist(x_R, bins, alpha=0.5, label='R1', color='red')
pyplot.hist(y_R, bins, alpha=0.5, label='R2', color='green')
pyplot.legend(loc='upper left')
pyplot.show()
dfAandR=df2[(df2.age_A<df2.age_B) & (df2.reach_A>df2.reach_B)]
dfAorR=df2[(df2.age_A<df2.age_B) | (df2.reach_A>df2.reach_B)]
dfAnotR=df2[(df2.age_A<df2.age_B) & (df2.reach_A<df2.reach_B)]

import seaborn as sns
sns.catplot(x='result', kind="count", palette="ch:.25", data=dfAandR)
sns.catplot(x='result', kind="count", palette="ch:.25", data=dfAorR)
sns.catplot(x='result', kind="count", palette="ch:.25", data=dfAnotR)
dfAandRandH=df2[(df2.age_A>df2.age_B) & (df2.reach_A<df2.reach_B)&(df2.height_A<df2.height_B)]
dfAorRorH=df2[(df2.age_A>df2.age_B) | (df2.reach_A<df2.reach_B)|(df2.height_A<df2.height_B)]

import seaborn as sns
sns.catplot(x='result', kind="count", palette="ch:.25", data=dfAandR)
sns.catplot(x='result', kind="count", palette="ch:.25", data=dfAorR)
sns.catplot(x='result', kind="count", palette="ch:.25", data=df2)