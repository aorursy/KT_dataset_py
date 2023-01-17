import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/heart.csv")

df.shape
fig = plt.figure(figsize=(8,4))



#  subplot #1

plt.subplot(121)

plt.title('subplot(121)', fontsize=14)

sns.countplot(data=df, x='cp')



#  subplot #2

plt.subplot(122)

plt.title('subplot(122)', fontsize=14)

sns.scatterplot(data=df,x='age',y='chol',hue='sex')



plt.show()
row = 2

col = 1

cnt = 1  # initialize



fig = plt.figure(figsize=(4,10))



#  subplot #1

plt.subplot(row,col,cnt)

plt.title('subplot #{}:  row = {}, column = {}, plot count = {}'.format(cnt,row,col,cnt), fontsize=14)

sns.countplot(data=df, x='cp')



cnt = cnt + 1  # increment counter



#  subplot #2

plt.subplot(row,col,cnt)

plt.title('subplot #{}:  row = {}, column = {}, plot count = {}'.format(cnt,row,col,cnt), fontsize=14)

sns.scatterplot(data=df,x='age',y='chol',hue='sex')



plt.show()
fig = plt.figure(figsize=(14,12))



#  subplot #1

plt.subplot(231)

plt.title('subplot(231)', fontsize=14)

sns.countplot(data=df, x='cp',hue='sex')



#  subplot #2

plt.subplot(2,3,2)

plt.title('subplot(2,3,2)', fontsize=14)

sns.scatterplot(data=df,x='age',y='chol',hue='sex')



#  subplot #3

plt.subplot(233)

plt.title('subplot(233)', fontsize=14)

sns.lineplot(data=df, x=df['age'],y=df['oldpeak'])



#  subplot #4

plt.subplot(2,3,4)

plt.title('subplot(2,3,4)', fontsize=14)

sns.boxplot(data=df[['chol','trestbps','thalach']])



#  subplot #5

plt.subplot(235)

plt.title('subplot(235)', fontsize=14)

#sns.countplot(data=df, x='slope',hue='sex')

sns.distplot(df.age, color='darkgreen')



plt.show()
#  Plots: Overall, no disease and disease

df2 = df[['sex','cp','slope','ca']] # select a few attributes



#  select "no disease" and "disease" data

df_target_0 = df[(df['target'] == 0)]

df_target_1 = df[(df['target'] == 1)]





#  SUBPLOTS - FOR Loop

rowCnt = len(df2.columns)

colCnt = 3     # cols:  OVERALL, NO DISEASE, DISEASE

subCnt = 1     # initialize plot number



fig = plt.figure(figsize=(12,24))



for i in df2.columns:

    # OVERALL plots

    plt.subplot(rowCnt, colCnt, subCnt)

    plt.title('OVERALL (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.countplot(df[i], hue=df.sex)

    subCnt = subCnt + 1



    # NO DISEASE PLOTS

    plt.subplot(rowCnt, colCnt, subCnt)

    plt.title('NO DISEASE (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.countplot(df_target_0[i], hue=df.sex)

    subCnt = subCnt + 1



    # PLOTS

    plt.subplot(rowCnt, colCnt, subCnt)

    plt.title('DISEASE (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.countplot(df_target_1[i], hue=df.sex)

    subCnt = subCnt + 1



plt.show()
#  SUBPLOTS

fig = plt.figure(figsize=(12,6))



# correlation - female

#---------------------

dfFemale     = df2[(df2['sex'] == 1)]

dfFemaleCorr = dfFemale.drop(["sex"], axis=1).corr()



plt.subplot(121)   #  subplot 1 - female

plt.title('correlation Heart Disease - FEMALE', fontsize=14)

sns.heatmap(dfFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')





# correlation - male

#-------------------

dfMale     = df2[(df2['sex'] == 0)]

dfMaleCorr = dfMale.drop(["sex"], axis=1).corr()



plt.subplot(122)   #  subplot 2 - male

plt.title('correlation Heart Disease - MALE', fontsize=14)

sns.heatmap(dfMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')



plt.show()
sns.pairplot(data = df[['age','chol','trestbps']])