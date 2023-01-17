# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

df = pd.read_csv('../input/rainfall in india 1901-2015.csv')

df.info()

df.dropna(how='any', inplace=True)

# Any results you write to the current directory are saved as output.
subdivs = df['SUBDIVISION'].unique()

num_of_subdivs = subdivs.size

print('Total # of Subdivs: ' + str(num_of_subdivs))

subdivs
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar', color='r',width=0.3,title='Subdivision wise Average Annual Rainfall', fontsize=20)

plt.xticks(rotation = 90)

plt.ylabel('Average Annual Rainfall (mm)')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)

print(df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])

print(df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

dfg = df.groupby('YEAR').sum()['ANNUAL']

dfg.plot('line', title='Overall Rainfall in Each Year', fontsize=20)

#df.groupby('YEAR').sum()['ANNUAL'].plot()

#plt.xlim(0, 115)

#plt.xticks(np.linspace(0,115,24,endpoint=True),np.linspace(1900,2015,24,endpoint=True).astype(int))

#plt.xticks(np.linspace(1901,2015,24,endpoint=True))

#plt.xticks(rotation = 90)

plt.ylabel('Overall Rainfall (mm)')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)

print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))

print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))

print('Mean: ' + str(dfg.mean()))



months = df.columns[2:14]

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

xlbls = df['SUBDIVISION'].unique()

xlbls.sort()

dfg = df.groupby('SUBDIVISION').mean()[months]

dfg.plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)

plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)

plt.xticks(  rotation = 90)

plt.ylabel('Rainfall (mm)')

plt.legend(loc='upper right', fontsize = 'xx-large')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)



dfg = dfg.mean(axis=0)

print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))

print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))

print('Mean: ' + str(dfg.mean()))





months = df.columns[2:14]

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df.groupby('YEAR').mean()[months].plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)

#plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)

plt.xticks(  rotation = 90)

plt.ylabel('Rainfall (mm)')

plt.legend(loc='upper right', fontsize = 'x-large')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)
df2 = df[['SUBDIVISION',months[0],months[1],months[2],months[3]]]

#df2 = df['YEAR','JAN','FEB','MAR','APR']

df2.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])



for k in range(1,9):

    df3 = df[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]

    df3.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])

    df2 = df2.append(df3)

df2.index = range(df2.shape[0])

    

#df2 = pd.concat([df2, pd.get_dummies(df2['SUBDIVISION'])], axis=1)



df2.drop('SUBDIVISION', axis=1,inplace=True)

#print(df2.info())

msk = np.random.rand(len(df2)) < 0.8



df_train = df2[msk]

df_test = df2[~msk]

df_train.index = range(df_train.shape[0])

df_test.index = range(df_test.shape[0])



reg = linear_model.LinearRegression()

reg.fit(df_train.drop('x4',axis=1),df_train['x4'])

predicted_values = reg.predict(df_train.drop('x4',axis=1))

residuals = predicted_values-df_train['x4'].values

print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))

df_res = pd.DataFrame(residuals)

df_res.columns = ['Residuals']



fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)

ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)





predicted_values = reg.predict(df_test.drop('x4',axis=1))

residuals = predicted_values-df_test['x4'].values

print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))

df_res = pd.DataFrame(residuals)

df_res.columns = ['Residuals']



fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)

ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)
df2 = df[['SUBDIVISION',months[0],months[1],months[2],months[3]]]



df2.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])



for k in range(1,9):

    df3 = df[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]

    df3.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])

    df2 = df2.append(df3)

df2.index = range(df2.shape[0])

df2 = pd.concat([df2, pd.get_dummies(df2['SUBDIVISION'])], axis=1)





df2.drop('SUBDIVISION', axis=1,inplace=True)



msk = np.random.rand(len(df2)) < 0.8



df_train = df2[msk]

df_test = df2[~msk]

df_train.index = range(df_train.shape[0])

df_test.index = range(df_test.shape[0])





reg = linear_model.LinearRegression()

reg.fit(df_train.drop('x4',axis=1),df_train['x4'])

#print(reg.coef_)

predicted_values = reg.predict(df_train.drop('x4',axis=1))

residuals = predicted_values-df_train['x4'].values

df_res = pd.DataFrame(residuals)

df_res.columns = ['Residuals']

print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))



fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res.plot.line(title='Different b/w Actual and Predicted', color = 'c', ax=ax,fontsize=20)

ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)







predicted_values = reg.predict(df_test.drop('x4',axis=1))

residuals = predicted_values-df_test['x4'].values

df_res = pd.DataFrame(residuals)

df_res.columns = ['Residuals']

print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))



fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res.plot.line(title='Different b/w Actual and Predicted', color='m', ax=ax,fontsize=20)

ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)


#subdivs = df['SUBDIVISION'].unique()

#num_of_subdivs = subdivs.size



df_res_training = pd.DataFrame(columns=np.array(['Residuals']))

df_res_testing = pd.DataFrame(columns=np.array(['Residuals']))

list_mad_training = []

mean_abs_diff_training = 0

list_mad_testing = []

mean_abs_diff_testing = 0

for subd in subdivs:

    df1 = df[df['SUBDIVISION']==subd]

    df2 = df1[[months[0],months[1],months[2],months[3]]]

    df2.columns = np.array(['x1','x2','x3','x4'])

    for k in range(1,9):

        df3 = df1[[months[k],months[k+1],months[k+2],months[k+3]]]

        df3.columns = np.array(['x1','x2','x3','x4'])

        df2 = df2.append(df3)

    df2.index = range(df2.shape[0])

    msk = np.random.rand(len(df2)) < 0.8

    df_train = df2[msk]

    df_test = df2[~msk]

    df_train.index = range(df_train.shape[0])

    df_test.index = range(df_test.shape[0])

    reg = linear_model.LinearRegression()

    reg.fit(df_train.drop('x4',axis=1),df_train['x4'])

    

    predicted_values = reg.predict(df_train.drop('x4',axis=1))

    residuals = predicted_values-df_train['x4'].values

    df_res_training = df_res_training.append(pd.DataFrame(residuals,columns=np.array(['Residuals'])))

    mean_abs_diff_training = mean_abs_diff_training + np.sum(np.abs(residuals))

    list_mad_training.append(np.mean(np.abs(residuals)))

    

    predicted_values = reg.predict(df_test.drop('x4',axis=1))

    residuals = predicted_values-df_test['x4'].values

    df_res_testing = df_res_testing.append(pd.DataFrame(residuals,columns=np.array(['Residuals'])))

    mean_abs_diff_testing = mean_abs_diff_testing + np.sum(np.abs(residuals))

    list_mad_testing.append(np.mean(np.abs(residuals)))

    

    

df_res_training.index = range(df_res_training.shape[0])

mean_abs_diff_training = mean_abs_diff_training/df_res_training.shape[0]

print('Overall MAD (Training): ' + str(mean_abs_diff_training))

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res_training.plot.line(title='Residuals (Training)', color='c',ax=ax,fontsize=20)

#ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)



df_res_testing.index = range(df_res_testing.shape[0])

mean_abs_diff_testing = mean_abs_diff_testing/df_res_testing.shape[0]

print('Overall MAD (Testing): ' + str(mean_abs_diff_testing))

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res_testing.plot.line(title='Residuals (Testing)', color='m',ax=ax,fontsize=20)

#ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)





pd_mad = pd.DataFrame(data=list_mad_training,columns=["MAD (Train)"])

pd_mad["MAD (Test)"] = list_mad_testing;

pd_mad['Subdivisions'] = subdivs;

fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

#pd_mad.groupby('Subdivisions').mean().plot(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)

pd_mad.groupby('Subdivisions').mean().plot.bar( width=0.5,title='MAD for Subdivisions', ax= ax, fontsize=20)

plt.xticks(rotation = 90)

plt.ylabel('Mean Abs. Difference')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)
