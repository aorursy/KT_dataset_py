# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt





# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



traintest = pd.concat([train, test], axis=0)

# traintest = traintest.round(3)



columns = list(train.columns)

excl = ['target', 'ID_code']



data_cols = columns

for ccc in excl :

    data_cols.pop(data_cols.index(ccc))



input_cols = data_cols

traintest.head()
stats = traintest[data_cols].describe()
#here going through all the columns 



checkdata = []

traintest['count'] = 1.0

traintest['tag'] = traintest['ID_code'].str[:5] #getting the tags of train and test

for i, ccc in enumerate(columns):

    df = traintest.pivot_table(columns='tag', index = ccc, values='count', aggfunc=np.sum)

    df = df.fillna(0)

    targetsum = traintest.groupby(ccc)['target','count' ].sum() # sum of target and counts

    targetsum['percent_target'] = targetsum['target'] /  targetsum['count']

    df = df.join(targetsum, how='left'   )

    df['train_v_test'] = df['train'] /  df['test_']

    checkdata.append(  df )



df.head(20)#example of the data we are looking at
checkdata[68].head()

len(checkdata[68])
len(checkdata[12])
traintest = traintest.sort_values(['var_68', 'var_12'])

plt.figure(figsize = (24,10))



plt.plot(traintest['var_12'][:].values)
traintest = traintest.sort_values(['var_12', 'var_68'])



plt.figure(figsize = (24,10))



plt.plot(traintest['var_68'][:100000].values)



traintest = traintest.sort_values(['var_68', 'var_126'])



plt.figure(figsize = (24,10))



plt.plot(traintest['var_126'][:10000].values)

traintest = traintest.sort_values(['var_0', 'var_1'])

plt.plot(traintest['var_1'][:100].values)

traintest[input_cols[:100]].boxplot(figsize=(24,10))
traintest[input_cols[100:]].boxplot(figsize=(24,10))
descr = train[data_cols].describe().transpose()

descr  =descr[descr['std'] > 7.5]

x = descr['mean']

y = descr['std']

n = list( descr.index)

fig, ax = plt.subplots(figsize=(30,30))



ax.scatter(x, y)



for i, txt in enumerate(n):

    ax.annotate(txt, (x[i], y[i]))

descr = train[data_cols].describe().transpose()

descr  =descr[(descr['std'] < 7.5) & (descr['std'] > 5)]

x = descr['mean']

y = descr['std']

n = list( descr.index)

fig, ax = plt.subplots(figsize=(30,30))



ax.scatter(x, y)



for i, txt in enumerate(n):

    ax.annotate(txt, (x[i], y[i]))
descr = train[data_cols].describe().transpose()

descr  =descr[descr['std'] < 5]

x = descr['mean']

y = descr['std']

n = list( descr.index)

fig, ax = plt.subplots(figsize=(30,30))



ax.scatter(x, y)



for i, txt in enumerate(n):

    ax.annotate(txt, (x[i], y[i]))
descr = train[data_cols].describe().transpose()

descr  =descr[descr['std'] < 3]

x = descr['mean']

y = descr['std']

n = list( descr.index)

fig, ax = plt.subplots(figsize=(30,30))



ax.scatter(x, y)



for i, txt in enumerate(n):

    ax.annotate(txt, (x[i], y[i]))
descr = train[data_cols].describe().transpose()



#sort1 

descr['meanround2'] = descr['mean'].round(1)

descr['stdround2'] = descr['std'].round(1)

descr = descr.sort_values(['stdround2', 'meanround2',])

sorted_cols = list( descr.index )    

    

for i in range(5):

    plt.figure(figsize = (30,10))

    train[sorted_cols[i * 40:(i+1) *40]].boxplot(figsize = (30,10))





#sort2

descr['mean_std'] = descr['mean'] + descr['std']

descr = descr.sort_values(['stdround2', 'mean_std'])

sorted_cols = list( descr.index )    

for i in range(5):

    plt.figure(figsize = (25,10))

    train[sorted_cols[i * 40:(i+1) *40]].boxplot(figsize = (25,10))

#sort3 

descr['range'] = descr['max'] - descr['min']

descr['range'] = descr['range'].round(0)

descr = descr.sort_values(['range', 'mean'])

sorted_cols = list( descr.index )    

for i in range(5):

    plt.figure(figsize = (25,10))

    train[sorted_cols[i * 40:(i+1) *40]].boxplot(figsize = (25,10))
#sort4 

descr['meanround2'] = descr['mean'].round(1)

descr['stdround2'] = descr['std'].round(1)

descr = descr.sort_values(['stdround2', 'meanround2'])

# descr = descr.sort_values(['meanround2', 'stdround2'])

sorted_cols = list( descr.index )     

for i in range(5):

    plt.figure(figsize = (25,10))

    train[sorted_cols[i * 40:(i+1) *40]].boxplot(figsize = (25,10))
# what if data was only shifted by a value but retains deviations?

temptrain = train[data_cols] - train[data_cols].mean()

descr = temptrain[data_cols].describe().transpose()



#sort5

# descr['stdround2'] = descr['std'].round(1)

descr = descr.sort_values(['std'])

sorted_cols = list( descr.index )     

for i in range(4):

    plt.figure(figsize = (20,10))

    temptrain[sorted_cols[i * 25:(i+1) *25]].boxplot(figsize = (20,10))


    

for i, dat in enumerate(checkdata):

    

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24,4))

    ax[0].plot(dat['count'])

    ax[0].title.set_text('Var_'+ str(i)+' count' )

    ax[1].plot(dat['target'])

    ax[1].title.set_text('target')

    ax[2].plot(dat['train_v_test'])

    ax[2].title.set_text('train_v_test')

    ax[3].plot(dat['percent_target'])

    ax[3].title.set_text('percent_target')



    plt.show()

    