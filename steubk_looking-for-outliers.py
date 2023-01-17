import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder





%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.options.display.max_columns = 999



# read datasets

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_test['y'] = np.nan





            

df_all = pd.concat ( [df_train, df_test])



le = LabelEncoder()



for c in ['X0','X1','X2','X3','X4','X5','X6', 'X8']:

          

    df_all['cat_' + c] = le.fit_transform(df_all[c].values)





avg_X0=df_all[['y','X0']].groupby(['X0']).mean().reset_index()

avg_X0.columns=['X0','avg_X0']

mean_X0 = df_all['y'].mean() 



df_all=df_all.merge(avg_X0, on='X0', how='left')

df_all['avg_X0'].fillna(mean_X0,inplace=True)

df_all['y-avg_X0'] = df_all['y'] - df_all['avg_X0'] 



df_train = df_all[ df_all['y'].notnull() ]



plt.figure(figsize=(40,20))



i=0

df = df_train.sort_values(by=['y-avg_X0']).reset_index()

i += 1

ax =  plt.subplot(1,2,i)

plt.scatter(df.index,  df['y-avg_X0'], c=df['cat_X5'],  linewidth=5, cmap=matplotlib.cm.get_cmap('rainbow') )

plt.xlabel( '{}'.format('y-avg_X0'), fontsize=12)

ax.set_ylim(-25,180)



i += 1

ax =  plt.subplot(1,2,i)



plt.hist(  df['y-avg_X0'] ,  weights=np.zeros_like(df['y-avg_X0']) + 100. / len(df['y-avg_X0'] ), bins=20)

plt.xlabel( '{}'.format('y-avg_X0'), fontsize=12)

ax.set_ylim(0,80)

        

        

plt.show()
X0=list(set(df_train['X0'].values))

X0.sort()

plt.figure(figsize=(20,40))

i=0



for x in X0:



    df = df_train[ df_train['X0'] == x ]

    

     

    if (df.shape[0]> 100):



        mean=df['avg_X0'].mean()



        df = df.sort_values(by=['y-avg_X0']).reset_index()

        i += 1

        ax =  plt.subplot(10,6,i)



        plt.scatter(df.index,  df['y-avg_X0'], c=df['cat_X5'], s=8, cmap=matplotlib.cm.get_cmap('rainbow') )

        plt.xlabel( '{}/{:.2f}'.format(x,mean), fontsize=12)

        ax.set_ylim(-25,180)

        

        i += 1

        ax =  plt.subplot(10,6,i)



        plt.hist(  df['y-avg_X0'] ,  weights=np.zeros_like(df['y-avg_X0']) + 100. / len(df['y-avg_X0'] ), bins=20)

        plt.xlabel( '{}/{:.2f}'.format(x,mean), fontsize=12)

        ax.set_ylim(0,70)

        

        

plt.show()
def plot_next ( cat, rows,min_y, max_y,  i):

    i += 1

    ax =  plt.subplot(rows,3,i)

    plt.scatter(df.index,  df['y-avg_X0'], c=df[cat], s=20, linewidth='0', cmap=matplotlib.cm.get_cmap('rainbow') )

    plt.xlabel( '{}/{:.2f}'.format(x,mean), fontsize=12)

    ax.set_ylim(min_y,max_y)

    i += 1

    ax =  plt.subplot(rows,3,i)

    plt.hist(  df[cat] ,  weights=np.zeros_like(df[cat]) + 100. / len(df[cat] ), bins=20)

    plt.xlabel( '{}/{:.2f}'.format(x,mean), fontsize=12)     



    i += 1

    ax =  plt.subplot(rows,3,i)



    plt.hist(  df['y-avg_X0'] ,  weights=np.zeros_like(df['y-avg_X0']) + 100. / len(df['y-avg_X0'] ), bins=20)

    plt.xlabel( '{}/{:.2f}'.format(x,mean), fontsize=12)

    ax.set_xlim(min_y,max_y)   

    return i



X0=['aj','o', 'az', 'n', 'ap']

plt.figure(figsize=(20,30))

i=0

for x in X0:



    df = df_train[ df_train['X0'] == x ]

    if (df.shape[0]> 100):

        mean=df['avg_X0'].mean()



        df = df.sort_values(by=['y-avg_X0']).reset_index()

        i = plot_next ('cat_X1',len(X0),-25, 75,   i)       

        

plt.show()

X0=['f', 'w', 'ak' ]

plt.figure(figsize=(20,20))

i=0

for x in X0:

    df = df_train[ df_train['X0'] == x ]

    mean=df['avg_X0'].mean()



    df = df.sort_values(by=['y-avg_X0']).reset_index()

    i = plot_next ('cat_X1', len(X0),-25, 60,   i)       

        

plt.show()

X0=['ay','j', 'x' , 'y', 'z', 's', 't' ]

plt.figure(figsize=(20,40))

i=0

for x in X0:



    df = df_train[ df_train['X0'] == x ]

    if (df.shape[0]> 100):

        mean=df['avg_X0'].mean()



        df = df.sort_values(by=['y-avg_X0']).reset_index()

        i = plot_next ('cat_X1',len(X0),-25, 185 , i)       

        

plt.show()