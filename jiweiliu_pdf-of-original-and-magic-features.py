import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

import seaborn as sns

import time

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
def logloss(y,yp):

    yp = np.clip(yp,1e-5,1-1e-5)

    return -y*np.log(yp)-(1-y)*np.log(1-yp)

    

def reverse(tr,te):

    reverse_list = [0,1,2,3,4,5,6,7,8,11,15,16,18,19,

                22,24,25,26,27,41,29,

                32,35,37,40,48,49,47,

                55,51,52,53,60,61,62,103,65,66,67,69,

                70,71,74,78,79,

                82,84,89,90,91,94,95,96,97,99,

                105,106,110,111,112,118,119,125,128,

                130,133,134,135,137,

                140,144,145,147,151,155,157,159,

                161,162,163,164,167,168,

                170,171,173,175,176,179,

                180,181,184,185,187,189,

                190,191,195,196,199]

    reverse_list = ['var_%d'%i for i in reverse_list]

    for col in tr.columns:

        colx = '_'.join(col.split('_')[:2])

        if colx in reverse_list and 'count' not in col: 

            tr[col] = tr[col]*(-1)

            te[col] = te[col]*(-1)

    return tr,te



def scale(tr,te):

    for col in tr.columns:

        if col.startswith('var_') and 'count' not in col:

            mean,std = tr[col].mean(),tr[col].std()

            tr[col] = (tr[col]-mean)/std

            if col in te.columns:

                te[col] = (te[col]-mean)/std

    return tr,te



def getp_vec_sum(x,x_sort,y,std,c=0.5):

    # x is sorted

    left = x - std/c

    right = x + std/c

    p_left = np.searchsorted(x_sort,left)

    p_right = np.searchsorted(x_sort,right)

    p_right[p_right>=y.shape[0]] = y.shape[0]-1

    p_left[p_left>=y.shape[0]] = y.shape[0]-1

    return (y[p_right]-y[p_left])



def get_prob(tr,col,x_query=None,smooth=3,silent=1):

    std = tr[col].std()

    N = tr.shape[0]

    tr = tr.dropna(subset=[col])

    if silent==0:

        print("null ratio %.4f"%(tr.shape[0]/N))

    df = tr.groupby(col).agg({'target':['sum','count']})

    cols = ['sum_y','count_y']

    df.columns = cols

    df = df.reset_index()

    df = df.sort_values(col)

    y,c = cols

    

    df[y] = df[y].cumsum()

    df[c] = df[c].cumsum()

    

    if x_query is None:

        rmin,rmax,res = -5.0, 5.0, 501

        x_query = np.linspace(rmin,rmax,res)

    

    dg = pd.DataFrame()

    tm = getp_vec_sum(x_query,df[col].values,df[y].values,std,c=smooth)

    cm = getp_vec_sum(x_query,df[col].values,df[c].values,std,c=smooth)+1

    dg['res'] = tm/cm

    dg.loc[cm<500,'res'] = 0.1

    return dg['res'].values



def get_probs(tr):

    y = []

    for i in range(200):

        name = 'var_%d'%i

        res = get_prob(tr,name)

        y.append(res)

    return np.vstack(y)



def split(tr):

    split = pickle.load(open('cache/kfolds.pkl','rb'))

    for trx,tex in split:

        break

    trx,vax = tr.iloc[trx],tr.iloc[tex]

    return trx,vax





def plot_pdf(tr,name):

    name1 = '%s_no_noise'%name

    name2 = '%s_no_noise2'%name

    rmin,rmax,res = -5.0, 5.0, 501

    x_query = np.linspace(rmin,rmax,res)

    plt.figure(figsize=(10,5))

    prob1 = get_prob(tr,name,x_query=x_query)

    prob2 = get_prob(tr,name1,x_query=x_query)

    prob3 = get_prob(tr,name2,x_query=x_query)

    plt.grid()

    plt.plot(x_query,prob1,color='b',label=name)

    plt.plot(x_query,prob2,color='r',label=name1)

    plt.plot(x_query,prob3,color='g',label=name2)

    plt.legend(loc='upper right')

    plt.title('PDF of '+name)

    

def plot_pdfs(tr):

    rmin,rmax,res = -5.0, 5.0, 501

    x_query = np.linspace(rmin,rmax,res)

    cols = [i for i in tr.columns if i.startswith('var')]

    for i in range(50):

        plt.figure(figsize=(18,10))

        print('plot var %d to var %d'%(i*4,i*4+4))

        for j in range(4):

            cx = i*4+j

            name = 'var_%d'%cx

            name1 = '%s_no_noise'%name

            name2 = '%s_no_noise2'%name

            plt.subplot(2,2,j+1)

            prob1 = get_prob(tr,name,x_query=x_query)

            prob2 = get_prob(tr,name1,x_query=x_query)

            prob3 = get_prob(tr,name2,x_query=x_query)

            plt.grid()

            plt.plot(x_query,prob1,color='b',label=name)

            plt.plot(x_query,prob2,color='r',label=name1)

            plt.plot(x_query,prob3,color='g',label=name2)

            plt.legend(loc='upper right')

            plt.title('PDF of '+name)

        plt.show()
def build_magic():

    tr_path,te_path = '../input/train.csv','../input/test.csv'



    tr = pd.read_csv(tr_path)#.drop([IDCOL,YCOL],axis=1)

    te = pd.read_csv(te_path)#.drop([IDCOL],axis=1)

    cols = [i for i in tr.columns if i.startswith('var_')]

    N = tr.shape[0]

    tr['real'] = 1

    te0 = te.copy()

    for col in cols:

        te[col] = te[col].map(te[col].value_counts())

    a = te[cols].min(axis=1)

    te0['real'] = (a == 1).astype('int')



    tr = tr.append(te0).reset_index(drop=True)

    for col in cols:

        tr[col+'_count'] = tr[col].map(tr.loc[tr.real==1,col].value_counts())

    for col in cols:

        tr.loc[tr[col+'_count']>1,col+'_no_noise'] = tr.loc[tr[col+'_count']>1,col]

        tr.loc[tr[col+'_count']>2,col+'_no_noise2'] = tr.loc[tr[col+'_count']>2,col]

    return tr
%%time

tr = build_magic()

te = tr[tr.target.isnull()]

tr = tr[tr.target.isnull()==0]

print(tr.shape)
%%time

tr,te = reverse(tr,te)

tr,te = scale(tr,te)
%%time

plot_pdfs(tr)