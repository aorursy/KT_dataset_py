import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import xgboost as xgb 

from sklearn.metrics import r2_score



%matplotlib inline



from IPython.display import display, HTML

# Shows all columns of a dataframe

def show_dataframe(X, rows = 5):

    display(HTML(X.to_html(max_rows=rows)))



# Datasets

train = pd.read_csv('../input/train.csv')

def vcm(X_):

    M11=np.ones((len(X_),len(X_)))

    X_a=X_-M11.dot(X_)/len(X_)

    return X_a.T.dot(X_a)/len(X_)



def vcminv(X_):

    kolom=X_.columns

    ym_=X_.mean(axis=0)

    X_vcm=pd.DataFrame( X_.cov()*2,index=kolom,columns=kolom) 

    X_vcm['y_']=ym_

    X_vcm['lamb']=1.0    

    X_vcmT=X_vcm.T

    X_vcmT['y_']=ym_.append(pd.DataFrame([0,0],index=['lamb','y_']))

    X_vcmT['lamb']=1.0    

    X_vcmT['lamb']['lamb']=0.0

    X_vcmT['y_']['lamb']=0.0

    X_vcmT['lamb']['y_']=0.0    

    X_vcmT.dropna(axis=(0,1), how='any')

    show_dataframe(X_vcmT)

    koloml=X_vcmT.columns

    return pd.DataFrame( np.linalg.pinv( X_vcmT) ,index=koloml,columns=koloml )
divisor=201

#create groups of divisor length

train['groep']=train.index/divisor-.499

train['groep']=train['groep'].round(0)

train['teller']=1.0



#groep mean - standarddeviation

me_st=train[['y','groep']].groupby(by='groep').mean()

me_st['v']=train[['y','groep']].groupby(by='groep').var()

#print(me_st)



#X0 mean - standarddeviation

Xvar='X0'

X0_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()

X0_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()





#print(X0_me_st)

Xvar='X1'

X8_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()

X8_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()

#print(X0_me_st)



import matplotlib.pyplot as plt

import seaborn as sns





plt.title('TESTTime Variability Group o X0 * X1 +')

plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)

plt.scatter(y=X0_me_st['y'], x=X0_me_st['v'], marker='*', alpha=0.5)

plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)

plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')



divisor=201

#create groups of divisor length

train['groep']=train.index/divisor-.499

train['groep']=train['groep'].round(0)

train['teller']=1.0



#groep mean - standarddeviation

me_st=train[['y','groep']].groupby(by='groep').mean()

me_st['v']=train[['y','groep']].groupby(by='groep').var()

#print(me_st)



#X0 mean - standarddeviation

Xvar='X2'

X0_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()

X0_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()



Xvar='X3'

X8_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()

X8_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()

#print(X0_me_st)



import matplotlib.pyplot as plt

import seaborn as sns





plt.title('TESTTime Variability Group o X2 * X3 + ')

#plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)

plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)

plt.scatter(y=X0_me_st['y'], x=X0_me_st['v'], marker='*', alpha=0.5)

#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)

plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)

plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')
divisor=201

#create groups of divisor length

train['groep']=train.index/divisor-.499

train['groep']=train['groep'].round(0)

train['teller']=1.0



#groep mean - standarddeviation

me_st=train[['y','groep']].groupby(by='groep').mean()

me_st['v']=train[['y','groep']].groupby(by='groep').var()

#print(me_st)



#X0 mean - standarddeviation

Xvar='X6'

X0_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()

X0_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()



Xvar='X8'

X8_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()

X8_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()

#print(X0_me_st)



import matplotlib.pyplot as plt

import seaborn as sns





plt.title('TESTTime Variability Group o X6 * X8 + ')

#plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)

plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)

plt.scatter(y=X0_me_st['y'], x=X0_me_st['v'], marker='*', alpha=0.5)

#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)

plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)

plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X0'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)



Am1=vcminv(X0_time)

b_=[0 for ci in X0_time.columns]+[0,0]

b_[-1]=1

X0m_=X0_time.mean(axis=0) 

X0m2_=X0m_.append(pd.DataFrame([1,1],index=['y_','lambda']))

#print(X0m_)

portRet=[]

portSTD=[]

vcm_=vcm(X0_time)

for ef in range(70,140,1):

    b_[-2]=ef

    portf=Am1.dot(b_)

    #print(portf)

    #porty=( portf.T*(X0m2_.T) ).sum(axis=1)  #exact target return

    portRet=portRet+[ef]

    

    var=(portf[:-2].T.dot(vcm_)).dot(portf[:-2]) #

    portSTD=portSTD+ [var ]



X0_ym=train[['y','X0']].groupby(by='X0').mean()

X0_yv=train[['y','X0']].groupby(by='X0').var()

#print(portSTD)

plt.title('Efficient frontier X0 versus group  ')

plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)

plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)

plt.scatter(y=X0_ym,x=X0_yv,marker='*', alpha=0.5)

#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)

#plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)

plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X3'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)



Am1=vcminv(X0_time)

b_=[0 for ci in X0_time.columns]+[0,0]

b_[-1]=1

X0m_=X0_time.mean(axis=0) 

X0m2_=X0m_.append(pd.DataFrame([1,1],index=['y_','lambda']))

#print(X0m_)

portRet=[]

portSTD=[]

vcm_=X0_time.cov()

for ef in range(70,140,1):

    b_[-2]=ef

    portf=Am1.dot(b_)

    #print(portf)

    #porty=( portf.T*(X0m2_.T) ).sum(axis=1)  #exact target return

    portRet=portRet+[ef]

    var=(portf[:-2].T.dot(vcm_)).dot(portf[:-2]) #

    portSTD=portSTD+ [var ]





X3_ym=train[['y','X3']].groupby(by='X3').mean()

X3_yv=train[['y','X3']].groupby(by='X3').var()

#print(portSTD)

plt.title('Efficient frontier X3 versus group  ')

plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)

plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)

plt.scatter(y=X3_ym,x=X3_yv,marker='*', alpha=0.5)

#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)

#plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)

plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X6'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)



Am1=vcminv(X0_time)

b_=[0 for ci in X0_time.columns]+[0,0]

b_[-1]=1

X0m_=X0_time.mean(axis=0) 

X0m2_=X0m_.append(pd.DataFrame([1,1],index=['y_','lambda']))

#print(X0m_)

portRet=[]

portSTD=[]

vcm_=X0_time.cov()

for ef in range(70,140,1):

    b_[-2]=ef

    portf=Am1.dot(b_)

    #print(portf)

    #porty=( portf.T*(X0m2_.T) ).sum(axis=1)  #exact target return

    portRet=portRet+[ef]

    var=(portf[:-2].T.dot(vcm_)).dot(portf[:-2]) #

    portSTD=portSTD+ [var ]



X6_ym=train[['y','X6']].groupby(by='X6').mean()

X6_yv=train[['y','X6']].groupby(by='X6').var()

#print(portSTD)

plt.title('Efficient frontier X6 versus group  ')

plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)

plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)

plt.scatter(y=X6_ym,x=X6_yv,marker='*', alpha=0.5)

plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')



b_[-2]=100

print('Percent')

print( Am1.dot(b_).round(2)*100 )

b_[-2]=103

print('Percent')

print( Am1.dot(b_).round(2)*100 )
def vcminv2(X_):

    kolom=X_.columns

    ym_=X_.mean(axis=0)

    X_vcm=pd.DataFrame( X_.cov()*2,index=kolom,columns=kolom) 

    X_vcm['y_']=ym_

    X_vcm['lamb']=1.0    

    Xm_vcm=pd.DataFrame( -X_.cov()*2,index=kolom,columns=kolom) 

    Xm_vcm['y_']=0

    Xm_vcm['lamb']=-1.0    

    Xt_vcm=X_vcm.append(Xm_vcm)

    #print(Xt_vcm)

    

    Xt_vcmT=Xt_vcm.T

    Xt_vcmT['y_']=ym_.append(pd.DataFrame([0,0],index=['lamb','y_']))

    Xt_vcmT['lamb']=1.0    

    Xt_vcmT['lamb']['lamb']=0.0

    Xt_vcmT['y_']['lamb']=0.0

    Xt_vcmT['lamb']['y_']=0.0    

    Xt_vcmT.dropna(axis=(0,1), how='any')

    X2_vcm=Xt_vcmT.T

    show_dataframe(X2_vcm)

    koloml=X2_vcm.columns

    rowi=X2_vcm.index

    return pd.DataFrame( np.linalg.pinv( X2_vcm) ,columns=rowi,index=koloml )
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X3'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)



Am1=vcminv2(X0_time)

#print(Am1)

b_=[0 for ci in X0_time.columns]+[0,0]

b_[-1]=1

b_[-2]=100

optimum=pd.DataFrame( Am1.T.dot(b_) ,index=Am1.columns)



print('100 seconds optimal distribution',Am1.T.dot(b_) )

#print(optimum)
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='teller', index=['groep'],columns=['X3'], aggfunc=np.sum).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

vergelijk = X0_time/2

optimum.columns=['optimum_100']

optimum=optimum[:7]*100

vergelijk = vergelijk.append(optimum.T)

print(vergelijk)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(vergelijk, cmap=cmap)

X3_ym=train[['y','X3']].groupby(by='X3').mean()

X3_ys=train[['y','X3']].groupby(by='X3').var()

x3vc=X3_ys/(X3_ym)

import matplotlib.pyplot as plt

ax = x3vc.plot(kind='bar', title ="Variance coeff", figsize=(15, 10), legend=True, fontsize=12)

plt.show()
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X6'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)



Am1=vcminv2(X0_time)

#print(Am1)

b_=[0 for ci in X0_time.columns]+[0,0]

b_[-1]=1

b_[-2]=100

optimum=pd.DataFrame( Am1.T.dot(b_) ,index=Am1.columns)



print('100 seconds optimal distribution',Am1.T.dot(b_) )

#print(optimum)
# groep  VCM with pivot and fillNA

X0_time = pd.pivot_table(train, values='teller', index=['groep'],columns=['X6'], aggfunc=np.sum).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

vergelijk = X0_time/2

optimum.columns=['optimum_100']

optimum=optimum[:12]*100

vergelijk = vergelijk.append(optimum.T)

print(vergelijk)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(vergelijk, cmap=cmap)
X6_ym=train[['y','X6']].groupby(by='X6').mean()

X6_ys=train[['y','X6']].groupby(by='X6').var()

x6vc=X6_ys/(X6_ym)

import matplotlib.pyplot as plt

ax = x6vc.plot(kind='bar', title ="Variance coeff", figsize=(15, 10), legend=True, fontsize=12)

plt.show()
X5_ym=train[['y','X5']].groupby(by='X5').mean()

X5_ys=train[['y','X5']].groupby(by='X5').var()

x5vc=X5_ys/(X5_ym)

import matplotlib.pyplot as plt

ax = x5vc.plot(kind='bar', title ="Variance coeff", figsize=(15, 10), legend=True, fontsize=12)

plt.show()



X2_ym=train[['y','X2']].groupby(by='X2').mean()

X2_ys=train[['y','X2']].groupby(by='X2').var()

x2vc=X2_ys/(X2_ym)

import matplotlib.pyplot as plt

ax = x2vc.plot(kind='bar', title ="Variance coeff", figsize=(15, 10), legend=True, fontsize=12)

plt.show()



X1_ym=train[['y','X1']].groupby(by='X1').mean()

X1_ys=train[['y','X1']].groupby(by='X1').var()

x1vc=X1_ys/(X1_ym)

import matplotlib.pyplot as plt

ax = x1vc.plot(kind='bar', title ="Variance coeff", figsize=(15, 10), legend=True, fontsize=12)

plt.show()



X0_ym=train[['y','X0']].groupby(by='X0').mean()

X0_ys=train[['y','X0']].groupby(by='X0').var()

x0vc=X0_ys/(X0_ym)

import matplotlib.pyplot as plt

ax = x0vc.plot(kind='bar', title ="Variance coeff", figsize=(15, 10), legend=True, fontsize=12)

plt.show()
def add_new_col(x):

    if x not in new_col.keys(): 

        # set n/2 x if is contained in test, but not in train 

        # (n is the number of unique labels in train)

        # or an alternative could be -100 (something out of range [0; n-1]

        return int(len(new_col.keys())/2)

    return new_col[x] # rank of the label



new_col= train[['y','X0']].groupby('X0').describe().fillna(method='bfill')

new_col.columns=['count','mean','std','min','p25','p50','p75','max']

new_col['eff']=new_col['std']/new_col['mean']

new_col['eff2']=new_col['eff']*new_col['std']



def clust(x):

    kl=0

    if x<0.75:    #

        kl=1

    if x>0.75 and x<1.33:   #slight problem

        kl=2

    if x>1.33:   # big problem

        kl=4

    return kl

new_col['clust']=new_col['eff2'].map(clust)

print(new_col)
train_new=pd.merge(train,new_col, how='inner', left_on='X0', right_index=True)

train_new=train_new.sort_values(by=['clust','X0'])

import seaborn as sns

sns.set(style="ticks")

sns.pairplot(train_new[['y','std','p50','max','eff2','X0']],hue='X0')

plt.show()

from sklearn.random_projection import SparseRandomProjection

from sklearn.cluster import KMeans



import seaborn as sns

from pandas.plotting import scatter_matrix

# INPUT df  (dataframe en welke kolommen je gebruikt om te klusteren)

# define 'clust' groep

# define drop colomns



#-------------------------------------

labels = train_new['clust']

y_values= train_new['y']  #transfer data before drop !

drop_columns=['y','mean','ID','min','p25','p50','p75','max']

#y values df_new['y']

# X = all the variables X10-X300 not dupl, not singular

X = train_new.drop(['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8','X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347','X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347', 'X382', 'X232', 'X279', 'X35', 'X37', 'X39', 'X302', 'X113', 'X134', 'X147', 'X222', 'X102', 'X214', 'X239', 'X76', 'X324', 'X248', 'X253', 'X385', 'X172', 'X216', 'X213', 'X84', 'X244', 'X122', 'X243', 'X320', 'X245', 'X94', 'X242', 'X199', 'X119', 'X227', 'X146', 'X226', 'X326', 'X360', 'X262', 'X266', 'X247', 'X254', 'X364', 'X365', 'X296', 'X299','y','ID','mean','min','p25','p50','p75','max'],axis=1)

n_comp = 5  #define number of clusters

#-------------------------------------



print('-------Sparse Random Projection---------')

# SRP

srp = SparseRandomProjection(n_components=5, dense_output=True, random_state=420)

results = srp.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()



from mpl_toolkits.mplot3d import Axes3D



# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(12, 12))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(results[0], results[1], results[2], c=labels, cmap=plt.cm.Paired)

ax.set_title("First three Sparse Random Projections")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()