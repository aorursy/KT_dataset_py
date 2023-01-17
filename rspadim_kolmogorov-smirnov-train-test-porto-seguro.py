import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

# We start by loading the training / test data and combining them with minimal preprocessing necessary

print('Loading')

xtrain = pd.read_csv('../input/train.csv')

xtest  = pd.read_csv('../input/test.csv')



#drop columns

print('Drop')

xtrain.drop(['id', 'target'], axis = 1, inplace = True)

xtest.drop(['id'], axis = 1, inplace = True)



# add identifier and combine

print('Merge')

xtrain['istrain'] = 1

xtest['istrain'] = 0

xdat = pd.concat([xtrain, xtest], axis = 0)



# convert non-numerical columns to integers

print('Transform')

df_numeric = xdat.select_dtypes(exclude=['object'])

df_obj = xdat.select_dtypes(include=['object']).copy()

    

for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]

    

xdat = pd.concat([df_numeric, df_obj], axis=1)

y = xdat['istrain']

xdat.drop('istrain', axis = 1, inplace = True)

print('Done')

from sklearn.decomposition import PCA,FastICA



pca = PCA(n_components=None)

ica = FastICA(n_components=10)



print("Fit PCA")

pca.fit(xdat[y==1]) #only running on train dataset

print("Fit ICA")

ica.fit(xdat[y==1])



print('PCA explained variance ratio: ',pca.explained_variance_ratio_)
print("Transform PCA")

pcadat=pca.transform(xdat)

print("Transform ICA")

icadat=ica.transform(xdat)



print('PCA: ',np.shape(pcadat))

print('ICA: ',np.shape(icadat))
for i in range(0,57):

    data0 = np.sort(pcadat[y==0,i])

    data1 = np.sort(pcadat[y==1,i])

    n0 = len(data0)

    n1 = len(data1)

    data_all = np.concatenate([data0,data1])

    cdf0 = np.searchsorted(data0,data_all,side='right')/(1.0*n0)

    cdf1 = np.searchsorted(data1,data_all,side='right')/(1.0*n1)

    ks=np.max(np.abs(cdf1-cdf0))

    print( 'len(cdf0)=',len(cdf0),

          ',len(cdf0)=',len(cdf1),

          ',KS Distance:',ks)

    plt.title('i='+str(i)+'\nks=>'+str(ks_2samp(data0,data1)))

    plt.plot(cdf0*ks,color='red' ,alpha=.5,label='istrain=0') # rescale to ks value

    plt.plot(cdf1*ks,color='blue',alpha=.5,label='istrain=1')

    plt.plot(np.abs(cdf1-cdf0),color='black',alpha=.5,label='KS Distance')

    plt.legend()

    plt.show()

pca_dict={}

for i in range(0,57):

    pca_dict[i]=ks_2samp(data0,data1)[1]

    plt.title('i='+str(i)+'\nks=>'+str(ks_2samp(data0,data1)))

    plt.hist(pcadat[y==0,i],color='red' ,alpha=.5,label='istrain=0',bins=100,cumulative=True)

    plt.hist(pcadat[y==1,i],color='blue',alpha=.5,label='istrain=1',bins=100,cumulative=True)

    plt.legend()

    plt.show()

ica_dict={}

for i in range(0,10):

    ica_dict[i]=ks_2samp(icadat[y==0,i],icadat[y==1,i])[1]

    plt.title('i='+str(i)+', ks=>'+str(ks_2samp(icadat[y==0,i],icadat[y==1,i])))

    plt.hist(icadat[y==0,i],color='red',alpha=.5,label='istrain=0',bins=100,cumulative=True)

    plt.hist(icadat[y==1,i],color='blue',alpha=.5,label='istrain=1',bins=100,cumulative=True)

    plt.legend()

    plt.show()

fields_dict={}

for i in xdat.columns:

    fields_dict[i]=ks_2samp(xdat[y==0][i],xdat[y==1][i])[1]

    plt.title('i='+str(i)+', ks=>'+str(ks_2samp(xdat[y==0][i],xdat[y==1][i])))

    plt.hist(xdat[y==0][i],color='red',alpha=.5,label='istrain=0',bins=100,cumulative=True)

    plt.hist(xdat[y==1][i],color='blue',alpha=.5,label='istrain=1',bins=100,cumulative=True)

    plt.legend()

    plt.show()

print('PCA:')

print(pd.Series(pca_dict).sort_values())
print('ICA:')

print(pd.Series(ica_dict).sort_values())
print('RAW:')

print(pd.Series(fields_dict).sort_values())

    
vals=pd.Series(fields_dict).sort_values().reset_index(drop=False)

for i in range(10):

    i=vals.loc[i]['index']

    plt.title('i='+str(i)+'\nks=>'+str(ks_2samp(xdat[y==0][i],xdat[y==1][i]))+"\n"+

             "uniques:"+str(len(xdat[i].unique())))

    plt.hist(xdat[y==0][i],color='red' ,alpha=.5,label='istrain=0',bins=1000,cumulative=True)

    plt.hist(xdat[y==1][i],color='blue',alpha=.5,label='istrain=1',bins=1000,cumulative=True)

    plt.legend()

    plt.show()
