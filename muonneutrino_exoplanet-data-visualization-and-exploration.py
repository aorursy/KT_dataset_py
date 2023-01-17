import pandas as pd

import sklearn as sk

import scipy

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



df = pd.read_csv('../input/exoTrain.csv',index_col=0)
df.head()
labels = df.LABEL

df = df.drop('LABEL',axis=1)
def stats_plots(df):

    means = df.mean(axis=1)

    medians = df.median(axis=1)

    std = df.std(axis=1)

    maxval = df.max(axis=1)

    minval = df.min(axis=1)

    skew = df.skew(axis=1)

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(231)

    ax.hist(means,alpha=0.8,bins=50)

    ax.set_xlabel('Mean Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(232)

    ax.hist(medians,alpha=0.8,bins=50)

    ax.set_xlabel('Median Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(233)

    ax.hist(std,alpha=0.8,bins=50)

    ax.set_xlabel('Intensity Standard Deviation')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(234)

    ax.hist(maxval,alpha=0.8,bins=50)

    ax.set_xlabel('Maximum Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(235)

    ax.hist(minval,alpha=0.8,bins=50)

    ax.set_xlabel('Minimum Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(236)

    ax.hist(skew,alpha=0.8,bins=50)

    ax.set_xlabel('Intensity Skewness')

    ax.set_ylabel('Num. of Stars')



stats_plots(df)

plt.show()
def stats_plots_label(df):

    means1 = df[labels==1].mean(axis=1)

    medians1 = df[labels==1].median(axis=1)

    std1 = df[labels==1].std(axis=1)

    maxval1 = df[labels==1].max(axis=1)

    minval1 = df[labels==1].min(axis=1)

    skew1 = df[labels==1].skew(axis=1)

    means2 = df[labels==2].mean(axis=1)

    medians2 = df[labels==2].median(axis=1)

    std2 = df[labels==2].std(axis=1)

    maxval2 = df[labels==2].max(axis=1)

    minval2 = df[labels==2].min(axis=1)

    skew2 = df[labels==2].skew(axis=1)

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(231)

    ax.hist(means1,alpha=0.8,bins=50,color='b',normed=True,range=(-250,250))

    ax.hist(means2,alpha=0.8,bins=50,color='r',normed=True,range=(-250,250))

    ax.get_legend()

    ax.set_xlabel('Mean Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(232)

    ax.hist(medians1,alpha=0.8,bins=50,color='b',normed=True,range=(-0.1,0.1))

    ax.hist(medians2,alpha=0.8,bins=50,color='r',normed=True,range=(-0.1,0.1))

    ax.get_legend()



    ax.set_xlabel('Median Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(233)    

    ax.hist(std1,alpha=0.8,bins=50,normed=True,color='b',range=(0,4000))

    ax.hist(std2,alpha=0.8,bins=50,normed=True,color='r',range=(0,4000))

    ax.get_legend()



    ax.set_xlabel('Intensity Standard Deviation')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(234)

    ax.hist(maxval1,alpha=0.8,bins=50,normed=True,color='b',range=(-10000,50000))

    ax.hist(maxval2,alpha=0.8,bins=50,normed=True,color='r',range=(-10000,50000))

    ax.get_legend()



    ax.set_xlabel('Maximum Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(235)

    ax.hist(minval1,alpha=0.8,bins=50,normed=True,color='b',range=(-50000,10000))

    ax.hist(minval2,alpha=0.8,bins=50,normed=True,color='r',range=(-50000,10000))

    ax.get_legend()



    ax.set_xlabel('Minimum Intensity')

    ax.set_ylabel('Num. of Stars')

    ax = fig.add_subplot(236)

    ax.hist(skew1,alpha=0.8,bins=50,normed=True,color='b',range=(-40,60))

    ax.hist(skew2,alpha=0.8,bins=50,normed=True,color='r',range=(-40,60)) 

    ax.get_legend()



    ax.set_xlabel('Intensity Skewness')

    ax.set_ylabel('Num. of Stars')



stats_plots_label(df)

plt.show()
df[labels==1].median(axis=1).describe()
fig = plt.figure(figsize=(12,40))

x = np.array(range(3197))

for i in range(37):

    ax = fig.add_subplot(13,3,i+1)

    ax.scatter(x,df[labels==2].iloc[i,:])
fig = plt.figure(figsize=(12,40))

x = np.array(range(3197))

for i in range(37):

    ax = fig.add_subplot(13,3,i+1)

    ax.scatter(x,df[labels==1].iloc[i,:])
fig = plt.figure(figsize=(12,4))

ax = fig.add_subplot(121)

plt.scatter(x,df[labels==2].iloc[35,:])

ax = fig.add_subplot(122)

plt.scatter(np.array(range(500)),df[labels==2].iloc[35,:500])

plt.show()
fig = plt.figure(figsize=(12,4))

ax = fig.add_subplot(121)

plt.scatter(x,df[labels==2].iloc[18,:])

ax = fig.add_subplot(122)

plt.scatter(x,df[labels==2].iloc[30,:])

plt.show()
fig = plt.figure(figsize=(10,15))

ax = fig.add_subplot(311)

ax.scatter(x,df[labels==2].iloc[9,:])

ax = fig.add_subplot(312)

ax.scatter(np.array(range(2500,3000)),df[labels==2].iloc[9,2500:3000])

ax = fig.add_subplot(313)

ax.scatter(np.array(range(1200,1700)),df[labels==2].iloc[9,1200:1700])

plt.show()
def reduce_upper_outliers(df,reduce = 0.01, half_width=4):

    length = len(df.iloc[0,:])

    remove = int(length*reduce)

    for i in df.index.values:

        values = df.loc[i,:]

        sorted_values = values.sort_values(ascending = False)

       # print(sorted_values[:30])

        for j in range(remove):

            idx = sorted_values.index[j]

            #print(idx)

            new_val = 0

            count = 0

            idx_num = int(idx[5:])

            #print(idx,idx_num)

            for k in range(2*half_width+1):

                idx2 = idx_num + k - half_width

                if idx2 <1 or idx2 >= length or idx_num == idx2:

                    continue

                new_val += values['FLUX-'+str(idx2)]

                

                count += 1

            new_val /= count # count will always be positive here

            #print(new_val)

            if new_val < values[idx]: # just in case there's a few persistently high adjacent values

                df.set_value(i,idx,new_val)

        

            

    return df
df_exo = df[labels==2]

df_non = df[labels==1]

df_non = df_non.sample(n=100,random_state=999)

for i in range(2):

    df_exo = reduce_upper_outliers(df_exo)

for i in range(2):

    df_non = reduce_upper_outliers(df_non)
fig = plt.figure(figsize=(12,40))

x = np.array(range(3197))

for i in range(37):

    ax = fig.add_subplot(13,3,i+1)

    ax.scatter(x,df_exo.iloc[i,:])
fig = plt.figure(figsize=(12,40))

x = np.array(range(3197))

for i in range(37):

    ax = fig.add_subplot(13,3,i+1)

    ax.scatter(x,df_non.iloc[i,:])
from scipy.signal import savgol_filter

from scipy.signal import gaussian

from scipy.signal import medfilt

from scipy.signal import lfilter



test = [0,7,11,12,31,34]

nfigs = 2 * len(test)

fig = plt.figure(figsize=[13,50])

count = 1

for i in test:

    ax = fig.add_subplot(nfigs,3,count)

    ax.scatter(np.array(range(len(df_exo.iloc[i,:]))),df_exo.iloc[i,:])

    count += 1

    y0 = medfilt(df_exo.iloc[i,:],41)

    for idx in range(len(y0)):

        y0[idx] = df_exo.iloc[i,idx] - y0[idx]

    y1 = savgol_filter(y0,21,4,deriv=0)

    ax = fig.add_subplot(nfigs,3,count)

    count += 1

    ax.scatter( np.array(range(len(y0))),y0)

    ax.set_label('Sample')

    ax.set_ylabel('Gaussian Smoothing')

    ax.set_title('Exoplanet Star '+str(i))

    

    ax = fig.add_subplot(nfigs,3,count)

    count += 1

    ax.scatter( np.array(range(len(y1)-40)),y1[20:-20])

    ax.set_label('Sample')

    ax.set_ylabel('Savitzky-Golay Estimate, 1st derivative')

    ax.set_title('Exoplanet Star '+str(i))

    

plt.show()
def short_transit_filter(df):



    length = df.shape[0]

    output = []

    for i in range(length):



        y0 = medfilt(df.iloc[i,:],41)

        for idx in range(len(y0)):

            y0[idx] = df.iloc[i,idx] - y0[idx]

        y1 = savgol_filter(y0,21,4,deriv=0) # remove edge effects

        output.append(y1)

    

    return output

    
out_exo = short_transit_filter(df_exo)

out_non = short_transit_filter(df_non)
fig = plt.figure(figsize=(13,40))

x = np.array(range(len(out_exo[0])-24))

for i in range(37):

    ax = fig.add_subplot(13,3,i+1)

    ax.scatter(x,out_exo[i][12:-12])
fig = plt.figure(figsize=(13,40))

x = np.array(range(len(out_exo[0])-24))

for i in range(37):

    ax = fig.add_subplot(13,3,i+1)

    ax.scatter(x,out_non[i][12:-12])
## After filtering



df_exo_filt = pd.DataFrame(out_exo)

df_non_filt = pd.DataFrame(out_non)
means2 = df_exo_filt.mean(axis=1)

std2 = df_exo_filt.std(axis=1)

medians2 = df_exo_filt.median(axis=1)

means1 = df_non_filt.mean(axis=1)

std1 = df_non_filt.std(axis=1)

medians1 = df_non_filt.median(axis=1)
fig = plt.figure(figsize=(10,10))



ax = fig.add_subplot(221)

ax.hist(means1,color='b',range=(-300,100),bins=20)

ax.hist(means2,color='r',range=(-300,100),bins=20)

ax.set_xlabel('Mean Intensity')

ax = fig.add_subplot(222)

ax.hist(medians1,color='b',range=(-50,50),bins=20)

ax.hist(medians2,color='r',range=(-50,50),bins=20)

ax.set_xlabel('Median Intensity')

ax = fig.add_subplot(223)

ax.hist(std1,color='b',range=(0,500),bins=10)

ax.hist(std2,color='r',range=(0,500),bins=10)

ax.set_xlabel('Intensity Std. Dev.')



plt.show()