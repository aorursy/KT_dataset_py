import numpy as np # linear algebra

import pandas as pd # data processing

from sklearn.preprocessing import Imputer

from sklearn.neighbors import NearestNeighbors 

import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns

sns.set_style('whitegrid')
aguah = pd.read_csv("../input/AguaH.csv")

aguah.head()

#aguah.info()
lookup = {'ENE':'01','FEB':'02','MAR':'03','ABR':'04','MAY':'05','JUN':'06','JUL':'07','AGO':'08','SEP':'09','OCT':'10','NOV':'11','DIC':'12'}

clist=[]

for col in aguah.columns[5:]:

    col = col[4:].split('_')

    clist.append('20'+col[1]+'-'+lookup[col[0]])



columns = ['LANDUSE_TYPE','USER','PIPE DIAM','VENDOR','JAN16']+clist

aguah.columns=columns

aguah.head()
## Countplot the number of NaN values for each entry

cons = aguah.iloc[0:, 5:]

cons['NumNull'] = cons.isnull().sum(axis=1)

print('The proportion of entries with non-NaN values is {:.2f}%'.format(len(cons[cons.NumNull==0])/len(cons)*100))

bins = [0,1,10,20,30,40,50,60,70,80,90]

cut = pd.cut(cons.NumNull, bins=bins, include_lowest=True, right=False)

fig, axis1 = plt.subplots(1,figsize=(8,4))

sns.countplot(x=cut, data=cut.to_frame(), ax=axis1)



sns.plt.show()
## Plot graphs to show how values for each entry evolves with time

NumNullwithTime = cons.drop('NumNull', axis=1).isnull().sum()



sns.set_style("darkgrid")

plt.figure(figsize=(10,4))

pbar = NumNullwithTime.plot.bar()

plt.xticks(list(range(0,len(NumNullwithTime.index),6)), list(NumNullwithTime.index[0::6]), rotation=45, ha='right')

plt.show()
## Return index of column (0-83) where the first non-NA number appears. If none, return 84

def FirstNonNull(row):

    count=0

    for col in row:

        if col==False: return count

        else: count = count+1

    return count



## Return index of column (0-83) where the last non-NA number appears, If none, return -1

def LastNonNull(row):

    count=0

    flag=-1

    for col in row:

        if col==False:

            flag=count

            count=count+1

        else: count=count+1

    return flag
## I need this function for the cases of all NaN entries (NullinService value becomes 0 from -83)

def Setzero(x):

    if x<0: return 0

    else: return x



## Number of NaN values before service period

groupnull = aguah.iloc[:,0:5]

groupnull = pd.concat([groupnull, cons], axis=1)



groupnull['FirstNonNull'] = cons.copy().drop(['NumNull'], axis=1).isnull().apply(FirstNonNull,axis=1)

groupnull['LastNonNull'] = cons.copy().drop(['NumNull'],axis=1).isnull().apply(LastNonNull,axis=1)

groupnull['NullInService'] = groupnull.NumNull - groupnull.FirstNonNull - (len(cons.columns)-1-groupnull.LastNonNull) +1    

groupnull['NullInService'] = groupnull['NullInService'].apply(Setzero)

groupnull.NullInService.value_counts(sort=False).head(6) ## Print only head values
contLong = groupnull[groupnull.NumNull==0]

contShort = groupnull[(groupnull.NumNull>0) & (groupnull.NullInService==0)]

interupted = groupnull[(groupnull.NullInService)>0]

print('Length of 3 groups: (Non-NA Group, Edge-NA Group, Interupted Group) = ({}, {}, {})'.format(len(contLong), len(contShort), len(interupted)))
## Test various imputation methods using the group of complete entries (Non-NA Group above)

rng = np.random.RandomState()

missing_rate = 0.01  ## Here, 1 for use the whole set of entries to score (0.01 only for display)



## Prepare a scoring set within Non-NA Group 

num_total = len(contLong)

num_score = int(np.floor(missing_rate*num_total))

missing_samp = np.hstack((np.zeros(num_total-num_score, dtype=np.bool), np.ones(num_score, dtype=np.bool)))

rng.shuffle(missing_samp)

cl_score = contLong.iloc[:,5:89][missing_samp.tolist()]



## Columns where holes to be made

col = rng.randint(0, 84, num_score)



## Save the answer set before making the "holes" in the scoring group (1 hole (missing data) per row)

cl_score_orig = cl_score.copy() ## save the original for KNN method (for reference)

answer = cl_score.as_matrix()[np.arange(num_score), col]

cl_score.as_matrix()[np.arange(num_score), col] = np.nan



## Function for scoring (squared mean error by answer and imputed values)

def Impute_error(imputed, answer):

    return np.sqrt(np.square(imputed-answer).sum())/len(answer)
## Start with simple imputation methods: Mean, Median, Most frequent value, Forward fill, and Backward fill

cl_score_mean = cl_score.copy()

imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=1, copy=False)

imp_mean.fit_transform(cl_score_mean)

imputed_mean = cl_score_mean.as_matrix()[np.arange(num_score), col]

                  

cl_score_median = cl_score.copy()

imp_median = Imputer(missing_values='NaN', strategy='median', axis=1, copy=False)

imp_median.fit_transform(cl_score_median)

imputed_median = cl_score_median.as_matrix()[np.arange(num_score), col]



cl_score_mfre = cl_score.copy()

imp_mfre = Imputer(missing_values='NaN', strategy='most_frequent', axis=1, copy=False)

imp_mfre.fit_transform(cl_score_mfre)

imputed_mfre = cl_score_mfre.as_matrix()[np.arange(num_score), col]



## NaN values at head can't be filled with ffill so complement with bfill

cl_score_ffill = cl_score.copy()

cl_score_ffill = cl_score_ffill.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1) 

imputed_ffill = cl_score_ffill.as_matrix()[np.arange(num_score), col]



## NaN values at tail can't be filled with bfill so complement with ffill

cl_score_bfill = cl_score.copy()

cl_score_bfill = cl_score_bfill.fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)

imputed_bfill = cl_score_bfill.as_matrix()[np.arange(num_score), col]



display(Impute_error(imputed_mean, answer))

display(Impute_error(imputed_median, answer))

display(Impute_error(imputed_mfre, answer))

display(Impute_error(imputed_ffill, answer))

display(Impute_error(imputed_bfill, answer))



## Scores when missing_rate is 1 (use the entire Non-NA Group)

## 0.73146667241762797

## 0.74416261219896196

## 1.2776284010594683

## 0.43119183851023574

## 0.44032679793299145
## Interpolate() in Pandas



cl_score_linear = cl_score.copy()

cl_score_linear.interpolate(method='linear', axis=1, inplace=True)

cl_score_linear = cl_score_linear.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)

imputed_linear = cl_score_linear.as_matrix()[np.arange(num_score), col] ## cl_score_linear.values works too



cl_score_akima = cl_score.copy()

cl_score_akima.columns=list(range(cl_score_akima.shape[1]))

cl_score_akima.interpolate(method='akima', axis=1, inplace=True)

cl_score_akima = cl_score_akima.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)

imputed_akima = cl_score_akima.as_matrix()[np.arange(num_score), col]



cl_score_spline = cl_score.copy()

cl_score_spline.columns=list(range(cl_score_spline.shape[1]))

cl_score_spline.interpolate(method='spline', axis=1, order=2, inplace=True)

cl_score_spline = cl_score_spline.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)

imputed_spline = cl_score_spline.as_matrix()[np.arange(num_score), col]





display(Impute_error(imputed_linear, answer))

display(Impute_error(imputed_akima, answer))

display(Impute_error(imputed_spline, answer))



## Scores when missing_rate is 1 (use the entire Non-NA Group - Warning. It may take several minutes to complete)

## 0.33370675683889633

## 0.33465191332331751

## 1.8506789840447648
## Imputation using K-nearest neighbor (to find the k closest row(s) in the sample set to the row x in question)

def KnnImputeSimple(sample, x, k):

    ## Mask the columns with NaN value (not to compare)

    x_mask = x.notnull().tolist()

    x_mask_toggled = x.isnull().tolist()

    sample_masked = sample.iloc[:,x_mask]

    x_masked = x[x_mask]

    

    ## Extent to which column comparison is carried out. Here 6 columns (months) before and after the the column in question

    comp_size = 6

    i = x_mask_toggled.index(True)

    ## Handle when there are less than 4 columns to look at before or after the column in question

    i = max(comp_size, min(i, 84-1-comp_size))

    x_masked = x_masked[i-comp_size:i+comp_size].values

    sample_masked = sample_masked.iloc[:,i-comp_size:i+comp_size]

    

    ## I use kd_tree algorithm here.

    nbrs = NearestNeighbors(k, algorithm='kd_tree', n_jobs=-1)

    nbrs.fit(sample_masked)

    n_ones = nbrs.kneighbors([x_masked])

    

    ## Find k nearest ones and average the predicted values to return

    value = []

    for n in range(x.isnull().sum()):

        temp=[]

        for j in range(k):

            temp.append(sample.iloc[n_ones[1][0][j]][x_mask_toggled][n])

        value.append(np.sum(temp)/k)



    return value
## Setting for K-Nearest Neighbor method  test

div = int(num_score/2)

set1 = cl_score.copy()[0:div]

set2 = cl_score.copy()[div:2*div]



answer_set1 = answer[:div]

answer_set2 = answer[div:2*div]  

    

imputed_knn_set1 = set1.apply(lambda x: KnnImputeSimple(cl_score_orig[div:2*div], x, 2)[0], axis=1)

imputed_knn_set2 = set2.apply(lambda x: KnnImputeSimple(cl_score_orig[:div], x, 2)[0], axis=1)

imputed_knn = imputed_knn_set1.append(imputed_knn_set2)

answer = np.append(answer_set1, answer_set2)



display(Impute_error(imputed_knn, answer))



## Scores when missing_rate is 1 with various variables k and comp_size (use the entire Non-NA Group)

## Warning - It may take several hours to complete if you do it with the entire Non-NA Group i.e. missing_rate=1

## 0.33834147148949706 (k=4, comp_size=4)

## 0.30221719764528121 (k=5, comp_size=3)

## 0.29975805378644343 (k=2, comp_size=6)
## Carry out imputation work using KnnImputeSimple function defined above

def KnnImputeInPlace(sample, itrp, k):

    fn = itrp['FirstNonNull']

    ln = itrp['LastNonNull']

    serv = itrp[int(fn):int(ln)+1]

    

    ## Indicate the index of NaNs in row

    index = serv[serv.isnull()].index.tolist()

    itrp_imputed = itrp

    for i in range(len(index)):

        idx = index.copy()

        ## Ignore the other NaN columns for i-th NaN calculation

        idx.pop(i)

        imp = KnnImputeSimple(sample.iloc[:,int(fn):int(ln)+1].drop(sample[idx],axis=1), serv.drop(idx), k)

        itrp_imputed.set_value(index[i],imp[0])

    

    return itrp_imputed
## For KNN, choose the size of sample table (same as missing_rate above)

rng = np.random.RandomState(0)

samp_rate = 0.01  ## Here, 1 for use the whole set of entries to search (0.01 only for display)



## Prepare a reference set within Non-NA Group 

num_total = len(contLong)

num_samp = int(np.floor(samp_rate*num_total))

rand_samp = np.hstack((np.zeros(num_total-num_samp, dtype=np.bool), np.ones(num_samp, dtype=np.bool)))

rng.shuffle(rand_samp)

sample = contLong.iloc[:,5:89][rand_samp.tolist()]
## KNN imputation (warning it may take a couple of minutes)

itrp = interupted.copy()[(interupted.NullInService>0) & (interupted.NullInService<6)].iloc[:,5:]

result_knn = itrp.apply(lambda x: KnnImputeInPlace(sample, x, 2), axis=1)

result_knn_head = interupted.copy()[(interupted.NullInService>0) & (interupted.NullInService<6)].iloc[:,0:5]

result_knn = pd.concat([result_knn_head, result_knn],axis=1)

result_knn.shape
## Linear imputation 

itrp = interupted.copy()[(interupted.NullInService>5) & (interupted.NullInService<84)].iloc[:,5:]

itrp.interpolate(method='linear', axis=1, inplace=True)

result_fill_head = interupted.copy()[(interupted.NullInService>5) & (interupted.NullInService<84)].iloc[:,0:5]

result_fill = pd.concat([result_fill_head, itrp],axis=1)

result_fill.shape
result = pd.concat([contLong, contShort, result_knn, result_fill])

result = result.iloc[:,:89]

result.shape
## Plot a few imputed graphs by highlighting imputed values in red

np.random.seed(14)

num_example = 10

pltnum = np.random.randint(0,len(interupted), num_example)

before = interupted.iloc[pltnum]



idx = before.index[list(range(num_example))]

after =result.ix[idx]

before = before.iloc[:,5:89]

after = after.iloc[:,5:89]



%matplotlib inline

fig, ax = plt.subplots(5,2, figsize=(10,10))



for i in range(num_example):

    #plt.plot(before, color='r')

    after.iloc[i].plot(kind='line',color='r', alpha=0.8, ax=ax[i//2,i%2])

    #plt.subplot(5,2,i+1)

    before.iloc[i].plot(kind='line',color='b', linewidth=1.5, alpha=1, ax=ax[i//2,i%2])

    #plt.subplot(5,2,i+1)

    

plt.setp(ax, xticks=list(range(6,len(NumNullwithTime.index),24)), xticklabels=list(NumNullwithTime.index[6::24]))

plt.tight_layout()

plt.show()