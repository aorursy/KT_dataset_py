import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os
sub_path = "../input/jigsaw-multilingual-submission-files"

all_files = os.listdir(sub_path)

all_files
# no 0.9361 led from LB 0.9471 to 0.9472

# no 0.9361 and 0.9383 led from LB 0.9472 to 0.9475

# no 0.9361 and 0.9383 and 0.9416 led from LB 0.9475 to 0.9468

# no 0.9361 and 0.9383 and 0.9459b led from LB 0.9475 to 0.9474

# no 0.9361 and 0.9383 and 0.9459 led from LB 0.9475 to 0.9473

# no 0.9361 and 0.9383, and added in 0.9422 from MLM (mixed language models) led from LB 0.9475 to 0.9477

# no 0.9361 and 0.9383, and added in 0.9422 and 0.9432 from MLM (mixed language models) led from LB 0.9477 to 0.9478

# no 0.9361 and 0.9383, and added in 0.9432 from MLM (mixed language models) and removing 0.9422 from MLM led from LB 0.9478 to 0.9479

# (seems like removing very correlated lower LB results improves LB!)

# ok removed 0.9459 which is very correlated to the others but LB dropped..

# found 0.9428 for MLM also, added in and no 0.9361 and 0.9383, and added in 0.9432 & 0.9428 from MLM (mixed language models) led from LB 0.9479 to 0.9480

# added 0.9431 MLM but LB dropped from 0.9480 to 0.9479

# added 0.9404 parcor regularization (https://www.kaggle.com/aiaiooas/parcor-regularised-classification) which led to 0.9482

# added 0.9366 wide and shallow CNN but dropped from 0.9482 to 0.9481

# removed 0.9322 but dropped from 0.9482 to 0.9481

# added 0.9426 MLM submission but dropped from 0.9482 to 0.9481

# added second parcor sub with 0.9414 but remained at 0.9482

# added 0.9409 parcor sub and LB increased from 0.9482 to 0.9485

all_files = [f for f in all_files if '0.9361' not in f and '0.9383' not in f and '0.9422' not in f] #+ _all_files

all_files
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "jigsaw" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
# check correlation

concat_sub.iloc[:,1:ncol].corr()
corr = concat_sub.iloc[:,1:].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# get the data fields ready for stacking

concat_sub['jigsaw_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)

concat_sub['jigsaw_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)

concat_sub['jigsaw_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['jigsaw_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.describe()
cutoff_lo = 0.3

cutoff_hi = 0.7
concat_sub['toxic'] = concat_sub['jigsaw_mean']

concat_sub[['id', 'toxic']].to_csv('submission1.csv', 

                                        index=False, float_format='%.6f')
concat_sub[['toxic']].describe()
plt.hist(concat_sub['jigsaw_mean'],bins=100)

plt.show()
concat_sub['toxic'] = concat_sub['jigsaw_median']

concat_sub[['id', 'toxic']].to_csv('submission2.csv', 

                                        index=False, float_format='%.6f')
concat_sub[['toxic']].describe()
plt.hist(concat_sub['jigsaw_median'],bins=100)

plt.show()
concat_sub['toxic'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, concat_sub['jigsaw_median']))

concat_sub[['id', 'toxic']].to_csv('submission3.csv', 

                                        index=False, float_format='%.6f')
plt.hist(concat_sub['toxic'],bins=100)

plt.show()
concat_sub['toxic'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['jigsaw_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['jigsaw_min'], 

                                             concat_sub['jigsaw_mean']))

concat_sub[['id', 'toxic']].to_csv('submission.csv', 

                                        index=False, float_format='%.6f')
plt.hist(concat_sub['toxic'],bins=100)

plt.show()
concat_sub['toxic'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['jigsaw_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['jigsaw_min'], 

                                             concat_sub['jigsaw_median']))

concat_sub[['id', 'toxic']].to_csv('submission5.csv', 

                                        index=False, float_format='%.6f')
plt.hist(concat_sub['toxic'],bins=100)

plt.show()
all_files
df1 = [f for f in all_files if '0.9462' in f][0]

df2 = [f for f in all_files if '0.9459b' in f][0]

df3 = [f for f in all_files if '0.9459' in f and '0.9459b' not in f][0]

df4 = [f for f in all_files if '0.9450' in f][0]

df5 = [f for f in all_files if '0.9432' in f][0]

df6 = [f for f in all_files if '0.9428' in f][0]

df7 = [f for f in all_files if '0.9428b' in f][0]

df8 = [f for f in all_files if '0.9416' in f][0]

df9 = [f for f in all_files if '0.9414' in f and '0.9414b' not in f][0]

df10 = [f for f in all_files if '0.9414b' in f][0]

df11 = [f for f in all_files if '0.9409' in f][0]

df12 = [f for f in all_files if '0.9322' in f][0]
df1
df1 = pd.read_csv(os.path.join(sub_path, df1))

df2 = pd.read_csv(os.path.join(sub_path, df2))

df3 = pd.read_csv(os.path.join(sub_path, df3))

df4 = pd.read_csv(os.path.join(sub_path, df4))

df5 = pd.read_csv(os.path.join(sub_path, df5))

df6 = pd.read_csv(os.path.join(sub_path, df6))

df7 = pd.read_csv(os.path.join(sub_path, df7))

df8 = pd.read_csv(os.path.join(sub_path, df8))

df9 = pd.read_csv(os.path.join(sub_path, df9))

df10 = pd.read_csv(os.path.join(sub_path, df10))

df11 = pd.read_csv(os.path.join(sub_path, df11))

df12 = pd.read_csv(os.path.join(sub_path, df12))

df1.head()
sub = df1.copy()[['id']]
W = 0.9462 + 2*0.9459 + 0.9450 + 2*0.9428 + 0.9422 + 0.9416 + 2*0.9414 + 0.9409 + 0.9322

w1 = 0.9462/W

w2 = 0.9459/W

w3 = 0.9459/W

w4 = 0.9450/W

w5 = 0.9432/W

w6 = 0.9428/W

w7 = 0.9428/W

w8 = 0.9416/W

w9 = 0.9414/W

w10 = 0.9414/W

w11 = 0.9409/W

w12 = 0.9322/W
sub['toxic'] = (w1*df1['toxic'] + 

                w2*df2['toxic'] + 

                w3*df3['toxic'] + 

                w4*df4['toxic'] + 

                w5*df5['toxic'] + 

                w6*df6['toxic'] + 

                w7*df7['toxic'] + 

                w8*df8['toxic'] + 

                w9*df9['toxic'] + 

                w10*df10['toxic'] + 

                w11*df11['toxic'] + 

                w12*df12['toxic'])

sub.head()
plt.hist(sub.toxic,bins=100)

plt.show()
sub.to_csv('submission_blend.csv', index=False)

#sub.to_csv('submission.csv', index=False)
from scipy.stats import gaussian_kde

from math import sqrt



class DistanceMetrics:

    '''

    - non-built-in distance metrics are found here

    - work in progress

    '''



    @staticmethod

    def get_density(x, cov_factor=0.1):

        #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.

        density = gaussian_kde(x)

        density.covariance_factor = lambda:cov_factor

        density._compute_covariance()

        return density

    

    @classmethod

    def battacharyya(cls, X1, X2, method='continuous'):

        '''

        Original Author: Eric Williamson (ericpaulwill@gmail.com)

        Obtained from: https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py

        - This calculates the Bhattacharyya distance between vectors X1 and X2. X1 and X2 should be 1D numpy arrays representing the same

          feature in two separate classes.

        '''

        #Combine X1 and X2, we'll use it later:

        cX = np.concatenate((X1,X2))

        if method == 'noiseless':

            ###This method works well when the feature is qualitative (rather than quantitative). Each unique value is

            ### treated as an individual bin.

            uX = np.unique(cX)

            A1 = len(X1) * (max(cX)-min(cX)) / len(uX)

            A2 = len(X2) * (max(cX)-min(cX)) / len(uX)

            bht = 0

            for x in uX:

                p1 = (X1==x).sum() / A1

                p2 = (X2==x).sum() / A2

                bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)



        elif method == 'hist':

            ###Bin the values into a hardcoded number of bins (This is sensitive to N_BINS)

            N_BINS = int(len(X1) * 2)

            #Bin the values:

            h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]

            h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]

            #Calc coeff from bin densities:

            bht = 0

            for i in range(N_BINS):

                p1 = h1[i]

                p2 = h2[i]

                bht += sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS



        elif method == 'autohist':

            ###Bin the values into bins automatically set by np.histogram:

            #Create bins from the combined sets:

            # bins = np.histogram(cX, bins='fd')[1]

            bins = np.histogram(cX, bins='doane')[1] #Seems to work best

            # bins = np.histogram(cX, bins='auto')[1]



            h1 = np.histogram(X1,bins=bins, density=True)[0]

            h2 = np.histogram(X2,bins=bins, density=True)[0]



            #Calc coeff from bin densities:

            bht = 0

            for i in range(len(h1)):

                p1 = h1[i]

                p2 = h2[i]

                bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)



        elif method == 'continuous':

            ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):

            N_STEPS = int(len(X1) * 20)

            #Get density functions:

            d1 = cls.get_density(X1)

            d2 = cls.get_density(X2)

            #Calc coeff:

            xs = np.linspace(min(cX),max(cX),N_STEPS)

            bht = 0

            for x in xs:

                p1 = d1(x)

                p2 = d2(x)

                bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS



        else:

            raise ValueError("The value of the 'method' parameter does not match any known method")



        ###Lastly, convert the coefficient into distance:

        if bht==0:

            return float('Inf')

        else:

            return -np.log(bht)
from scipy.spatial.distance import euclidean, cosine, jaccard, chebyshev, correlation, cityblock, canberra, braycurtis, hamming



def get_dist_preds(predictions, metric):

    new_preds = []

    for j, pred in enumerate(predictions):

        distances = []

        remaining_preds = predictions[:j] + predictions[j+1:]

        for pred_ in remaining_preds:

            if metric == 'euclid':

                distances += [euclidean(pred_, pred)]

            elif metric == 'cosine':

                distances += [cosine(pred_, pred)]

            elif metric == 'jaccard': # i think this is only for boolean

                distances += [jaccard(pred_, pred)]

            elif metric == 'chebyshev':

                distances += [chebyshev(pred_, pred)]

            elif metric == 'correlation':

                distances += [correlation(pred_, pred)]

            elif metric == 'cityblock':

                distances += [cityblock(pred_, pred)]

            elif metric == 'canberra':

                distances += [canberra(pred_, pred)]

            elif metric == 'braycurtis':

                distances += [braycurtis(pred_, pred)]

            elif metric == 'hamming': # i think this is only for boolean

                distances += [hamming(pred_, pred)]

            elif metric == 'battacharyya':

                distances += [DistanceMetrics.battacharyya(pred_, pred, method='continuous')]

        new_preds += [(pred, sum(distances))] # (precdictions, weight)



    weights = [tup[1] for tup in new_preds]

    W = sum(weights) # total weight

    # those with lower distances have higher weight

    # sort in ascending order of aggregated distances

    preds_ascending_dist = sorted(new_preds, key=lambda x: x[1])

    weights_descending = sorted(weights, reverse=True)

    weighted_pred = sum([pred_tup[0]*(weights_descending[k]/W) for k, pred_tup in enumerate(preds_ascending_dist)])

    return weighted_pred
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]

preds = [np.array(df['toxic']) for df in dfs]
df_deboost = get_dist_preds(preds, 'correlation')
_sub = df1.copy()[['id']]

_sub['toxic'] = df_deboost

_sub.head()
plt.hist(_sub.toxic,bins=100)

plt.show()
_sub.to_csv('submission_deboost.csv', index=False)

#_sub.to_csv('submission.csv', index=False)
n_folds = len(dfs)

n_folds
nrow = len(dfs[0])

nrow
# create fold partition indices

num_row_per_df = nrow//n_folds

num_row_per_df
remainder_rows = nrow - n_folds * num_row_per_df

remainder_rows
end_indices = [num_row_per_df*i for i in range(1,n_folds)] + [nrow]

end_indices
partitions = [[[]]*n_folds]*n_folds

partitions
data = {}



for i, df in enumerate(dfs):

    data[str(i)] = df



ranks = pd.DataFrame(columns=data.keys())

for key in data.keys():

    ranks[key] = data[key].toxic.rank(method='min')

ranks['Average'] = ranks.mean(axis=1)

ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())

ranks.corr()[:1]
!ls ../input/jigsaw-multilingual-toxic-comment-classification
weights = [0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]

ranks['Score'] = ranks[[str(i) for i in range(n_folds)]].mul(weights).sum(1) / ranks.shape[0]

submission_lb = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

submission_lb['toxic'] = ranks['Score']

submission_lb.to_csv("WEIGHT_AVERAGE_RANK.csv", index=False)

#submission_lb.to_csv("submission.csv", index=False)

submission_lb.head()
plt.hist(submission_lb.toxic,bins=100)

plt.show()
'''

Recall

------

df1 = [f for f in all_files if '0.9462' in f][0]

df2 = [f for f in all_files if '0.9459b' in f][0]

df3 = [f for f in all_files if '0.9459' in f and '0.9459b' not in f][0]

df4 = [f for f in all_files if '0.9450' in f][0]

df5 = [f for f in all_files if '0.9432' in f][0]

df6 = [f for f in all_files if '0.9428' in f][0]

df7 = [f for f in all_files if '0.9416' in f][0]

df8 = [f for f in all_files if '0.9322' in f][0]

'''
labels = ["toxic"]

for label in labels:

    print(label)

    print(np.round(np.corrcoef([df[label].rank(pct=True) for df in dfs]), 4))
submission = dfs[0][['id']]

for label in labels:

    preds = []

    submission[label] = (

        0.15*(0.9*dfs[0][label] + 0.1*dfs[5][label]) + 

        0.15*(0.9*dfs[1][label] + 0.1*dfs[5][label]) + 

        0.15*(0.9*dfs[2][label] + 0.1*dfs[7][label]) + 

        0.15*(0.9*dfs[3][label] + 0.1*dfs[7][label]) + 

        0.1*(0.5*dfs[4][label] + 0.5*dfs[5][label]) + 

        0.1*(0.5*dfs[5][label] + 0.5*dfs[4][label]) + 

        0.1*(0.1*dfs[6][label] + 0.9*dfs[4][label]) + 

        0.1*(0.5*dfs[7][label] + 0.5*dfs[5][label])

    )
plt.hist(submission.toxic,bins=100)

plt.show()
submission.head()
submission.to_csv('submission_correlations.csv', index=False)

#submission.to_csv('submission.csv', index=False)
'''

all_files = 

['submission_0.9450.csv',

 'submission_0.9459.csv',

 'submission_0.9462.csv',

 'submission_0.9416.csv',

 'submission_0.9432.csv',

 'submission_0.9428b.csv',

 'submission_0.9414.csv',

 'submission_0.9428.csv',

 'submission_0.9459b.csv',

 'submission_0.9322.csv']

'''
df1 = [f for f in all_files if '0.9462' in f][0]

df2 = [f for f in all_files if '0.9459b' in f][0]

df3 = [f for f in all_files if '0.9459' in f and '0.9459b' not in f][0]

df4 = [f for f in all_files if '0.9450' in f][0]

df5 = [f for f in all_files if '0.9432' in f][0]

df6 = [f for f in all_files if '0.9428b' in f][0]

df7 = [f for f in all_files if '0.9428' in f][0]

df8 = [f for f in all_files if '0.9416' in f][0]

df9 = [f for f in all_files if '0.9414' in f][0]

df10 = [f for f in all_files if '0.9322' in f][0]

df1 = pd.read_csv(os.path.join(sub_path, df1))

df2 = pd.read_csv(os.path.join(sub_path, df2))

df3 = pd.read_csv(os.path.join(sub_path, df3))

df4 = pd.read_csv(os.path.join(sub_path, df4))

df5 = pd.read_csv(os.path.join(sub_path, df5))

df6 = pd.read_csv(os.path.join(sub_path, df6))

df7 = pd.read_csv(os.path.join(sub_path, df7))

df8 = pd.read_csv(os.path.join(sub_path, df8))

df9 = pd.read_csv(os.path.join(sub_path, df9))

df10 = pd.read_csv(os.path.join(sub_path, df10))
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
p = 1.5 # power
labels = ["toxic"]

for label in labels:

    print(label)

    print(np.round(np.corrcoef([df[label].rank(pct=True) for df in dfs]), 4))
sub_power = submission[['id']].copy()

sub_power['toxic'] = (df1['toxic']**p * 0.2 + 

                      df2['toxic']**p * 0.1 + 

                      df3['toxic']**p * 0.1 + 

                      df4['toxic']**p * 0.15 + 

                      df5['toxic']**p * 0.25 + 

                      df6['toxic']**p * 0.025 + 

                      df7['toxic']**p * 0.025 + 

                      df8['toxic']**p * 0.05 + 

                      df9['toxic']**p * 0.05 + 

                      df10['toxic']**p * 0.05)
sub_power[['toxic']].describe()
sub_power.head()
plt.hist(sub_power.toxic,bins=100)

plt.show()
sub_power.to_csv('submission_power_ensemble.csv', index=False)

#sub_power.to_csv('submission.csv', index=False)
sub = submission[['id']].copy()
sub['toxic'] = 0.1*df10['toxic'] + 0.9*df9['toxic']

sub['toxic'] = 0.51*df9['toxic'] + 0.49*sub['toxic']



sub['toxic_temp'] = 0.1*df8['toxic'] + 0.9*df7['toxic']

sub['toxic_temp'] = 0.51*df7['toxic'] + 0.49*sub['toxic_temp']

sub['toxic'] = 0.5*sub['toxic_temp'] + 0.5*sub['toxic']



sub['toxic_temp'] = 0.1*df6['toxic'] + 0.9*df5['toxic']

sub['toxic_temp'] = 0.51*df5['toxic'] + 0.49*sub['toxic_temp']

sub['toxic'] = 0.5*sub['toxic_temp'] + 0.5*sub['toxic']



sub['toxic_temp'] = 0.1*df6['toxic'] + 0.9*df5['toxic']

sub['toxic_temp'] = 0.51*df5['toxic'] + 0.49*sub['toxic_temp']

sub['toxic'] = 0.5*sub['toxic_temp'] + 0.5*sub['toxic']



sub['toxic_temp'] = 0.1*df4['toxic'] + 0.9*df3['toxic']

sub['toxic_temp'] = 0.51*df3['toxic'] + 0.49*sub['toxic_temp']

sub['toxic'] = 0.5*sub['toxic_temp'] + 0.5*sub['toxic']



sub['toxic_temp'] = 0.1*df2['toxic'] + 0.9*df1['toxic']

sub['toxic_temp'] = 0.51*df1['toxic'] + 0.49*sub['toxic_temp']

sub['toxic'] = 0.5*sub['toxic_temp'] + 0.5*sub['toxic']
plt.hist(sub.toxic,bins=100)

plt.show()
sub[['id', 'toxic']].to_csv('submission_pairwise_blend_ensemble.csv', index=False)

#sub[['id', 'toxic']].to_csv('submission.csv', index=False)

sub[['id', 'toxic']].head()
sub = sub[['id']].copy()

sub_final = sub[['id']].copy()

sub['toxic'] = concat_sub['jigsaw_mean']

sub.head()
upper = 0.95

lower = 0.05

lp = 0.8 # lower power

up = 1.5 # upper power
sub[(sub['toxic'] <= lower)] = sub[(sub['toxic'] <= lower)]**lp

#sub[(sub['toxic'] >= upper)] = sub[(sub['toxic'] >= upper)]**up
sub_final['toxic'] = sub['toxic']

sub_final.head()
plt.hist(sub_final.toxic,bins=100)

plt.show()
sub_final.to_csv('submission_minmax_power_scaling_ensemble.csv', index=False)

#sub_final.to_csv('submission.csv', index=False)

sub_final.head()