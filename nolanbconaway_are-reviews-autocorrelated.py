import sqlite3, datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.signal import savgol_filter



pd.set_option('precision', 2)

np.set_printoptions(precision=2)



con = sqlite3.connect('../input/database.sqlite')

all_reviews = pd.read_sql('SELECT * FROM reviews WHERE pub_year < 2017', con)

con.close()
# convert pub_date to datetime object, get unix timestamps

reviews = all_reviews.copy(deep = True) # slice to new object

reviews['pub_date'] = pd.to_datetime(reviews.pub_date, format = '%Y-%m-%d')

reviews['unix_time'] = reviews.pub_date.astype(np.int64, copy = True) // 10**9



# find the first best new music, get rid of everything before it

first_bnm = reviews.loc[reviews.best_new_music == True, 'unix_time'].min()

reviews = reviews.loc[reviews.unix_time >= first_bnm]



# print out date of first bnm

idx = (reviews.unix_time == first_bnm) & (reviews.best_new_music == True)

first_bnm_str = datetime.datetime.fromtimestamp(first_bnm)

print('First best new music: ' + first_bnm_str.strftime('%B %d, %Y'))



# find overall proportion of best new music

proportion_bnm = np.mean(reviews.best_new_music)

print('Global p(bnm): ' + str(proportion_bnm))
# add columns for the author's review count, time since last review

reviews['review_num'] = pd.Series(index = reviews.index)

reviews['days_elapsed'] = pd.Series(index = reviews.index)



for a, rows in reviews.copy().groupby('author'):

    rows_sort = rows.sort_values(by = 'unix_time')

    

    # add review number

    nums = np.arange(rows.shape[0], dtype = int)

    reviews.loc[rows_sort.index, 'review_num'] = nums

    

    # add days elapsed

    days_elapsed = np.zeros(nums.shape) -1

    for j in nums[1:]:

        curr = rows_sort.iloc[j]

        prev = rows_sort.iloc[j-1]

        seconds_elapsed = curr.unix_time - prev.unix_time

        days_elapsed[j] = seconds_elapsed / 86400.0

        

    days_elapsed[days_elapsed<0] = np.NAN

    reviews.loc[rows_sort.index, 'days_elapsed'] = days_elapsed            



reviews.days_elapsed.describe()
data = reviews.days_elapsed.dropna()

data = data[data<50]



sns.distplot(data, bins = 50)

plt.xlabel('Days Since Previous Review')

plt.ylabel('Density')

plt.xlim([-1,50])

plt.show()
def autocorrfun(x, col, lag = 1):

    x_sort = x.sort_values(by = 'review_num')

    return x_sort[col].autocorr(lag)



autocorrelations = pd.DataFrame(

    index = pd.unique(reviews.author), 

    columns = ['score', 'best_new_music']

)



for i, rows in reviews.groupby('author'):

    if rows.shape[0] < 5:

        autocorrelations = autocorrelations.drop(i)

        continue

    

    for j in autocorrelations.columns:

        autocorrelations.loc[i, j] = autocorrfun(rows, j)



autocorrelations = autocorrelations.astype(float)        

autocorrelations.describe()
from scipy.stats import ttest_1samp



f, axs = plt.subplots(1,2)

for i, col in enumerate(autocorrelations.columns.values):

    h = axs[i]

    data = autocorrelations[col].dropna()

    

    p = ttest_1samp(data,0).pvalue

    print(col + ': p = ' + str(p))

    sns.kdeplot(data, ax = h, shade=True,gridsize=200, legend = False)

    h.set_xlim(np.array([-1,1]))

    h.set_title(col)

    h.set_xlabel('Autocorrelation')

    h.set_ylabel('Density')

    

plt.show()
def get_next_reviews(df):

    res = pd.DataFrame(index = None, columns = reviews.columns)

    for a, rows in df.groupby('author'):

        nums = rows.review_num+1

        idx = (reviews.author == a) & reviews.review_num.isin(nums)

        res = res.append(reviews.loc[idx])

    return res





score_split = dict(low = dict(), mid = dict(), high = dict())

for i in score_split.keys():

    

    if i=='low':

        idx = reviews.score<5.5

    elif i == 'mid':

        idx = (reviews.score<=8.2) & (reviews.score>=5.5)

    elif i=='high':

        idx = reviews.score>8.2

    

    score_split[i]['N'] = reviews.loc[idx]

    score_split[i]['N+1'] = get_next_reviews(score_split[i]['N'])     
for i, k in enumerate(['high','mid','low']):

    data = score_split[k]['N+1']

    sns.kdeplot(data.score.dropna(), shade=True, label = k)

    

    m1 = score_split[k]['N+1'].score.mean()

    m0 = score_split[k]['N'].score.mean()

    

    print(k + '\t N mean= ' + str(m0) + '\t N+1 mean=' + str(m1))

    

# plt.axis([0,10,0,1.6])





plt.xlabel('Score')

plt.ylabel('Density')

plt.title('$N+1$ Distributions')

plt.show()