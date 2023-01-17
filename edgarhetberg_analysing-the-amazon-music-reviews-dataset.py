import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

######## HELPER FUNCTIONS ##########
def look_at_categories(df,col):
    # general look
    print(df[col].describe())
    # how often do reviewers review?
    count_reviews = df[col].value_counts()

    #print(count_reviews)
    print('mean',count_reviews.mean())
    print('min',count_reviews.min())
    print('max',count_reviews.max())

    print('0.5',count_reviews.quantile(0.5))
    print('0.95',count_reviews.quantile(0.95))
    print('0.99',count_reviews.quantile(0.99))

    return count_reviews
    
def look_at_numerics(df,col):
    print(df[col].describe())
    print('mean',df[col].mean())
    print('min',df[col].min())
    print('max',df[col].max())

    print('0.5',df[col].quantile(0.5))
    print('0.95',df[col].quantile(0.95))
    print('0.99',df[col].quantile(0.99))
    
    return df[col]

    
def draw_hist(count_reviews,ylabel,xlabel='number of reviews',bins=30):

    fig  = plt.figure()
    #hist
    plt.hist(count_reviews,bins=bins)
    # vertical lines
    plt.axvline(count_reviews.mean(),color='yellow',label='mean')
    plt.axvline(count_reviews.quantile(0.95),color='orange',label='Top 5%')
    plt.axvline(count_reviews.quantile(0.99),color='red',label='Top 1%')
    # labels
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set size
    fig.set_size_inches(18.5, 10.5)
# read data
df= pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')#
# Is the shape as anticipated?
print('Is the shape as anticipated? ',df.shape==(10261,9))
# Are the columns as anticipated and what kind of index is used? 
print('Are the columns as anticipated and what kind of index is used?')
print('index: ',df.head().index)
print('columns: ', df.columns)
# Are there any NaN entries?
print('Number of nan entries? \n',df.isna().sum())
# Are there any duplicated rows?
print('Are there any duplicated rows? ',df.duplicated().sum()>0)
# can i fill up the missing names
print('can i fill up the missing names?')
no_names_id = df[df['reviewerName'].isna()]['reviewerID'].unique()
i_can='No'
for id in no_names_id:
    if df[df['reviewerID'].isna()]['reviewerName'].shape[0]>0:
        i_can='Yes'
print(i_can)
    
###
count_reviews_rvid = look_at_categories(df,'reviewerID')

draw_hist(count_reviews_rvid,'number of reviewers')
count_reviews_asin = look_at_categories(df,'asin')

# Can a reviewer review a product multiple times? 
# (is there a duplicated row when only looking at asin and reviewerID?)
print('Can a reviewer review a product multiple times? ',df[['asin','reviewerID']].duplicated().sum()>0)

draw_hist(count_reviews_asin,'number of products')

# Can a reviewer review a product multiple times? 
# (is there a duplicated row when only looking at asin and reviewerID?)
print('Can a reviewer review a product multiple times? ',df[['asin','reviewerID']].duplicated().sum()>0)

# how often do reviewers review?
count_reviews_rvn = look_at_categories(df,'reviewerName')
fig,ax = plt.subplots(2,sharex=True)


#fig  = plt.figure()
#hist
ax[0].hist(count_reviews_rvn,bins=60,label = 'number of reviewer names')
# vertical lines
ax[0].axvline(count_reviews_rvn.mean(),color='yellow',label='mean')
ax[0].axvline(count_reviews_rvn.quantile(0.95),color='orange',label='Top 5%')
ax[0].axvline(count_reviews_rvn.quantile(0.99),color='red',label='Top 1%')
# labels
ax[0].legend()


#hist
ax[1].hist(count_reviews_rvid,bins=40,label='number of reviewer IDs')
# vertical lines
ax[1].axvline(count_reviews_rvid.mean(),color='yellow',label='mean')
ax[1].axvline(count_reviews_rvid.quantile(0.95),color='orange',label='Top 5%')
ax[1].axvline(count_reviews_rvid.quantile(0.99),color='red',label='Top 1%')
# labels
ax[1].legend()

ax.flat[0].set( ylabel='number of reviewer names')
ax.flat[1].set(xlabel='number of reviews', ylabel='number of reviewer IDs')

# set size
fig.set_size_inches(16.5, 12.5)
## names with multiple IDs
for name in df['reviewerName'].unique():
    if df[df['reviewerName']==name]['reviewerID'].unique().shape[0]>1:
        print(name)


        
## multiple names with one ID
for ID in df['reviewerID'].unique():
    if df[df['reviewerID']==ID]['reviewerName'].unique().shape[0]>1:
        print(df[df['reviewerID']==ID]['reviewerName'].unique())

### helpful
# make helpful ratio
from ast import literal_eval

def ratio_fun(x):
    evaled = literal_eval(x)
    if evaled[1] == 0:
        return float('NaN')
    else:
        return evaled[0]/evaled[1]

def all_votes_fun(x):
    return literal_eval(x)[1]

df['helpful_ratio'] = df['helpful'].apply(ratio_fun)
df['helpful_all_votes'] = df['helpful'].apply(all_votes_fun)
count_reviews_hpr = look_at_numerics(df,'helpful_ratio')
count_reviews_hpav = look_at_numerics(df,'helpful_all_votes')



draw_hist(df['helpful_ratio'].dropna(),'number of reviews','helpful ratios')

draw_hist(df[df['helpful_all_votes']>0]['helpful_all_votes'],'helpful ratios', 'all votes',bins=80)
_= look_at_numerics(df[df['helpful_all_votes']>0],'helpful_all_votes')
count_reviews_ov = look_at_numerics(df,'overall')
# are all entries integers?
sum(df['overall']==df['overall'].apply(lambda x:int(x)))
# how many 5 star reviews are there?
df['overall'].value_counts()
draw_hist(df['overall'],'number of reviews','overall')

df['reviewText_len'] = df['reviewText'].apply(lambda x: len(str(x)))
_ = look_at_numerics(df,'reviewText_len')
# Minimum without "NaN"
no_na_review = df[['reviewText']].dropna()
no_na_review['reviewText_len'] = no_na_review['reviewText'].apply(lambda x: len(str(x)))
print('Minimum without "NaN": ',no_na_review['reviewText_len'].min())
# Whats the shortest review?
print('Whats the shortest review? \n',no_na_review[no_na_review['reviewText_len'] == no_na_review['reviewText_len'].min()]['reviewText'])
# Is the shortest review helpful?
print('Is the shortest review helpful? ',df[df['reviewText_len'] == no_na_review['reviewText_len'].min()]['helpful'])
# Whats the longest review?
#print('Whats the longest review? \n',no_na_review[no_na_review['reviewText_len'] == no_na_review['reviewText_len'].max()]['reviewText'].values)

draw_hist(df['reviewText_len'],'number of reviews','reviewer text length',bins=80)
df['summary_len'] = df['summary'].apply(lambda x: len(str(x)))
_ = look_at_numerics(df,'summary_len')
# Whats a summary of a NaN review?
print(' Whats a summary of a NaN review? \n',df[df['reviewText'].isna()]['summary'])
# Whats the shortest summary?
print('Whats the shortest summary? \n',df[df['summary_len']==df['summary_len'].min()]['summary'])
# Whats the longest summary?
print('Whats the longest summary? \n',df[df['summary_len']==df['summary_len'].max()]['summary'].values)
#

draw_hist(df['summary_len'],'number of reviews','length of summary',bins=60)
df['reviewTime_dt'] = pd.to_datetime(df['reviewTime'])
_ = look_at_numerics(df,'reviewTime_dt')
# Is unixTime = Time?
print('Is unixTime = Time? ',sum(pd.to_datetime(df['reviewTime']) == pd.to_datetime(df['unixReviewTime'],unit='s'))>0)
# Where is 95% of the data?
print('Where is 95% of the data? past ', df['reviewTime_dt'].quantile(0.05))

draw_hist(df['reviewTime_dt'],'number of reviews','timeline')
sns.pairplot(df)