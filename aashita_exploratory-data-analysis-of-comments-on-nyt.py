import pandas as pd
import numpy as np

from textblob import TextBlob

import warnings 
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
plt.style.use('fivethirtyeight')
%matplotlib inline

# Next we import the dataframe containing all the comments on New York Times articles published in April 2018
curr_dir = '../input/'
comments = pd.read_csv(curr_dir + 'CommentsApril2018.csv')
articles = pd.read_csv(curr_dir + 'ArticlesApril2018.csv')

# We write the two functions that are used often:
def print_largest_values(s, n=5):
    s = sorted(s.unique())
    for v in s[-1:-(n+1):-1]:
        print(v)
    print()
    
def print_smallest_values(s, n=5):
    s = sorted(s.unique())
    for v in s[:n]:
        print(v)
    print()
    

comments.sample(5)
comments.shape
comments.info()
comments.commentBody.sample()
def preprocess(commentBody):
    commentBody = commentBody.str.replace("(<br/>)", "")
    commentBody = commentBody.str.replace('(<a).*(>).*(</a>)', '')
    commentBody = commentBody.str.replace('(&amp)', '')
    commentBody = commentBody.str.replace('(&gt)', '')
    commentBody = commentBody.str.replace('(&lt)', '')
    commentBody = commentBody.str.replace('(\xa0)', ' ')  
    return commentBody
comments.commentBody = preprocess(comments.commentBody)
comments.commentTitle.value_counts()
comments.drop(['commentTitle', 'recommendedFlag', 'reportAbuseFlag', 'userURL'], axis=1, inplace=True)
describe = comments.describe(include=['O']).transpose()
table = ff.create_table(describe, index=True, index_title='Categorical columns')
iplot(table)
comments.describe().transpose()
comments.isnull().sum()
comments['sentiment'] = comments.commentBody.map(lambda text: TextBlob(text).sentiment.polarity)
print("5 random comments with highest positive sentiment polarity: \n")
cL = comments.loc[comments.sentiment==1, ['commentBody']].sample(5).values
for c in cL:
    print(c[0])
    print()
print("5 random comments with most negative sentiment polarity: \n")
cL = comments.loc[comments.sentiment==-1, ['commentBody']].sample(5).values
for c in cL:
    print(c[0])
    print()
print("5 random comments with most neutral (zero) sentiment polarity:\n ")
cL = comments.loc[comments.sentiment==0, ['commentBody']].sample(5).values
for c in cL:
    print(c[0])
    print()
mpl.rcParams['figure.figsize'] = (18, 8)
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'x-large'
sns.distplot(comments.sentiment);
plt.title("Distribution of sentiment polarity of comments");
sns.set(rc={'axes.labelsize':25})
grid = sns.jointplot('sentiment', 'recommendations', data=comments)
grid.fig.set_figwidth(18)
grid.fig.set_figheight(18)
grid.fig.subplots_adjust(top=0.95)
grid.fig.suptitle("Joint distribution of sentiment polarity of comments vs number of upvotes on them");
mpl.rcParams['figure.figsize'] = (18, 8)
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'x-large'
fig, ax = plt.subplots()
sns.distplot(comments.loc[comments.editorsSelection==0, 'sentiment'], ax=ax);
sns.kdeplot(comments.loc[comments.editorsSelection==0, 'sentiment'], ax=ax, label = "NYT's pick");
sns.distplot(comments.loc[comments.editorsSelection==1, 'sentiment'], ax=ax);
sns.kdeplot(comments.loc[comments.editorsSelection==1, 'sentiment'], ax=ax, label = "Not NYT's pick");
plt.title("Distribution of sentiment polarity of comments based on NYT's pick");
ax.legend(loc='upper left');
mpl.rcParams['figure.figsize'] = (18, 8)
sns.barplot(x='commentID', y='recommendations', data=comments.sample(2000)); 
plt.xticks([]);
sns.distplot(comments.recommendations);
plt.title("Distribution of upvotes (recommendations) on comments");
print("Top 5 highest number of on comments: ")
print_largest_values(comments.recommendations)

comments.recommendations.quantile(0.99), comments.recommendations.quantile(0.95)
percs = np.linspace(0,100,40)
qn = np.percentile(comments.recommendations, percs)
plt.scatter(percs[:-1], qn[:-1]);
plt.title('Q-Q plot for the distribution of upvotes');
sns.distplot(comments.loc[comments.recommendations<=comments.recommendations.quantile(0.95), 'recommendations']);
plt.title("Distribution of upvotes (recommendations) on comments for the bottom 95%");
comments[comments.recommendations > 2500].shape[0]
comments = comments[comments.recommendations <= 2500]
articles.head()
articles.info()
set(articles.columns).intersection(set(comments.columns))
articles.describe(include=['O']).transpose()
articles.describe().transpose()
articles.isnull().sum()
grouped = comments.groupby('articleID')
grouped_articles = pd.concat([grouped.commentID.count(), grouped.recommendations.median(),
           grouped[['editorsSelection', 'sharing', 'timespeople', 'trusted']].mean()], 
          axis=1).reset_index().rename(columns = {'commentID': 'commentsCount'})
grouped_articles.sample(5)
articles.shape, grouped_articles.shape
articles = articles.merge(grouped_articles)
articles.sample(5)
articles.shape
articles.columns
mpl.rcParams['figure.figsize'] = (16, 8)
sns.distplot(articles.commentsCount);
plt.title("Distribution of number of comments on articles");
print("Top 5 articles with the highest number of comments have the following count of comments: ")
print_largest_values(articles.commentsCount)
sns.distplot(articles.articleWordCount);
plt.title("Distribution of word counts on articles");
print("Top 5 lengthiest articles contains the following number of words: ")
print_largest_values(articles.articleWordCount)
print("Top 5 shortest articles contains the following number of words: ")
print_smallest_values(articles.articleWordCount)
sns.distplot(articles.recommendations);
plt.title("Distribution of average number of upvotes on articles");
print("Top 5 articles in terms of the highest number of median upvotes on the comments have the following count of upvotes: ")
print_largest_values(articles.recommendations)
print("Top 5 articles in terms of the least number of median upvotes on the comments have the following count of upvotes: ")
print_smallest_values(articles.recommendations)
comments.editorsSelection.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'medium'
plt.axis('equal')
plt.pie(comments.editorsSelection.value_counts(), labels=('', "NYT's pick"));
plt.title("Selection of comments as NYT's pick");
mpl.rcParams['figure.figsize'] = (6, 6)
sns.barplot(x='editorsSelection', y='recommendations', data=comments);
plt.title("Average number of upvotes on comments that are NYT's pick vs not");
mpl.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'x-large'
sns.distplot(comments.loc[comments.editorsSelection==1, 'recommendations']);
plt.title("Distribution of number of upvotes on comments that are NYT's pick");
comments.editorsSelection.corr(comments.recommendations)
comments.commentType.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'medium'
plt.axis('equal')
plt.pie(comments.commentType.value_counts(), labels=('Comments', "Replies", "")); 
plt.title('Type of comments');
mpl.rcParams['figure.figsize'] = (8, 6)
sns.barplot(x='commentType', y='recommendations', data=comments);
plt.title("Average number of upvotes for each type of comments");
sns.barplot(x='commentType', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick for each type of comments");
comments.depth.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
plt.axis('equal')
plt.pie(comments.depth.value_counts(), labels=('1','2', '3', '', '')); 
plt.title('Depth of comments');
mpl.rcParams['figure.figsize'] = (10, 6)
sns.barplot(x='depth', y='recommendations', data=comments);
plt.title("Average number of upvotes for each depth of comments");
sns.barplot(x='depth', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick for each depth of comments");
sns.barplot(x='depth', y='recommendations', hue='commentType', data=comments);
plt.title("Average number of upvotes for each depth and respective type of comments");
sns.barplot(x='depth', y='editorsSelection', hue='commentType', data=comments);
plt.title("Proportion of comments selected as NYT's pick for each depth and respective type of comments");
sns.barplot(x='commentType', y='recommendations', hue='depth', data=comments);
plt.title("Average number of upvotes for each type and respective depth of comments");
sns.barplot(x='commentType', y='editorsSelection', hue='depth', data=comments);
plt.title("Proportion of comments selected as NYT's pick for each type and respective depth of comments");
mpl.rcParams['figure.figsize'] = (18, 15)
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'x-large'
sns.barplot(x='replyCount', y='recommendations', data=comments[comments.replyCount>0]);
plt.title("Average number of upvotes based on number of replies to comments");
sns.barplot(x='replyCount', y='editorsSelection', data=comments[comments.replyCount>0]);
plt.title("Proportion of comments selected as NYT's pick based on number of replies to comments");
sns.countplot(x="printPage", data=comments[comments.printPage>0]);
plt.title("Total number of comments on articles on each print page");
sns.distplot(articles.printPage);
plt.title("Distribution of average number of articles on each print page");
sns.barplot(x='printPage', y='recommendations', data=comments);
plt.title("Average number of upvotes for comments in each print page");
sns.barplot(x='printPage', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick for articles on each print page");
mpl.rcParams['figure.figsize'] = (16, 16)
mpl.rcParams['axes.labelsize'] = 'xx-large'
mpl.rcParams['ytick.labelsize'] = 'large'
sns.countplot(y="newDesk", data=comments, order=comments.newDesk.value_counts().index);
mpl.rcParams['figure.figsize'] = (18, 20)
sns.barplot(y='newDesk', x='recommendations', data=comments, order=comments.newDesk.value_counts().index);
plt.title("Average number of upvotes for comments in each newDesk");
sns.barplot(y='newDesk', x='editorsSelection', data=comments, order=comments.newDesk.value_counts().index);
plt.title("Proportion of comments selected as NYT's pick for each newDesk of comments");
top_desk = set(comments.newDesk.value_counts()[:4].index)
top_desk
sample_frequent_newDesk = comments.loc[comments.newDesk.isin(top_desk),
                                 ['newDesk', 'recommendations']].sample(2000)

sample_frequent_newDesk.newDesk = sample_frequent_newDesk.newDesk.astype('object')
sns.swarmplot(x='newDesk', y='recommendations', data=sample_frequent_newDesk);
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'medium'
plt.title("Number of upvotes on comments on articles from the top 4 desks");
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'large'
sns.countplot(y="sectionName", data=comments, order=comments.sectionName.value_counts().index);
plt.title("Total number of comments on articles in each section");
sns.barplot(y='sectionName', x='recommendations', data=comments, order=comments.sectionName.value_counts().index);
plt.title("Average number of upvotes for comments on articles in each section");
sns.barplot(y='sectionName', x='editorsSelection', data=comments, order=comments.sectionName.value_counts().index);
plt.title("Proportion of comments selected as NYT's pick from articles in each section");
sorted_articles = articles[['articleWordCount', 'editorsSelection', 'recommendations']].sort_values(by='articleWordCount')
mpl.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'medium'
sns.barplot(x='articleWordCount', y='recommendations', data=sorted_articles, order=sorted_articles.articleWordCount);
plt.xticks([]);
plt.title("Average number of upvotes for comments based on word count of the articles");
sns.barplot(x='articleWordCount', y='editorsSelection', data=sorted_articles, order=sorted_articles.articleWordCount);
plt.xticks([]);
plt.title("Proportion of comments selected as NYT's pick based on word count of the articles");
fig, ax = plt.subplots()
sns.distplot(comments.createDate, ax=ax);
sns.kdeplot(comments.createDate, ax=ax);
sns.kdeplot(comments.approveDate, ax=ax);
sns.kdeplot(comments.updateDate, ax=ax);
plt.title("Distribution of timeline of comments' create, approve and update date");
ax.legend(loc='upper left');
comments.sharing.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
mpl.rcParams['axes.titlesize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'medium'
plt.axis('equal')
plt.pie(comments.sharing.value_counts(), labels=('False', "True"));
plt.title("Sharing of comments");
sns.barplot(x='sharing', y='recommendations', data=comments);
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'medium'
plt.title("Average number of upvotes based on sharing of comments");
sns.barplot(x='sharing', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick based on sharing of comments");
comments.trusted.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
plt.axis('equal')
plt.pie(comments.trusted.value_counts(), labels=('False', "True"));
plt.title("Comments by trusted commenters");
sns.barplot(x='trusted', y='recommendations', data=comments);
plt.title("Average number of upvotes based on whether commenters were trusted or not");
sns.barplot(x='trusted', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick based on whether commenters were trusted or not");
comments.timespeople.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
plt.axis('equal')
plt.pie(comments.timespeople.value_counts(), labels=("True", 'False'));
plt.title("Comments by timespeople");
sns.barplot(x='timespeople', y='recommendations', data=comments);
plt.title("Average number of upvotes based on whether comments were made by timespeople");
sns.barplot(x='timespeople', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick based on whether comments were made by timespeople");
comments.picURL.value_counts()[:10]
def custom_pic_feature(df):
    url1 = 'https://graphics8.nytimes.com/images/apps/timespeople/none.png'
    url2 = 'http://graphics8.nytimes.com/images/apps/timespeople/none.png'
    df['customPic'] = np.where((df.picURL == url1) | (df.picURL == url2), 0, 1)
    df.customPic = df.customPic.astype('category').cat.codes
    return df

comments = custom_pic_feature(comments)
comments.customPic.value_counts()
mpl.rcParams['figure.figsize'] = (6, 5)
plt.axis('equal')
plt.pie(comments.customPic.value_counts(), labels=('False', "True"));
plt.title("Comments by commenters with custom pic");
sns.barplot(x='customPic', y='recommendations', data=comments);
plt.title("Average number of upvotes based on whether the commenters' use custom pic");
sns.barplot(x='customPic', y='editorsSelection', data=comments);
plt.title("Proportion of comments selected as NYT's pick based on whether commenters use custom pic");
plt.figure(figsize=(12, 10))
ax = plt.subplot(211)
ax.axis('equal')
ax.pie(comments.timespeople.value_counts(), labels=('default pic', 'custom pic'));
ax.set_title('All comments')

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(223)
ax1.axis('equal')
ax1.pie(comments.loc[comments.timespeople==0, 'customPic'].value_counts(), labels=('default pic', 'custom pic'));
ax1.set_title('Timespeople==0')

ax2 = plt.subplot(224)
ax2.axis('equal')
ax2.pie(comments.loc[comments.timespeople==1, 'customPic'].value_counts(), labels=('default pic', 'custom pic'));
ax2.set_title('Timespeople==1')

plt.show()
plt.figure(figsize=(12, 10))
ax = plt.subplot(211)
ax.axis('equal')
ax.pie(comments.timespeople.value_counts(), labels=('timespeople', 'not timespeople'));
ax.set_title('All comments')

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(223)
ax1.axis('equal')
ax1.pie(comments.loc[comments.customPic==0, 'timespeople'].value_counts(), labels=('timespeople', 'not timespeople'));
ax1.set_title('Default pic')

ax2 = plt.subplot(224)
ax2.axis('equal')
ax2.pie(comments.loc[comments.customPic==1, 'timespeople'].value_counts(), labels=('timespeople', 'not timespeople'));
ax2.set_title('Custom pic')

plt.show()
sns.barplot(x='customPic', y='recommendations', hue='timespeople', data=comments);
plt.title("Average number of upvotes based on custom pic and timespeople");
sns.barplot(x='timespeople', y='recommendations', hue='customPic', data=comments);
plt.title("Average number of upvotes based on timespeople and custom pic ");
comments.userTitle.fillna('Unknown', inplace=True)
comments.userTitle.value_counts()
mpl.rcParams['figure.figsize'] = (20, 18)
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'xx-large'
mpl.rcParams['ytick.labelsize'] = 'large'
sns.barplot(x='recommendations', y="userTitle", data=comments);
plt.title("Average number of upvotes for comments based on the title of the commenter");