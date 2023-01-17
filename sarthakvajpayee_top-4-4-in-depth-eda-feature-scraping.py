# importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import os
from urllib.parse import urlparse
%matplotlib inline
from IPython.display import Image 
sns.set()
os.listdir('../input/google-quest-challenge')
# reading the data into dataframe using pandas
train = pd.read_csv('../input/google-quest-challenge/train.csv')
test = pd.read_csv('../input/google-quest-challenge/test.csv')
submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
# Let's check the top 5 entries of train data.
train.head()
# Let's check the statistical description of the numerical features in train data
train.describe()
train.iloc[:, 11:].columns
# let's check the unique values in target features
np.unique(train.iloc[:, 11:].values)
# These are the features provided in the test data
test.columns
# these are the features that we need to include while submitting the results
submission.columns
Image('../input/google-quest-qna-eda-img/url.png', width=920, height=480)
# A text feature that represents the title of the question.
train['question_title'].head()
# Let's calculate the length of each question title
length = train['question_title'].apply(lambda x:len(x.split(' ')))
length.describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=length)], 
                layout = go.Layout(title='histogram of length of question title in train data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
test['question_title'].head()
# Let's calculate the length of each question title
length = test['question_title'].apply(lambda x:len(x.split(' ')))
length.describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=length)], 
                layout = go.Layout(title='histogram of length of question title in test data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# this is another and the main text feature that represents the full description of the question asked
train['question_body'].head()
# Lets check the length of the questions body
length = train['question_body'].apply(lambda x:len(x.split(' ')))
length.describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=length, marker_color='#39f79b')], 
                layout = go.Layout(title='histogram of length of question body in train data',
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=np.log1p(length), marker_color='#39f79b')], 
                layout = go.Layout(title='histogram of log of length of question body in train data', 
                                  xaxis=dict(title='log of length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
test['question_body'].head()
# length of question in test data
length = test['question_body'].apply(lambda x:len(x.split(' ')))
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=length, marker_color='#39f79b')], 
                layout = go.Layout(title='histogram of length of question body in test data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=np.log1p(length), marker_color='#39f79b')], 
                layout = go.Layout(title='histogram of log of length of question body in test data', 
                                  xaxis=dict(title='log of length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
train['question_user_name'].head()
train['answer_user_name'].head()
# Another important text type feature that represents the answers that given to the questions.
train['answer'].head()
# Length of answers
length = train['answer'].apply(lambda x:len(x.split(' ')))
length.describe()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=length, marker_color='#eb4034')], 
                layout = go.Layout(title='histogram of length of answer in train data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=np.log1p(length), marker_color='#eb4034')], 
                layout = go.Layout(title='histogram of log of length of answer in train data', 
                                  xaxis=dict(title='log of length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
test['answer'].head()
length = test['answer'].apply(lambda x:len(x.split(' ')))
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=length, marker_color='#eb4034')], 
                layout = go.Layout(title='histogram of length of answer in test data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=np.log1p(length), marker_color='#eb4034')], 
                layout = go.Layout(title='histogram of length of answer in test data', 
                                  xaxis=dict(title='length of sentences'), 
                                  yaxis=dict(title='frequency')))
plt.show()
# This feature represents the category that the question answer pair belong to.
train['category'].head(10)
# There are 5 categories
train['category'].value_counts()
categories = train['category'].value_counts()
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(title_text="'category' Pie chart for train data",
                  annotations=[dict(text='category', x=0.5, y=0.5, 
                                    font_size=20, showarrow=False)])
fig.show()
# This feature represents the category that the question answer pair belong to.
test['category'].head(10)
# There are 5 categories
test['category'].value_counts()
categories = test['category'].value_counts()
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(title_text="'category' Pie chart for test data",
                  annotations=[dict(text='category', x=0.5, y=0.5, 
                                    font_size=20, showarrow=False)])
fig.show()
# this feature represents the host/domain name of the question answer page url.
train['host'].head(10)
# We can see that there are 63 type of these host names
train.host.value_counts()
categories = train['host'].value_counts()
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(title_text="'host' Pie chart for train data",
                  annotations=[dict(text='host', x=0.5, y=0.5, 
                                    font_size=20, showarrow=False)])
fig.show()
# this feature represents the host/domain name of the question answer page url.
test['host'].head(10)
# We can see that there are 63 type of these host names
test.host.value_counts()
categories = test['host'].value_counts()
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(title_text="'host' Pie chart for test data",
                  annotations=[dict(text='host', x=0.5, y=0.5, 
                                    font_size=20, showarrow=False)])
fig.show()

Image('../input/google-quest-qna-eda-img/posts.png', width=920, height=480)
Image('../input/google-quest-qna-eda-img/upvotes_comments.png', width=920, height=480)
train['url'].head(10)
# function for scraping the answers and their topmost comment. 
# Since all of the urls are of stackoverflow, they have the same html hierarchy.
def get_answers_comments(url): 
  try:
    get = request.urlopen(url).read() # read the html data from the url page
    src = BeautifulSoup(get, 'html.parser') # convert the data into a beautifulsoup object
    upvotes, answer = [], [] 
    correct_ans, comments = [], []
    new_features = []
    post_layout = src.find_all("div", class_ = 'post-layout') # Collecting all the posts from the page
    l = len(post_layout) # number of answers present
    for p in post_layout[:l]: # collecting answer, upvotes, comments from posts
      answer.append(p.find_all('div', class_='post-text')[0].text.strip())
      upvotes.append(int(p.find_all("div", class_ = 'js-vote-count grid--cell fc-black-500 fs-title g rid fd-column ai-center')[0].get('data-value')))
      correct_ans.append(len(p.find_all("div", class_ = 'js-accepted-answer-indicator grid--cell fc-g reen-500 ta-center py4')))
      comments.append('\n'.join([i.text.strip() for i in p.find_all('span', class_='comment-copy')]))
    idx = np.argmax(correct_ans) # index of the correct answer among all the posts
    new_features.append(upvotes.pop(idx)) # correct answer's upvotes
    new_features.append(comments.pop(idx)) # correct answer's comments
    del answer[idx]
    # collecting the answer and top comment from the top 3 posts apart from the one already provided in train.csv
    if l < 3: k=l
    else: k=3
    for a,b in zip(answer[:k], comments[:k]): 
      new_features.append(a) 
      new_features.append(b)
    for a,b in zip(answer[:3-k], comments[:3-k]): 
      new_features.append('') 
      new_features.append('')

    return new_features
    
  except:
    return [np.nan]*8 # return np.nan if the code runs into some error like page not found
Image('../input/google-quest-qna-eda-img/user.png', width=920, height=480)
train['question_user_page'].head()
train['answer_user_page'].head()
# code for scraping the data. Since all of the urls are of stackoverflow, they have the same html hierarchy.
def get_user_rating(url):
  try:
    get = request.urlopen(url).read()
    src = BeautifulSoup(get, 'html.parser')
    reputation, gold = [], []
    silver, bronze = [], []
    template = src.find_all("div", class_ = 'grid--cell fl-shrink0 ws2 overflow-hidden')[0] 
    reputation = int(''.join(template.find_all('div', class_='grid--cell fs-title fc-dark')[0].text.strip().split(',')))
    gold = int(''.join(template.find_all('div', class_='grid ai-center s-badge s-badge__gold')[0].text.strip().split(',')))
    silver = int(''.join(template.find_all('div', class_='grid ai-center s-badge s-badge__silver')[0].text.strip().split(',')))
    bronze = int(''.join(template.find_all('div', class_='grid ai-center s-badge s-badge__bronze')[0].text.strip().split(',')))
    output = [reputation, gold, silver, bronze] 
  except:
    output = [np.nan]*4 # return np.nan if the code runs into some error like page not found return output

  return output
a = [[1,2],[3,4]]
b = [[4,5,6],[7,8,9]]
np.hstack((a,b))
from tqdm.notebook import tqdm
def scrape_data(df):
    answers_comments = []
    for url in tqdm(df['url']):
      answers_comments.append(get_answers_comments(url))
    question_user_rating = []
    for url in tqdm(df['question_user_page']):
      question_user_rating.append(get_user_rating(url))
    answer_user_rating = []
    for url in tqdm(df['answer_user_page']):
      answer_user_rating.append(get_user_rating(url))
    
    return np.hstack((answerd_comments, user_rating, answer_user_rating))

# # Saving as dataframe
# columns = ['upvotes', 'comments_0', 'answer_1', 'comment_1', 'answer_2','comment_2',
#             'answer_3', 'comment_3', 'reputation_q', 'gold_q','silver_q', 'bronze_q', 
#             'reputation_a', 'gold_a', 'silver_a','bronze_a']
# scraped_train = pd.DataFrame(scrape_data(train), columns=columns)
# scraped.to_csv(f'scraped_train.csv', index=False)
# scraped_test = pd.DataFrame(scrape_data(train), columns=columns)
# scraped.to_csv(f'scraped_test.csv', index=False)
# Since I've already scraped the data once, I'll use that for the further analysis
scraped_train = pd.read_csv('../input/google-quest-qna-scraped-data/scraped_features_train.csv')
scraped_test = pd.read_csv('../input/google-quest-qna-scraped-data/scraped_features_test.csv')
scraped_train.head()
upvotes = scraped_train['upvotes'].replace(' ', np.nan).dropna().apply(lambda x:int(x.split('.')[0]))
# histogram of upvotes
plt = go.Figure(data=[go.Histogram(x=upvotes, marker_color='#00a0a0')], 
                layout = go.Layout(title='histogram upvotes for train data', 
                                  xaxis=dict(title='upvotes count'), 
                                  yaxis=dict(title='frequency')))
plt.show()
upvotes = scraped_test['upvotes'].replace(' ', np.nan).dropna().apply(lambda x:int(x.split('.')[0]))
# histogram of length of question titles
plt = go.Figure(data=[go.Histogram(x=upvotes, marker_color='#00a0a0')], 
                layout = go.Layout(title='histogram upvotes for test data', 
                                  xaxis=dict(title='upvotes count'), 
                                  yaxis=dict(title='frequency')))
plt.show()
length_c0 = scraped_train['comments_0'].apply(lambda x:len(x.split(' ')))
length_c1 = scraped_train['comment_1'].apply(lambda x:len(x.split(' ')))
length_c2 = scraped_train['comment_2'].apply(lambda x:len(x.split(' ')))
length_c3 = scraped_train['comment_3'].apply(lambda x:len(x.split(' ')))
# histogram of length of comments
plt = go.Figure(data=[go.Histogram(x=np.log1p(length_c0), marker_color='#941759', name='comment_0'),
                      go.Histogram(x=np.log1p(length_c1), marker_color='#386082', name='comment_1'),
                      go.Histogram(x=np.log1p(length_c2), marker_color='#789501', name='comment_2'),
                      go.Histogram(x=np.log1p(length_c3), marker_color='#e80995', name='comment_3')], 
                layout = go.Layout(title='histogram of log of length of comments for train data', 
                                  xaxis=dict(title='comment length'), 
                                  yaxis=dict(title='frequency')))
plt.show()
length_c0 = scraped_test['comments_0'].apply(lambda x:len(x.split(' ')))
length_c1 = scraped_test['comment_1'].apply(lambda x:len(x.split(' ')))
length_c2 = scraped_test['comment_2'].apply(lambda x:len(x.split(' ')))
length_c3 = scraped_test['comment_3'].apply(lambda x:len(x.split(' ')))
# histogram of length of comments
plt = go.Figure(data=[go.Histogram(x=np.log1p(length_c0), marker_color='#941759', name='comment_0'),
                      go.Histogram(x=np.log1p(length_c1), marker_color='#386082', name='comment_1'),
                      go.Histogram(x=np.log1p(length_c2), marker_color='#789501', name='comment_2'),
                      go.Histogram(x=np.log1p(length_c3), marker_color='#e80995', name='comment_3')], 
                layout = go.Layout(title='histogram of log of length of comments for test data', 
                                  xaxis=dict(title='comment length'), 
                                  yaxis=dict(title='frequency')))
plt.show()
length_a1 = scraped_train['answer_1'].apply(lambda x:len(x.split(' ')))
length_a2 = scraped_train['answer_2'].apply(lambda x:len(x.split(' ')))
length_a3 = scraped_train['answer_3'].apply(lambda x:len(x.split(' ')))
# histogram of length of answers
plt = go.Figure(data=[go.Histogram(x=np.log1p(length_a1), marker_color='#386082', name='answer_1'),
                      go.Histogram(x=np.log1p(length_a2), marker_color='#789501', name='answer_2'),
                      go.Histogram(x=np.log1p(length_a3), marker_color='#e80995', name='answer_3')], 
                layout = go.Layout(title='histogram of log of length of answers for train data', 
                                  xaxis=dict(title='answer_length'), 
                                  yaxis=dict(title='frequency')))
plt.show()
length_a1 = scraped_test['answer_1'].apply(lambda x:len(x.split(' ')))
length_a2 = scraped_test['answer_2'].apply(lambda x:len(x.split(' ')))
length_a3 = scraped_test['answer_3'].apply(lambda x:len(x.split(' ')))
# histogram of length of answers
plt = go.Figure(data=[go.Histogram(x=np.log1p(length_a1), marker_color='#386082', name='answer_1'),
                      go.Histogram(x=np.log1p(length_a2), marker_color='#789501', name='answer_2'),
                      go.Histogram(x=np.log1p(length_a3), marker_color='#e80995', name='answer_3')], 
                layout = go.Layout(title='histogram of log of length of answers for test data', 
                                  xaxis=dict(title='answer_length'), 
                                  yaxis=dict(title='frequency')))
plt.show()
scraped_train.columns[-8:]
# For train data
scraped_train.iloc[:, -8:].describe()
# For test data
scraped_test.iloc[:, -8:].describe()

import matplotlib.pyplot as plt
# histograms of the target labels
f,ax = plt.subplots(5,6, figsize=(24,20))
for i,label in enumerate(train.columns[11:]):
  plt.subplot(5,6,i+1)
  plt.hist(train[label], bins=20)
  plt.title(label)

plt.show()
plt.figure(figsize=(16,14))
Var_Corr = train.iloc[11:].corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns) 
plt.title('Correlation between target features.')
plt.show()
