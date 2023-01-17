import numpy as np

import pandas as pd

import string

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

color = sns.color_palette()

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')



print(f'Train Shape: {df_train.shape}')

print(f'Test Shape: {df_test.shape}')
df_train.head()
df_test.head()
df_train.isnull().sum()
df_train['target'].value_counts()
np.mean(df_train['target'].values)
#Bar graph for showing count of each target

count = df_train['target'].value_counts()



plot = go.Bar(x=count.index,y=count.values,marker=dict(color=count.values,colorscale = 'Picnic'))

layout = go.Layout(title = 'Target Value Count')



fig = go.Figure(data = [plot],layout=layout)

py.iplot(fig,filename='Value Count')



#Pie Chart for showing distribution



labels = count.index

value = np.array((count/count.sum())*100)



plot = go.Pie(labels=labels,values = value)

layout = go.Layout(title='Target Value Distribution')

fig = go.Figure(data=[plot],layout=layout)

py.iplot(fig,filename='Target Distribution')
#Wordcloud Representation

#Thanks for the Kernel: https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

from wordcloud import STOPWORDS,WordCloud



def wcloud(text,title=None,figure_size=(24.0,16.0)):

    stopwords = set(STOPWORDS)

    stopwords = stopwords.union({'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'})

    

    wordcloud = WordCloud(stopwords=stopwords,random_state = 42,width=800, 

                    height=400,).generate(str(text))

    

    plt.figure(figsize=figure_size)

    plt.title(title,fontdict={'size': 40,})

    plt.imshow(wordcloud)
wcloud(df_train[df_train['target']==0]['question_text'],'Sincere Questions Cloud')
wcloud(df_train[df_train['target']==1]['question_text'],'InSincere Questions Cloud')
# Thanks SRK for the great kernel: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc



from collections import defaultdict

train1_df = df_train[df_train["target"]==1]

train0_df = df_train[df_train["target"]==0]



def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



def generate_bar(df,color):

    plot = go.Bar(x=df['word_count'].values[::-1],y=df['word'].values[::-1],

                 orientation = 'h',showlegend=False,marker=dict(color=color))

    return plot
#Bar for insincere questions

freq_dict = defaultdict(int)

for que in train1_df['question_text']:

    for word in generate_ngrams(que,2):

        freq_dict[word]+=1

sorted_freq = pd.DataFrame(sorted(freq_dict.items(),key= lambda x:x[1])[::-1])

sorted_freq.columns = ['word', 'word_count']

plot1 = generate_bar(sorted_freq.head(50),'red')
#Bar for sincere questions

freq_dict = defaultdict(int)

for que in train0_df['question_text']:

    for word in generate_ngrams(que,2):

        freq_dict[word]+=1

sorted_freq = pd.DataFrame(sorted(freq_dict.items(),key= lambda x:x[1])[::-1])

sorted_freq.columns = ['word', 'word_count']

plot0 = generate_bar(sorted_freq.head(50),'blue')
# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of sincere questions", 

                                          "Frequent words of insincere questions"])

fig.append_trace(plot0, 1, 1)

fig.append_trace(plot1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
#Number of words#

df_train['word_count'] =  df_train['question_text'].apply(lambda x: len(str(x).split()))

df_test['word_count'] =  df_test['question_text'].apply(lambda x: len(str(x).split()))



#Number of unique words

df_train['unique'] =  df_train['question_text'].apply(lambda x: len(set(str(x).split())))

df_test['unique'] =  df_test['question_text'].apply(lambda x: len(set(str(x).split())))



#Number of characters

df_train['char_count'] =  df_train['question_text'].apply(lambda x: len(str(x)))

df_test['char_count'] =  df_test['question_text'].apply(lambda x: len(str(x)))



# Number of stopwords

df_train['stopwords'] =  df_train['question_text'].apply(lambda x: len([word for word in str(x).lower().split() if word in STOPWORDS]))

df_test['stopwords'] =  df_test['question_text'].apply(lambda x: len([word for word in str(x).lower().split() if word in STOPWORDS]))



#Number of Puncuations

df_train['punct'] =  df_train['question_text'].apply(lambda x: len([char for char in str(x) if char in string.punctuation]))

df_test['punct'] =  df_test['question_text'].apply(lambda x: len([char for char in str(x) if char in string.punctuation]))



#Number of UpperCase

df_train['upper'] =  df_train['question_text'].apply(lambda x: len([word for word in str(x).split() if word.isupper()]))

df_test['upper'] =  df_test['question_text'].apply(lambda x: len([word for word in str(x).split() if word.isupper()]))



#Number of Title Case

df_train['title'] =  df_train['question_text'].apply(lambda x: len([word for word in str(x).split() if word.istitle()]))

df_test['title'] =  df_test['question_text'].apply(lambda x: len([word for word in str(x).split() if word.istitle()]))



#Average length of word

#Number of characters

df_train['word_len'] =  df_train['question_text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))

df_test['word_len'] =  df_test['question_text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
df_train.head()
## Truncate some extreme values for better visuals ##

df_train['word_count'].loc[df_train['word_count']>60] = 60 #truncation for better visuals

df_train['punct'].loc[df_train['punct']>10] = 10 #truncation for better visuals

df_train['char_count'].loc[df_train['char_count']>350] = 350 #truncation for better visuals



f, axes = plt.subplots(3, 1, figsize=(10,20))

sns.boxplot(x='target', y='word_count', data=df_train, ax=axes[0])

axes[0].set_xlabel('Target', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='target', y='punct', data=df_train, ax=axes[1])

axes[1].set_xlabel('Target', fontsize=12)

axes[1].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='target', y='char_count', data=df_train, ax=axes[2])

axes[2].set_xlabel('Target', fontsize=12)

axes[2].set_title("Number of punctuations in each class", fontsize=15)

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,classification_report, log_loss,f1_score
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['word_count','unique','char_count','stopwords','punct','upper','title','word_len']

cv_scores = []

pred_val = np.zeros([df_train.shape[0]])

for train_index, val_index in kf.split(df_train):

    X_train, X_val = df_train.loc[train_index][eng_features].values,df_train.loc[val_index][eng_features].values

    y_train, y_val = df_train.loc[train_index]['target'].values,df_train.loc[val_index]['target'].values

    classifier = LogisticRegression(class_weight='balanced',n_jobs = -1, solver='lbfgs')

    classifier.fit(X_train,y_train)

    pred = classifier.predict_proba(X_val)[:,1]

    pred_val[val_index] = pred

    cv_scores.append(log_loss(y_val,pred))

    
for thresh in np.arange(0.66, 0.67, 0.001):

    print(f"f1-score for threshold:{thresh} is {f1_score(y_val,pred>thresh)}")
print(accuracy_score(y_val,pred>0.66))
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(stop_words='english',ngram_range=(1,3))



tf_idf.fit(df_train['question_text'].values.tolist() + df_test['question_text'].values.tolist())



tf_train = tf_idf.transform(df_train['question_text'].values.tolist())

tf_test = tf_idf.transform(df_test['question_text'].values.tolist())
tf_train
from sklearn.svm import LinearSVC



train_y = df_train["target"].values

cv_scores = []

pred_train = np.zeros([df_train.shape[0]])



svc = LinearSVC(random_state = 42)

kf = KFold(n_splits=5,shuffle=True,random_state=42)



for train_index, val_index in kf.split(df_train):

    X_train, X_val = tf_train[train_index], tf_train[val_index]

    y_train, y_val = train_y[train_index],train_y[val_index]

    svc.fit(X_train,y_train)

    pred = svc.predict(X_val)

    pred_train[val_index] = pred

    cv_scores.append(log_loss(y_val,pred))
cv_scores
for thresh in np.arange(0.1,0.8,0.1):

    print(f"f1-score for threshold:{thresh} is {f1_score(y_val,pred>thresh)}")

    
sub = pd.read_csv('../input/sample_submission.csv')



sub.head()
df_test['prediction'] = svc.predict(tf_test)
df_sub = df_test.drop(labels=['question_text','word_count','word_count','unique',\

                              'char_count','stopwords','punct','upper','title','word_len'],axis = 1)

df_sub.head()
df_sub.to_csv(path_or_buf='submission.csv',index=False)