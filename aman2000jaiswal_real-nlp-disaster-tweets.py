import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import re

import emoji



from IPython.display import Markdown as md

plt.style.use('ggplot')
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
train_path = "../input/nlp-getting-started/train.csv"

test_path = "../input/nlp-getting-started/test.csv"

sample_submission_path = "../input/nlp-getting-started/sample_submission.csv"
df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)

submission = pd.read_csv(sample_submission_path)
df_train.head()
df_train.info()
print(df_train.info())
df_test.info()
df_train = df_train[['text','target']]

df_test = df_test[['text']]
y = np.array(df_train.target.value_counts())

sns.barplot(x = [0,1],y = y,palette='gnuplot2_r')

difference = y[0]-y[1]

print("Difference between target 0 and 1: ",y[0]-y[1])
df_train.text.describe()
df_train['Char_length'] = df_train['text'].apply(len)
df_train[df_train['target']==0].Char_length.head()
f, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

f.suptitle("Histogram of char length of text",fontsize=20)

sns.distplot(df_train[df_train['target']==0].Char_length.values,kde=False,bins=20,hist=True,ax=axes[0],label="Histogram of 20 bins of label 0",

            kde_kws={"color": "r", "lw": 2, "label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

axes[0].legend(loc="best")

axes[0].set_ylabel("Rows Count")

sns.distplot(df_train[df_train['target']==1].Char_length.values,kde=False,bins=20,hist=True,ax=axes[1],label="Histogram of 20 bins of label 1",

            kde_kws={"color": "g", "lw": 2, "label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

axes[1].legend(loc="best")



plt.figure(figsize=(14,4))

sns.distplot(df_train[df_train['target']==0].Char_length.values,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 0",

            kde_kws={"color": "r", "lw": 2,"label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})



sns.distplot(df_train[df_train['target']==1].Char_length.values,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 1",

            kde_kws={"color": "g", "lw": 2,"label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

plt.ylabel("density")

plt.legend(loc="best")
def word_count(sent):

    return len(sent.split())

df_train['word_count'] = df_train.text.apply(word_count)
df_train.head()
f, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

f.suptitle("Histogram of word count",fontsize=20)

sns.distplot(df_train[df_train['target']==0].word_count.values,kde=False,bins=20,hist=True,ax=axes[0],label="Histogram of label 0",

            kde_kws={"color": "r", "lw": 2, "label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

axes[0].legend(loc="best")

axes[0].set_ylabel("Rows Count")

sns.distplot(df_train[df_train['target']==1].word_count.values,kde=False,bins=20,hist=True,ax=axes[1],label="Histogram of label 1",

            kde_kws={"color": "g", "lw": 2, "label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

axes[1].legend(loc="best")



plt.figure(figsize=(14,4))

sns.distplot(df_train[df_train['target']==0].word_count,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 0",

            kde_kws={"color": "r", "lw": 2,"label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})



sns.distplot(df_train[df_train['target']==1].word_count,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 1",

            kde_kws={"color": "g", "lw": 2,"label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

plt.ylabel("Density")

plt.legend(loc="best")
def urls(sent):

    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',sent)

def url_counts(sent):

    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',sent))

def remove_urls(sent):

    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',sent)
s ='Working on Nlp. So much fun  - https://www.helloworld.com, https://www.worldhello.com'

print(urls(s))

print(url_counts(s))

print(remove_urls(remove_urls(s)))
%%time



df_train['url_count'] = df_train.text.apply(url_counts)

df_train['urls'] = df_train.text.apply(urls)
# An overview of dataframe after above transformations

df_train.head()
print("Total Urls : ",sum(df_train.url_count))
f, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

f.suptitle("Histogram of url_counts",fontsize=20)

sns.distplot(df_train[df_train['target']==0].url_count,kde=False,bins=10,hist=True,ax=axes[0],label="Histogram of label 0",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

axes[0].legend(loc="best")

axes[0].set_ylabel("Rows Count")

sns.distplot(df_train[df_train['target']==1].url_count,kde=False,bins=10,hist=True,ax=axes[1],label="Histogram of label 1",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

axes[1].legend(loc="best")



plt.figure(figsize=(14,4))

sns.distplot(df_train[df_train['target']==0].url_count,kde=False,bins=10,hist=True,label="Histogram of 10 bins of label 0",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})



sns.distplot(df_train[df_train['target']==1].url_count,kde=False,bins=10,hist=True,label="Histogram of 10 bins of label 1",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 0.8, "color": "pink"})

plt.ylabel("Rows count")

plt.legend(loc="best")
# Actual Url Count  and differnece

print("Actual Url Count in 0 : ",df_train[df_train['target']==0].url_count.sum())

print("Actual Url Count in 1 : ",df_train[df_train['target']==1].url_count.sum())

print("Actual Url Count  differnece : "

      ,df_train[df_train['target']==1].url_count.sum() - df_train[df_train['target']==0].url_count.sum())
# Unique Urls



total_uniques = np.unique(np.ravel(df_train.urls.values)).shape[0]

uniques_in_1 = np.unique(np.ravel(df_train[df_train['target']==1].urls.values)).shape[0]

uniques_in_0 = np.unique(np.ravel(df_train[df_train['target']==0].urls.values)).shape[0]

print("Total uniques url : ", total_uniques)

print("Uniques in 0 : ",uniques_in_0)

print("Uniques in 1 : ",uniques_in_1)
df_train['text'] = df_train.text.apply(remove_urls)
# Only for emojis not work for special content

def emoji_extraction(s):

    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)

def emoji_count(s):

    return len(''.join(c for c in s if c in emoji.UNICODE_EMOJI))
# Example



s = "Working on Nlp 🙂. So much 😀 fun 😀 "

print("emoji_text                      : ", emoji_extraction(s))

print("Count of emojis                 : ",emoji_count(s))
# Work for both emojis and special content.



def emoji_extraction(sent):

    e_sent = emoji.demojize(sent)

    

    return re.findall(':(.*?):',e_sent)

def emoji_count(sent):

    e_sent = emoji.demojize(sent)

    return len(re.findall(':(.*?):',e_sent))



def emoji_to_text(sent):

    e_sent = emoji.demojize(sent)

    emo = re.findall(':(.*?):',e_sent)

    for e in emo:

        e_sent = e_sent.replace(':{}:'.format(e),'{}'.format(e))

    return e_sent
# Example



s = "Working on Nlp 🙂. So much 😀 fun 😀 "

print("emoji_text                      : ", emoji_extraction(s))

print("Count of emojis                 : ",emoji_count(s))



print("Placing text in place of emojis : ",emoji_to_text(s))
%%time

df_train['emoji_count'] = df_train.text.apply(emoji_count)

df_train['emojis'] = df_train.text.apply(emoji_extraction)
df_train[85:90]
emoji_count0  = df_train[df_train.target==0].emoji_count.value_counts()

emoji_count0 = emoji_count0.sort_index()



emoji_count1  = df_train[df_train.target==1].emoji_count.value_counts()

emoji_count1 = emoji_count1.sort_index()
f,axes = plt.subplots(1,2,figsize=(14, 4))

f.suptitle("# of Emojis",fontsize=20)

sns.barplot(emoji_count0.index,emoji_count0.values,ax = axes[0], label = "# of emojis in 0")

axes[0].set(ylim=(0, 4500))

plt.legend()

axes[0].legend(loc="best")



sns.barplot(emoji_count1.index,emoji_count1.values,ax = axes[1], label = "# of emojis in 1")

axes[1].set(ylim=(0, 4500))

axes[1].legend(loc="best")
print("Actual emoji Count in 0 : ",df_train[df_train['target']==0].emoji_count.sum())

print("Actual emoji Count in 1 : ",df_train[df_train['target']==1].emoji_count.sum())

print("Actual Url Count  differnece : "

      ,df_train[df_train['target']==1].emoji_count.sum() - df_train[df_train['target']==0].emoji_count.sum())
# Converting list of Emojis_text to Single full_text



def concatlists(lists):

    full_text =""

    for l in lists:

        full_text = full_text + " "+l[0]

    return full_text    





lists0 = df_train[np.logical_and(df_train.emoji_count>0,df_train.target==0)].emojis.values



emoji_text_for_target0 = concatlists(lists0)





lists1 = df_train[np.logical_and(df_train.emoji_count>0,df_train.target==1)].emojis.values



emoji_text_for_target1 = concatlists(lists1)
plt.figure(figsize = (18,12))

cloud = WordCloud(background_color='black',max_font_size =80).generate(emoji_text_for_target0)

plt.imshow(cloud)

plt.axis('off')

plt.title("EMOJIS TEXT FOR TARGET 0",fontsize=35)

plt.show()
plt.figure(figsize = (18,12))

cloud = WordCloud(background_color='black',max_font_size =50).generate(emoji_text_for_target1)

plt.imshow(cloud)

plt.axis('off')

plt.title("EMOJIS TEXT FOR TARGET 1",fontsize=35)

plt.show()
%%time

df_train['text'] = df_train['text'].apply(emoji_to_text)
def get_text(df):

    astext = '. '.join(list(df_train.text.values))

    text_file = open("tweets.txt", "w")

    text_file.write(astext)

    text_file.close()

    

get_text(df_train)
def find_hashtags(text):

    gethashtags = re.findall('#\w*[a-zA-Z]\w*',text)

    return gethashtags



def count_hashtags(text):

    gethashtags = re.findall('#\w*[a-zA-Z]\w*',text)

    return len(gethashtags)



def remove_hashtags(text):

    return re.sub('#\w*[a-zA-Z]\w*','',text)
# Example



s = "Working on Nlp #Nlp. So much fun #getfun awesome #tag89tag  #99999" 

print("Hashtags    : ",find_hashtags(s))

print("HashCount   : ",count_hashtags(s))

print("withouthash : ",remove_hashtags(s))
%%time

df_train['hash_count'] = df_train.text.apply(count_hashtags)

df_train['hashtags'] = df_train.text.apply(find_hashtags)
df_train.head()
hash_count0  = df_train[df_train.target==0].hash_count.value_counts()

hash_count0 = hash_count0.sort_index()



hash_count1  = df_train[df_train.target==1].hash_count.value_counts()

hash_count1 = hash_count1.sort_index()
# Dropping Count 0 as it will unbalance our plot



hash_count0 = hash_count0.drop(0)

hash_count1 = hash_count1.drop(0)
f,axes = plt.subplots(1,2,figsize=(14, 4))

f.suptitle("# of HashTags",fontsize=20)

sns.barplot(hash_count0.index,hash_count0.values,ax = axes[0], label = "# of HashTags in 0")

axes[0].set(ylim=(0, 3500))

plt.legend()

axes[0].legend(loc="best")



sns.barplot(hash_count1.index,hash_count1.values,ax = axes[1], label = "# of HashTags in 1")

axes[1].set(ylim=(0, 3500))

axes[1].legend(loc="best")
# Converting list of Hashtags to Single full_text



lists0 = df_train[np.logical_and(df_train.hash_count>0,df_train.target==0)].hashtags.values



hash_for_target0 = concatlists(lists0)





lists1 = df_train[np.logical_and(df_train.hash_count>0,df_train.target==1)].hashtags.values



hash_for_target1 = concatlists(lists1)
plt.figure(figsize = (18,12))

cloud = WordCloud(background_color='black',max_font_size =80).generate(hash_for_target0)

plt.imshow(cloud)

plt.axis('off')

plt.title("HASHTAGS FOR TARGET 0",fontsize=35)

plt.show()
plt.figure(figsize = (18,12))

cloud = WordCloud(background_color='black',max_font_size =50).generate(hash_for_target1)

plt.imshow(cloud)

plt.axis('off')

plt.title("HASHTAGS FOR TARGET 1",fontsize=35)

plt.show()
df_train['text'] = df_train.text.apply(remove_hashtags)
def extract_username(sent):

    usernames = re.findall('@[A-Za-z0-9_$]*',sent)

    return usernames



def count_username(sent):

    return len(re.findall('@[A-Za-z0-9_$]*',sent))



def replace_username(sent):

    usernames = extract_username(sent)

    for un in usernames:

        un = re.sub('@','',un)

        sent = sent.replace('@{}'.format(un),'{}'.format(un))

    return sent
# Example



s = "hello this is @Aman. Wanna talk to @some_one99 urgently."



print("usernames       : ", extract_username(s))

print("Count username  : ", count_username(s))

print("replace text    : ", replace_username(s))
%%time

df_train['text'] = df_train.text.apply(replace_username)
def find_number(text):

    getnumber = re.findall('#[0-9]+',text)

    return getnumber



def count_number(text):

    getnumber = re.findall('#[0-9]+',text)

    return len(getnumber)



def remove_number(text):

    return re.sub('#[0-9]+','',text)
# Example



s = "Working on Nlp #Nlp. So much fun #getfun awesome #tag89tag  #99999" 

print("Number        : ",find_number(s))

print("Numbercount   : ",count_number(s))

print("withoutNumber : ",remove_number(s))
%%time

df_train['count_number'] = df_train.text.apply(count_number)

df_train['number'] = df_train.text.apply(find_number)
print("Total number found : ",df_train.count_number.sum())
df_train['text'] = df_train.text.apply(remove_number)
def find_punctuations(text):

    getpunctuation = re.findall('[.?"\'`\,\-\!:;\(\)\[\]\\/“”]+?',text)

    return getpunctuation



def count_punctuations(text):

    getpunctuation = re.findall('[.?"\'`\,\-\!:;\(\)\[\]\\/“”]+?',text)

    return len(getpunctuation)



def remove_punctuations(text):

    return re.sub('[.?"\'`\,\-\!:;\(\)\[\]\\/“”]+?','',text)
s = 'Aman : “It is a historic moment ,” What about! ... your thoughts? 100/100' 

print("Punctuation        : ",find_punctuations(s))

print("Punctuationcount   : ",count_punctuations(s))

print("withoutPunctuation : ",remove_punctuations(s))
%%time

df_train['count_punctuation'] = df_train.text.apply(count_punctuations)

df_train['punctuation'] = df_train.text.apply(find_punctuations)
punct_count = df_train.count_punctuation.value_counts()

punct_count = punct_count.sort_index()
plt.figure(figsize=(18,6))

plt.title("# of punctuations",fontsize=20)

sns.barplot(punct_count.index,punct_count.values)

plt.xlabel("# of punctuation per row",fontsize=20)

for row,col in zip([i for i in range(len(punct_count))],punct_count.values):

    plt.text(row,col,col,ha = 'center')

plt.ylabel("# of rows ",fontsize=20)
def remove_symbols(text):

    return re.sub('[~:*ÛÓ_å¨È$#&%^ª|+-]+?','',text)
s = 'abcd Û sdÓ_daåfs%^ª|+-fgdas' 



print("withoutsymbols : ",remove_symbols(s))
df_train['text'] = df_train.text.apply(remove_punctuations)
df_train['text'] = df_train.text.apply(remove_symbols)
get_text(df_train)
!pip install pyspellchecker

!pip install textblob
from spellchecker import SpellChecker

from textblob import TextBlob



def find_typo(sent):

    spell = SpellChecker()

    words = sent.split()

    words = spell.unknown(words)

    return words



def count_typo(sent):

    return len(find_typo(sent))



def correct_typo(sent):

    spell = SpellChecker()

    words = sent.split()

    words = spell.unknown(words)

    find = []

    for word in words:

        find.append(spell.correction(word))

    return find    



def correct_byspellchecker(sent):

    ic, c = list(find_typo(sent)), list(correct_typo(sent))

    for i in range(len(ic)):

        sent = sent.replace(ic[i],c[i])

    return sent



def correct_bytextblob(sent):

    return str(TextBlob(sent).correct())
# Example



s = "Good to workk with naturl laguage procesing Can yoou tell me more about your work how well you doneee it"

print("orginal_text            :  ",s)

print("typos                   :  ",find_typo(s))

print("count typos             :  ",count_typo(s))

print("correct typos           :  ",correct_typo(s))

print("correct_by spellchecker :  ",correct_byspellchecker(s))

print("correct_bytextblob      :  ",correct_bytextblob(s))
# %%time

# df_train['count_typo'] = df_train.text.apply(count_typo)

# df_train['typo'] = df_train.text.apply(find_typo)

# try:

#     df_train['correct_typo'] = df_train.text.apply(correct_typo)

# except:

#     pass



# df_train.to_csv('getdata.csv',index=False)
df_train = pd.read_csv("../input/real-nlp-disaster-tweets-processed-dataframe/getdata.csv")
plt.figure(figsize = (14, 4))

plt.title("Histogram of typo_count",fontsize=20)

sns.distplot(df_train.count_typo,kde=False,bins=30,hist=True,

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

plt.ylabel("Rows Count")
from sklearn.model_selection import train_test_split



train,valid = train_test_split(df_train,test_size = 0.2,random_state=0,stratify = df_train.target.values)



print("train shape : ", train.shape)

print("valid shape : ", valid.shape)
from nltk.corpus import stopwords

stop = list(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)



X_train = vectorizer.fit_transform(train.text.values)

X_valid = vectorizer.transform(valid.text.values)



y_train = train.target.values

y_valid = valid.target.values



print("X_train.shape : ", X_train.shape)

print("X_train.shape : ", X_valid.shape)

print("y_train.shape : ", y_train.shape)

print("y_valid.shape : ", y_valid.shape)
from sklearn.naive_bayes import MultinomialNB



baseline_clf = MultinomialNB()

baseline_clf.fit(X_train,y_train)
baseline_prediction = baseline_clf.predict(X_valid)

baseline_accuracy = accuracy_score(y_valid,baseline_prediction)

print("training accuracy Score    : ",baseline_clf.score(X_train,y_train))

print("Validdation accuracy Score : ",baseline_accuracy )
plt.figure(figsize = (4,4))

class_label = [0,1]

fig = sns.heatmap(confusion_matrix(y_valid,baseline_prediction),cmap= "coolwarm",annot=True,vmin=0,cbar = False,

            center = True,xticklabels=class_label,yticklabels=class_label, fmt='d' )

fig.set_xlabel("Prediction",fontsize=30)

fig.xaxis.set_label_position('top')

fig.set_ylabel("True",fontsize=30)

fig.xaxis.tick_top()
md("### Our Baseline model validation accuracy is {}% (overfitting)".format(round(baseline_accuracy,2)))
from sklearn.linear_model import SGDClassifier

linear_model_sgd = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

linear_model_sgd.fit(X_train,y_train)
linear_model_sgd_prediction = linear_model_sgd.predict(X_valid)

linear_model_sgd_accuracy = accuracy_score(y_valid,linear_model_sgd_prediction)

print("training accuracy Score    : ",linear_model_sgd.score(X_train,y_train))

print("Validdation accuracy Score : ",linear_model_sgd_accuracy )
md("### Our sgd model validation accuracy is {}% (heavily overfitting)".format(round(linear_model_sgd_accuracy,2)))
plt.figure(figsize = (4,4))

class_label = [0,1]

fig = sns.heatmap(confusion_matrix(y_valid,linear_model_sgd_prediction),cmap= "coolwarm",annot=True,vmin=0,cbar = False,

            center = True,xticklabels=class_label,yticklabels=class_label, fmt='d' )

fig.set_xlabel("Prediction",fontsize=30)

fig.xaxis.set_label_position('top')

fig.set_ylabel("True",fontsize=30)

fig.xaxis.tick_top()
from sklearn.model_selection import GridSearchCV



params = {

     'max_iter': (100,500,1000),

     'alpha': (1e-1,1e-2,1e-4),

      'learning_rate': ('optimal','invscaling'),

    'eta0' : (0.1,0.05),

                     

}



gridcv = GridSearchCV(linear_model_sgd,param_grid = params, cv = 5)



gridcv.fit(X_train,y_train)

print("best parameter : ")

for param_name in sorted(params.keys()):

    print("      %s: %r" % (param_name, gridcv.best_params_[param_name]))
linear_model_sgd_prediction = gridcv.predict(X_valid)

linear_model_sgd_accuracy = accuracy_score(y_valid,linear_model_sgd_prediction)

print("Tune training accuracy Score    : ",gridcv.score(X_train,y_train))

print("Tune Validation accuracy Score : ",linear_model_sgd_accuracy )
plt.figure(figsize = (4,4))

class_label = [0,1]

fig = sns.heatmap(confusion_matrix(y_valid,linear_model_sgd_prediction),cmap= "coolwarm",annot=True,vmin=0,cbar = False,

            center = True,xticklabels=class_label,yticklabels=class_label, fmt='d' )

fig.set_xlabel("Prediction",fontsize=30)

fig.xaxis.set_label_position('top')

fig.set_ylabel("True",fontsize=30)

fig.xaxis.tick_top()
print(classification_report(y_valid,linear_model_sgd_prediction))
md("### Our tune sgd model validation accuracy is {}% (less overfitting)".format(round(linear_model_sgd_accuracy,2)))
def preprocessing(df):

    df['Char_length']       = df['text'].apply(len)

    df['word_count']        = df.text.apply(word_count)

    df['url_count']         = df.text.apply(url_counts)

    df['urls']              = df.text.apply(urls)

    df['text']              = df.text.apply(remove_urls)

    df['emoji_count']       = df.text.apply(emoji_count)

    df['emojis']            = df.text.apply(emoji_extraction)

    df['text']              = df['text'].apply(emoji_to_text)

    df['hash_count']        = df.text.apply(count_hashtags)

    df['hashtags']          = df.text.apply(find_hashtags)

    df['text']              = df.text.apply(remove_hashtags)

    df['text']              = df.text.apply(replace_username)

    df['count_number']      = df.text.apply(count_number)

    df['number']            = df.text.apply(find_number)

    df['text']              = df.text.apply(remove_number)

    df['count_punctuation'] = df.text.apply(count_punctuations)

    df['punctuation']       = df.text.apply(find_punctuations)

    df['text']              = df.text.apply(remove_punctuations)

    df['text']              = df.text.apply(remove_symbols)

                                            

    return df
%%time

processed_test_df = preprocessing(df_test)
def bagofword(df):

    X = vectorizer.transform(df.text.values)

    return X    



def predict_test(model,x):

    return model.predict(x)
X_test = bagofword(processed_test_df)

target = predict_test(gridcv,X_test)

submission['target'] = target

submission.to_csv('submission.csv',index=False)

submission.head()