

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression

from sklearn import linear_model

from sklearn.metrics import log_loss





#NLP tools

import re

import string

import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud

stopwords = nltk.corpus.stopwords.words('english')





#Plot and image tools

from PIL import Image

from matplotlib import pyplot as plt

from matplotlib import gridspec

import seaborn as sns

sns.set_style("dark")



train_data_path='../input/jigsaw-toxic-comment-classification-challenge/train.csv'



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Loading the Data

train = pd.read_csv(train_data_path)

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
#a quick look at our training dataset

train.head()
# the size of our training dataset

train.shape
rowsums=train.iloc[:,2:].sum(axis=1)

train['clean']=(rowsums==0)

train['clean'].sum()

colors_list = ["brownish green", "pine green", "ugly purple",

               "blood", "deep blue", "brown", "azure"]



palette= sns.xkcd_palette(colors_list)



x=train.iloc[:,2:].sum()



plt.figure(figsize=(9,6))

ax= sns.barplot(x.index, x.values,palette=palette)

plt.title("Class")

plt.ylabel('Occurrences', fontsize=12)

plt.xlabel('Type ')

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, 

            ha='center', va='bottom')



plt.show()
# Just a list that contains all the text data. For me not to load the whole dataset everytime

comment_text_list = train.apply(lambda row : nltk.word_tokenize( row['comment_text']),axis=1)
#An odd comment contains a high rate of punctuation symbols or capital letters

rate_punctuation=0.7

rate_capital=0.7

def odd_comment(comment):

    punctuation_count=0

    capital_letter_count=0

    total_letter_count=0

    for token in comment:

        if token in list(string.punctuation):

            punctuation_count+=1

        capital_letter_count+=sum(1 for c in token if c.isupper())

        total_letter_count+=len(token)

    return((punctuation_count/len(comment))>=rate_punctuation or 

           (capital_letter_count/total_letter_count)>rate_capital)



odd=comment_text_list.apply(odd_comment)

odd_ones=odd[odd==True]

#list(ponctuation_polluted.index)

odd_comments=train.loc[list(odd_ones.index)]

odd_comments[odd_comments.clean==False].count()/len(odd_comments)
colors_list = ["brownish green", "pine green", "ugly purple",

               "blood", "deep blue", "brown", "azure"]



palette= sns.xkcd_palette(colors_list)



x=odd_comments.iloc[:,2:].sum()





plt.figure(figsize=(9,6))

ax= sns.barplot(x.index, x.values, alpha=0.8, palette=palette)

plt.title("# per class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)



rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
# quick check for empty comments

empty_com=train[train.comment_text==""]

empty_com
#quick check for duplicated comments

duplicate=train.comment_text.duplicated()

duplicate[duplicate==True]
#Just storing each categories of non clean comments in specific arrays

toxic=train[train.toxic==1]['comment_text'].values

severe_toxic=train[train.severe_toxic==1]['comment_text'].values

obscene=train[train.obscene==1]['comment_text'].values

threat=train[train.threat==1]['comment_text'].values

insult=train[train.insult==1]['comment_text'].values

identity_hate=train[train.identity_hate==1]['comment_text'].values
mask=np.array(Image.open('../input/imagetc/twitter.png'))

mask=mask[:,:,1]

from wordcloud import WordCloud, STOPWORDS

# The wordcloud of Toxic Comments

plt.figure(figsize=(16,13))

wc = WordCloud(background_color="black", max_words=500,mask=mask 

             , stopwords=stopwords, max_font_size= 60)

wc.generate(" ".join(toxic))

plt.title("Twitter Wordlcloud Toxic Comments", fontsize=30)

# plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

plt.imshow(wc.recolor( colormap= 'Set1' , random_state=1), alpha=0.98)

plt.axis('off')

plt.savefig('twitter_wc.png')




replacement_patterns = [

 (r'won\'t', 'will not'),

 (r'can\'t', 'cannot'),

 (r'i\'m', 'i am'),

 (r'ain\'t', 'is not'),

 (r'(\w+)\'ll', '\g<1> will'),

 (r'(\w+)n\'t', '\g<1> not'),

 (r'(\w+)\'ve', '\g<1> have'),

 (r'(\w+)\'s', '\g<1> is'),

 (r'(\w+)\'re', '\g<1> are'),

 (r'(\w+)\'d', '\g<1> would')

]

class RegexpReplacer(object):

    def __init__(self, patterns=replacement_patterns):

         self.patterns = [(re.compile(regex), repl) for (regex, repl) in

         patterns]

     

    def replace(self, text):

        s = text

        for (pattern, repl) in self.patterns:

             s = re.sub(pattern, repl, s)

        return s
from nltk.stem import WordNetLemmatizer

lemmer = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')

from nltk.tokenize import TweetTokenizer

#from replacers import RegexpReplacer

replacer = RegexpReplacer()

tokenizer=TweetTokenizer()



def comment_process(category):

    category_processed=[]

    for i in range(category.shape[0]):

        comment_list=tokenizer.tokenize(replacer.replace(category[i]))

        comment_list_cleaned= [word for word in comment_list if ( word.lower() not in stopwords 

                              and word.lower() not in list(string.punctuation) )]

        comment_list_lemmed=[lemmer.lemmatize(word, 'v') for word in comment_list_cleaned]

        category_processed.extend(list(comment_list_lemmed))

    return category_processed

toxic1=comment_process(toxic)
fd=nltk.FreqDist(word for word in toxic1)



x=[fd.most_common(150)[i][0] for i in range(99)]

y=[fd.most_common(150)[i][1] for i in range(99)]

#palette=sns.color_palette("PuBuGn_d",100)

palette= sns.light_palette("crimson",100,reverse=True)

plt.figure(figsize=(45,15))

ax= sns.barplot(x, y, alpha=0.8,palette=palette)

plt.title("Occurences per word in Toxic comments 1")

plt.ylabel('Occurrences', fontsize=30)

plt.xlabel(' Word ', fontsize=30)

#adding the text labels

rects = ax.patches

labels = y

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

    plt.xticks(rotation=60, fontsize=18)

#plt.savefig('Toxic_Word_count1.png')    

plt.show()





toxic2=[]

for i in range(toxic.shape[0]):

    comment_list=nltk.word_tokenize(toxic[i])

    comment_list_cleaned= [word for word in comment_list if ( word.lower() not in stopwords 

                          and word.lower() not in list(string.punctuation) )]

    toxic2.extend(list(set(comment_list_cleaned)))
fd2=nltk.FreqDist(word for word in toxic2)

x=[fd2.most_common(100)[i][0] for i in range(99)]

y=[fd2.most_common(100)[i][1] for i in range(99)]

palette= sns.light_palette("crimson",100,reverse=True)

plt.figure(figsize=(45,15))

ax= sns.barplot(x, y, alpha=0.8,palette=palette)

plt.title("Occurence per word in Toxic comments")

plt.ylabel('Occurrences', fontsize=30)

plt.xlabel(' Word ', fontsize=30)



rects = ax.patches

labels = y

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

    plt.xticks(rotation=60, fontsize=18)

#plt.savefig('Toxic_Word_count2.png')    

plt.show()

#Check the bigger picture in imagetc files 
def wordcloud_plot(category, name) : 

    plt.figure(figsize=(20,15))

    wc = WordCloud(background_color="black", max_words=500,mask=mask, min_font_size=6 

                 , stopwords=stopwords, max_font_size= 60)

    wc.generate(" ".join(category))

    plt.title("Twitter Wordlcloud " + name +  " Comments", fontsize=30)

    # plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

    plt.imshow(wc.recolor( colormap= 'Set1' , random_state=21), alpha=0.98)

    plt.axis('off')

    plt.savefig(name+'_wc.png')

    return(True)



wordcloud_plot(toxic1,'Toxic')
severe_toxic1=comment_process(severe_toxic)

obscene1=comment_process(obscene)

threat1=comment_process(threat)

insult1=comment_process(insult)

identity_hate1=comment_process(identity_hate)
wordcloud_plot(severe_toxic1,'Severe_toxic')
wordcloud_plot(obscene1,'Obscene')
wordcloud_plot(threat1,'Threat')
wordcloud_plot(insult1,'Insult')
wordcloud_plot(identity_hate1,'Identity_Hate')
toxic3=train[train.clean==False]['comment_text']

toxic3

toxic_count_word=toxic3.apply(lambda x: len(str(x).split()))

print(toxic_count_word.describe(),'\n \n',toxic_count_word.quantile(q=0.9))
clean3=train[train.clean==True]['comment_text']

clean_count_word=clean3.apply(lambda x: len(str(x).split()))

print(clean_count_word.describe(), clean_count_word.quantile(q=0.9))
x=rowsums.value_counts()



#plot

plt.figure(figsize=(8,4))

ax = sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Multiple tags per comment")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# of tags ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
#Credit Jagan



train['count_unique_word']=train["comment_text"].apply(lambda x: len(set(str(x).split())))

train['count_word']=train["comment_text"].apply(lambda x: len(str(x).split()))

train['word_unique_percent']=train['count_unique_word']*100/train['count_word']

spammers=train[train['word_unique_percent']<30]



plt.figure(figsize=(16,12))

plt.suptitle("What's so unique ?",fontsize=20)

#gridspec.GridSpec(2,1)





plt.subplot2grid((2,1),(0,0))

plt.title("Percentage of unique words of total words in comment")

#sns.boxplot(x='clean', y='word_unique_percent', data=train_feats)

ax=sns.kdeplot(train[train.clean == 0].word_unique_percent, label="Bad",shade=True,color='r')

ax=sns.kdeplot(train[train.clean == 1].word_unique_percent, label="Clean")

plt.legend()

plt.ylabel('Number of occurances', fontsize=12)

plt.xlabel('Percent unique words', fontsize=12)



x=spammers.iloc[:,2:9].sum()

plt.subplot2grid((2,1),(1,0),colspan=2)

plt.title("Count of comments with low(<30%) unique words",fontsize=15)

ax=sns.barplot(x=x.index, y=x.values,color='crimson')



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.xlabel('Threat class', fontsize=12)

plt.ylabel('# of comments', fontsize=12)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer



tf = TfidfVectorizer( strip_accents='unicode',analyzer='word',ngram_range=(1,1),

            use_idf=True,smooth_idf=True,sublinear_tf=True,

            stop_words = 'english')

"""tf = TfidfVectorizer(min_df=100,  max_features=100000, 

            strip_accents='unicode', analyzer='word',ngram_range=(1,1),

            use_idf=1,smooth_idf=1,sublinear_tf=1)"""
def category_to_tfidf(category):

    tvec_weights = tf.fit_transform(category)

    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()

    weights_df = pd.DataFrame({'term': tf.get_feature_names(), 'weight': weights})

    return(weights_df.sort_values(by='weight', ascending=False).head(10))



toxic_idf=category_to_tfidf(toxic1)

severe_toxic_idf=category_to_tfidf(severe_toxic1)

threat_idf=category_to_tfidf(threat1)

insult_idf=category_to_tfidf(insult1)

obscene_idf=category_to_tfidf(obscene1)

identity_hate_idf=category_to_tfidf(identity_hate1)

toxic_idf

color_list = ["xkcd:brownish green", "xkcd:pine green", "xkcd:ugly purple",

               "xkcd:blood", "xkcd:deep blue", "xkcd:brown"]

plt.figure(figsize=(20,22))

plt.suptitle("TF-IDF ranking ",fontsize=20)

gridspec.GridSpec(3,2)

plt.subplot2grid((3,2),(0,0))

sns.barplot(toxic_idf.term,

            toxic_idf.weight,color=color_list[0])

plt.title("Toxic",fontsize=15)

plt.xlabel('Term', fontsize=12)

plt.ylabel('Score', fontsize=12)



plt.subplot2grid((3,2),(0,1))

sns.barplot(severe_toxic_idf.term,

            severe_toxic_idf.weight,color=color_list[1])

plt.title(" Severe toxic",fontsize=15)

plt.xlabel('Term', fontsize=12)

plt.ylabel('Score', fontsize=12)





plt.subplot2grid((3,2),(1,0))

sns.barplot(obscene_idf.term,

            obscene_idf.weight,color=color_list[2])

plt.title("Obscene",fontsize=15)

plt.xlabel('Term', fontsize=12)

plt.ylabel('Score', fontsize=12)





plt.subplot2grid((3,2),(1,1))

sns.barplot(threat_idf.term,

            threat_idf.weight,color=color_list[3])

plt.title("Threat",fontsize=15)

plt.xlabel('Word', fontsize=12)

plt.ylabel('Score', fontsize=12)





plt.subplot2grid((3,2),(2,0))

sns.barplot(insult_idf.term,

            insult_idf.weight,color=color_list[4])

plt.title("Insult",fontsize=15)

plt.xlabel('Term', fontsize=12)

plt.ylabel('Score', fontsize=12)





plt.subplot2grid((3,2),(2,1))

sns.barplot(identity_hate_idf.term,

            identity_hate_idf.weight,color=color_list[5])

plt.title("Identity hate",fontsize=15)

plt.xlabel('Term', fontsize=12)

plt.ylabel('Score', fontsize=12)





plt.show()