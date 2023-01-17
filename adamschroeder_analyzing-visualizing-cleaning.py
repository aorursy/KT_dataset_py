import time
start_time=time.time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
numrow_train=train.shape[0]
numrow_test=test.shape[0]
sum=numrow_train+numrow_test
print("       : train : test")
print("rows   :",numrow_train,":",numrow_test)
print("percnt :",round(numrow_train*100/sum),"   :",round(numrow_test*100/sum))
x=train.iloc[:,2:].sum()

#marking comments without any tags
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)

#count number of tags
train['clean'].sum()

print("Total comments: ", len(train))
print("Total % of Clean comments: ",round(((train['clean'].sum())/len(train))*100),"%")
print("Total % of Toxic comments: ",round(((len(train)-(train['clean'].sum()))/len(train))*100),"%")
train['clean'] = train['clean'].astype(int) #convert boolean to numbers zero or one
x=train.iloc[:,2:].sum()                    #sum all rows

fig=plt.gcf()
fig.set_size_inches(1,3)
sns.barplot(x.values[6],orient='vertical')
plt.xlabel('Clean comments')
plt.title('# of Clean comments')
plt.show()

all_tags_but_clean = x.drop('clean')
sns.barplot(all_tags_but_clean.index, all_tags_but_clean.values)
plt.title('# of \'Unclean\' comments')
plt.show()
print("\nIt appears that what is insulting is usually obscene as well")
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #correlation matrix of our data
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.xticks(rotation=90)
plt.show()
toxic_comnts = train[train.toxic==1]['comment_text']
toxic_comnts = toxic_comnts[:10000]                                          # reduce comments size due to memory limitations
t_tokenized_sents = [word_tokenize(sentences) for sentences in toxic_comnts] # create sentences
t_tagged = [pos_tag(words) for words in t_tokenized_sents]                   # create tags for each token
t_categories = [y for lists in t_tagged for x,y in lists]                    # extract tags
t_counting_catg = Counter(t_categories)                                      # count tags
df = pd.DataFrame.from_dict(t_counting_catg, orient='index').reset_index()
df.columns=('category','amount')
df = df.sort_values(by='amount', ascending=False)
df = df[:15]                                                                 # limit to the top 15 highest counts of taggs
df['%'] = df['amount']/ df['amount'].sum(axis=0)                             # calculate percentage

sns.barplot(x='category', y='%', data=df)            
plt.title("% of the time a tag appears in Toxic comments")
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()
clean_comnts = train[train.toxic==0]['comment_text']
clean_comnts = clean_comnts[:10000]                                           # reduce comments size due to memory limitations
tokenized_sents = [word_tokenize(sentences) for sentences in clean_comnts]
tagged = [pos_tag(words) for words in tokenized_sents] 
categories = [y for lists in tagged for x,y in lists]                     
counting_catg = Counter(categories)                            
dfc = pd.DataFrame.from_dict(counting_catg, orient='index').reset_index()
dfc.columns=('category','amount')
dfc = dfc.sort_values(by='amount', ascending=False)
dfc = dfc[:15]                                                 
dfc['%'] = dfc['amount']/ dfc['amount'].sum(axis=0)            

sns.barplot(x='category', y='%', data=dfc)                     
plt.title("% of the time a tag appears in Clean comments")
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()
print(nltk.help.upenn_tagset())
# Word-tokenize all toxic comments
tknzr = TweetTokenizer()
t_sentences = [sent_tokenize(s) for s in toxic_comnts]
t_lists_of_words = [tknzr.tokenize(words) for strings in t_sentences for words in strings]

# Keep only letters (remove special characters and digits) 
t_only_letters = [regexp_tokenize(token, pattern='[a-zA-Z]+') for sublist in t_lists_of_words for token in sublist]

# Flatten list of lists
t_flat_list = [token for sublist in t_only_letters for token in sublist]

# Lowercase all words
t_all_lower = [l.lower() for l in t_flat_list]

# Remove stop words
stopWords = set(stopwords.words('english'))
t_no_stpwrds = [token for token in t_all_lower if token not in stopWords]

# Count words and list most common ones
commonly_toxic = Counter(t_no_stpwrds)
wording, counting = [], []
for x,y in commonly_toxic.most_common(n=20):
    wording.append(x)
    counting.append(y)
sns.barplot(x=wording, y=counting)
plt.xticks(rotation=60)
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.title("Toxic comments: top 20 most used words")
plt.ylabel('Count')
plt.show()
# Word-tokenize clean comments
tknzr = TweetTokenizer()
sentences = [sent_tokenize(s) for s in clean_comnts]
lists_of_words = [tknzr.tokenize(words) for strings in sentences for words in strings]

# Keep only letters (remove special characters and digits) 
only_letters = [regexp_tokenize(token, pattern='[a-zA-Z]+') for sublist in lists_of_words for token in sublist]

# Flatten list of lists
flat_list = [token for sublist in only_letters for token in sublist]

# Lowercase all words
all_lower = [l.lower() for l in flat_list]

# Remove stop words
stopWords = set(stopwords.words('english'))
no_stpwrds = [token for token in all_lower if token not in stopWords]

#Count words and list most common ones
commonly_clean = Counter(no_stpwrds)
words, counts = [], []
for x,y in commonly_clean.most_common(n=20):
    words.append(x)
    counts.append(y)

sns.barplot(x=words, y=counts)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
plt.xticks(rotation=60)
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.title("Clean comments: top 20 most used words")
plt.ylabel('Count')
plt.show()
from wordcloud import WordCloud

# Convert list of words into one long string for WordCloud to read
toxicity = ' '.join(t_no_stpwrds)

#0 rel_scaling= only word-ranks are considered; 1 rel_scaling = a word that is twice as frequent will have twice the size
#min font size = anything smaller than integer won't go in wordcloud
wordcloud2 = WordCloud(collocations=False, relative_scaling=1, min_font_size=13,width=400, height=200)
wordcloud2.generate(toxicity)
plt.figure(figsize=(16,7))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.title('Toxic WordCloud')
plt.show()
clean = ' '.join(no_stpwrds)

#0 rel_scaling= only word-ranks are considered; 1 rel_scaling = a word that is twice as frequent will have twice the size
#min font size = anything smaller than integer won't go in wordcloud
wordcloud2 = WordCloud(collocations=False, relative_scaling=1, min_font_size=13,width=400, height=200)
wordcloud2.generate(clean)
plt.figure(figsize=(16,7))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.title('Clean WordCloud')
plt.show()
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
# Trigrams
finder = TrigramCollocationFinder.from_words(t_no_stpwrds)

# only trigrams that appear 40+ times
finder.apply_freq_filter(40)
 
# return the 50 trigrams with the highest PMI
print (finder.nbest(trigram_measures.pmi, 50))
ttoxic = nltk.Text(t_no_stpwrds)
print (ttoxic.concordance('homeland', 54, lines=10))
severe_tox = train[train['severe_toxic']==1]['comment_text']
print("Mean length of characters in severly toxic comments:", round(severe_tox.map(lambda x: len(x)).mean()))
print("Median length of characters in severly toxic comments:",round(severe_tox.map(lambda x: len(x)).median()))
# A few severe toxic comments and their length
severe_tox.map(lambda x: len(x))[20:26]
print(severe_tox[2249])
train['uniq_word']=train['comment_text'].apply(lambda x: len(set(str(x).split())))
cl1 = train.loc[train['clean']==1]['uniq_word'].median()
sev1 = train.loc[train['severe_toxic']==1]['uniq_word'].median()
tox1 = train.loc[train['toxic']==1]['uniq_word'].median()
ob1 = train.loc[train['obscene']==1]['uniq_word'].median()
thrt1 = train.loc[train['threat']==1]['uniq_word'].median()
insl1 = train.loc[train['insult']==1]['uniq_word'].median()
ih1 = train.loc[train['identity_hate']==1]['uniq_word'].median()

cag_uniq = [cl1,sev1,tox1,ob1,thrt1,insl1,ih1]
labels=['clean','severe','toxic','obscene','threat','insult','identity-hate']
order_of_labels = range(len(cag_uniq))

plt.figure(figsize=(12,4))
plt.plot(cag_uniq, 'r--', marker='o')
plt.xticks(order_of_labels, labels)
plt.title('Average ammount of unique words in each comment category')
plt.show()
import string
train['punc_len'] = train["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
c2 = train.loc[train['clean']==1]['punc_len'].median()
s2 = train.loc[train['severe_toxic']==1]['punc_len'].median()
t2 = train.loc[train['toxic']==1]['punc_len'].median()
ob2 = train.loc[train['obscene']==1]['punc_len'].median()
th2 = train.loc[train['threat']==1]['punc_len'].median()
ins2 = train.loc[train['insult']==1]['punc_len'].median()
ih2 = train.loc[train['identity_hate']==1]['punc_len'].median()

cat2 = [c2,s2,t2,ob2,th2,ins2,ih2]
labels=['clean','severe','toxic','obscene','threat','insult','identity-hate']
order_of_labels = range(len(cat2))

plt.figure(figsize=(12,4))
plt.plot(cat2, 'r--', marker='o')
plt.xticks(order_of_labels, labels)
plt.title('Average ammount of punctuation in each comment category')
plt.show()
train['len_I'] = train['comment_text'].apply(lambda x: len(re.findall(r"(\s+-?I\s)|(\s[iI]'m\s)|(\si\s)|(\s[iI]'d\s)|((\s[iI]'ve\s)|((\s[iI]'ll\s)))", str(x))))
CL = train.loc[train['clean']==1]['len_I'].mean()
SV = train.loc[train['severe_toxic']==1]['len_I'].mean()
TO = train.loc[train['toxic']==1]['len_I'].mean()
INS = train.loc[train['insult']==1]['len_I'].mean()
TH = train.loc[train['threat']==1]['len_I'].mean()
OB = train.loc[train['obscene']==1]['len_I'].mean()
IH = train.loc[train['identity_hate']==1]['len_I'].mean()

c_list = [CL, SV, TO, INS, TH, OB, IH]
labels=['clean','severe','toxic','insult','threat','obscence','identity-hate']
order_of_labels = range(len(c_list))

plt.figure(figsize=(12,4))
plt.bar(x=order_of_labels ,height=c_list, color='purple')
plt.xticks(order_of_labels, labels, rotation=90)
plt.title('Average ammount of personal pronoun (I) in each comment category')
plt.show()
train['len_will'] = train['comment_text'].apply(lambda x: len(re.findall(r"\s[wW]ill|[Iiuey]'ll|gonna", str(x))))
c = train[train['clean']==1]
CLEAN = len(c[c['len_will']>0]) / len(c)

s = train[train['severe_toxic']==1]
SEVERE = len(s[s['len_will']>0]) / len(s)

t = train[train['toxic']==1]
TOXIC = len(t[t['len_will']>0]) / len(t)

i = train[train['insult']==1]
INSULT = len(i[i['len_will']>0]) / len(i)

thr = train[train['threat']==1]
THREAT = len(thr[thr['len_will']>0]) / len(thr)

o = train[train['obscene']==1]
OBSCENE = len(o[o['len_will']>0]) / len(o)

ith = train[train['identity_hate']==1]
IDENTITY = len(ith[ith['len_will']>0]) / len(ith)
cag = [CLEAN,SEVERE,TOXIC,INSULT,THREAT,OBSCENE,IDENTITY]
labels=['clean','severe','toxic','insult','threat','obscence','identity-hate']
order_of_labels = range(len(cag))

plt.figure(figsize=(12,4))
plt.bar(x=order_of_labels ,height=cag, color='purple')
plt.xticks(order_of_labels, labels, rotation=90)
plt.ylabel('Percentage %')
plt.title('Percentage of comments that use the future tense')
plt.show()
sample = train.copy()
for comments in train,test:
    # remove '\\n'
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
    # remove any text starting with User... 
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
    
    # remove IP addresses or user IDs
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    
    # lower uppercase letters
    comments['comment_text'] = comments['comment_text'].map(lambda x: str(x).lower())
    
    #remove http links in the text
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))
    
    #remove all punctuation except for apostrophe (')
    comments['comment_text'] = comments['comment_text'].map(lambda x: re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','',str(x)))

print("**Before:**\n")
print (sample['comment_text'][73])
print("**After:**\n")
print(train['comment_text'][73])
end_time=time.time()
print("total time",end_time-start_time)
