#import required packages
#basics
import pandas as pd 
import numpy as np
#nlp

import re    #for regex
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()


train = pd.read_csv("../input/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3,encoding='utf-8') 
test = pd.read_csv("../input/testData.tsv", header=0, \
                    delimiter="\t", quoting=3,encoding='utf-8')
train.head()
test.head()
train['review'][0]
print ("number of rows for sentiment 1: {}".format(len(train[train.sentiment==1])))
print ( "number of rows for sentiment 0: {}".format(len(train[train.sentiment==0])))
#sentiments are equally split
#concat both train and test
merge=pd.concat([train[['id','review']],test[['id','review']]])
df=merge.reset_index(drop=True)
print(df.head())

from bs4 import BeautifulSoup
def review_to_words( raw_review ):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, 'lxml').get_text() 
    
    # 2. Remove non-letters with regex
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                           
    
    # 4. Create set of stopwords
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

df['review_cleaned']=df['review'].apply(review_to_words)
df_justclean = df[['id','review_cleaned']]
df['count_word']=df["review_cleaned"].apply(lambda x: len(str(x).split()))             #Word count in each comment:
df['count_unique_word']=df["review_cleaned"].apply(lambda x: len(set(str(x).split()))) #split creates groups

df['count_letters']=df["review_cleaned"].apply(lambda x: len(str(x)))                  #Letter count
                                                                                       
df["mean_word_len"] = df["review_cleaned"].apply(                                           
    lambda x: np.mean([len(w) for w in str(x).split()]))                               #Average length of the words
df['word_unique_percent']=df['count_unique_word']*100/df['count_word']                    #Word count percent in each comment:
#serperate train and test features
train_feats=df.iloc[0:len(train),] 
test_feats=df.iloc[len(train):,]

train_tags=train['sentiment']
train_feats=pd.concat([train_feats,train_tags],axis=1)
train_feats.head()
train_feats.describe()
# place bounds 
train_feats['count_word'].loc[train_feats['count_word']>150] = 150                   # set columns with count sent longer than 90 to 90
train_feats['count_unique_word'].loc[train_feats['count_unique_word']>136] = 136     # set columns with count sent longer than 90 to 90
train_feats['count_letters'].loc[train_feats['count_letters']>1154] = 1154
train_feats[train_feats.sentiment==0].describe()
train_feats.rename(columns={"sentiment": "target_sentiment"},inplace=True)
train_feats.head()


merge['cleaned_review']=merge['review'].apply(review_to_words)
corpus = merge.cleaned_review
#dictionary of apostrophe words
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

#Its important to use a clean dataset before creating count features.

def clean(comment):
    words=tokenizer.tokenize(comment)                                   #Split the sentences into word
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word,"v") for word in words]                   #lemmatizes based on position v
    clean_sent=" ".join(words)
    return(clean_sent)
clean_corpus=corpus.apply(lambda x :clean(x))

# create vectorizer
tf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=0.001,  max_features=5000, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,2),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tf = tf_vectorizer.fit_transform(clean_corpus)
features=np.array(tf_vectorizer.get_feature_names())
print (features)
print (len(features))
df1 = pd.DataFrame(tf.toarray(),columns=features)
df1.head()
merged_df=pd.concat([df[['id','count_word','count_unique_word','count_letters','mean_word_len','word_unique_percent']],df1],axis=1)

X=merged_df.iloc[:len(train),1:]
Y=train['sentiment']
unk_features=merged_df.iloc[len(train):,1:]
unk_ids=merged_df.iloc[len(train):,0]
# from sklearn.linear_model import LogisticRegression #logistic regression
# #Logistic Regression has the highest accuracy
# from sklearn.model_selection import GridSearchCV
# C=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# penalty=['l1','l2']
# hyper={'C':C,'penalty':penalty}
# gd=GridSearchCV(estimator=LogisticRegression(random_state=0),param_grid=hyper,verbose=True,cv=5,scoring='roc_auc')
# gd.fit(X,Y)
# print(gd.best_score_)
# print(gd.best_estimator_)
clf=LogisticRegression(C=0.9,penalty='l2',n_jobs=-1)
rounds = 15
for i in range(rounds):
    clf.set_params(random_state = i + 1)
    clf.fit(X, Y)
    preds = clf.predict(unk_features)
    
sample_sub=pd.read_csv('../input/sampleSubmission.csv')
result=pd.DataFrame({'id':sample_sub['id'],'sentiment':preds}).reset_index(drop=True)
result.to_csv('result.csv', index = False)