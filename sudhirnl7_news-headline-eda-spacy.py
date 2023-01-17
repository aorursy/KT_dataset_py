import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
seed = 100
#path = 'dataset/'
path = '../input/'
news = pd.read_json(path+ 'Sarcasm_Headlines_Dataset.json',lines=True)
news.head()
news['num_words'] = news['headline'].apply(lambda x: len(str(x).split()))
print('Maximum number of word',news['num_words'].max())

print('\nSentence:\n',news[news['num_words'] == 39]['headline'].values)
text = news[news['num_words'] == 39]['headline'].values
# Word tokenize
nlp = spacy.load('en')
doc = nlp(text[0])

# List compresion method to get tokens
token = [w.text for w in doc ]
print(token)
# Data preprocessing
# Remove punctuation
print('Quotes:',spacy.lang.punctuation.LIST_QUOTES)
print('\nPunctuations:',spacy.lang.punctuation.LIST_PUNCT)
#print('\n Currency:',spacy.lang.punctuation.LIST_CURRENCY)

# list of punctuation contains most of punctuation, we will use only that for our analysis
punc = [w.text for w in doc  if  w.is_punct ]
print('\nPunctuation:',punc)
stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
print('Number of stopwords is','-'*20,len(stopwords))
print('Ten stop words',list(stopwords)[:10])
stop = [w.text for w in doc if w.is_stop]
print('*'*100,'\n\nStop word in sentence: ',stop)
digit = [w.text for w in doc if w.is_digit]
print('Digit in sentence: ',digit)
lemma = [w.lemma_ for w in doc]
print(lemma)
spacy.displacy.render(doc, style='ent', jupyter=True)
df = pd.DataFrame(
{
    'token': [w.text for w in doc],
    'lemma':[w.lemma_ for w in doc],
    'POS': [w.pos_ for w in doc],
    'TAG': [w.tag_ for w in doc],
    'DEP': [w.dep_ for w in doc],
    'is_stopword': [w.is_stop for w in doc],
    'is_punctuation': [w.is_punct for w in doc],
    'is_digit': [w.is_digit for w in doc],
})

def highlight_True(s):
    """
    Highlight True and False
    """
    return ['background-color: yellow' if v else '' for v in s]
df.style.apply(highlight_True,subset=['is_stopword', 'is_punctuation', 'is_digit'])
def clean_text(df):
    """
    Text preprocessing:
    tokenize, make lower case,
    Remove Stop word, punctuation, digit
    lemmatize
    """
    nlp = spacy.load('en')
    for i in range(df.shape[0]):
        doc = nlp(df['headline'][i])
        # Word Tokenize
        #token = [w.text for w in doc]
        
        # Make Lower case
        # Remove Stop word, punctuation, digit and lemmatize
        text = [w.lemma_.lower().strip() for w in doc 
               if not (w.is_stop |
                    w.is_punct |
                    w.is_digit)
               ]
        text = " ".join(text)
        
        if i <5: print('Sentence:',i,text)
        df['headline'][i] = text
    return df
news_df = clean_text(news)
sns.countplot(news['is_sarcastic'])
tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=5000)
X = tf.fit_transform(news_df['headline'])
y = news_df['is_sarcastic']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=seed)
nb = BernoulliNB()
nb.fit(X_train,y_train)
pred = nb.predict(X_valid)
print('Confusion matrix\n',confusion_matrix(y_valid,pred))
print('Classification_report\n',classification_report(y_valid,pred))
proba = nb.predict_proba(X_valid)[:,1]
fpr,tpr, threshold = roc_curve(y_valid,proba)
auc_val = auc(fpr,tpr)

plt.figure(figsize=(14,8))
plt.title('Reciever Operating Charactaristics')
plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc_val)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')