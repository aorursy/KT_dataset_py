# load packages
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

# tokenizer for stemming and lemmatization
def tokenize(text):
    def convert_tag(tag):
        part = {'ADJ' : 'a',
                'ADV' : 'r',
                'VERB' : 'v',
                'NOUN' : 'n'}
        if tag in part.keys():
            return part[tag]
        else:
            # other parts of speech will be tagged as nouns
            return 'n'

    tokens = nltk.word_tokenize(text)
    tokens_pos = pos_tag(tokens,tagset='universal')
    stems = []
    for item in tokens_pos:
        term = item[0]
        pos = item[1]
        # stem/lemmatize tokens consisting of alphabetic characters only
        if term.isalpha():
            stems.append(WordNetLemmatizer().lemmatize(term, pos=convert_tag(pos)))
            #stems.append(PorterStemmer().stem(item))
    return stems

# vectorize corpus
train_df = pd.read_csv("../input/train.csv")

print("Train shape : ",train_df.shape)
print(train_df.columns)

# target variable and some basic stats
y_all = train_df['target']
print('fraction of datapoints in class 1: ',1e0*np.sum(y_all == 1)/len(y_all)) # fraction of datapoints in class 1
print('number of datapoints in class 1: ',np.sum(y_all == 1)) # number of datapoints in class 1

# n-grams with n = 1 - 3, no stopwords, use words that appear in at least min_df documents
vectorizer = TfidfVectorizer(ngram_range=(1,3),tokenizer=tokenize,min_df=100,\
                             sublinear_tf=True) 
X_all = vectorizer.fit_transform(train_df['question_text'])#.sample(100,random_state=seed))
print(np.shape(X_all))
# a quick look at the unique terms in the corpus
terms = np.array(vectorizer.get_feature_names())
print(terms[:100]) # the first 100 terms
print(terms[-100:]) # the last 100 terms
print(np.random.choice(terms,size=100,replace=False)) # 100 randomly selected terms
# n -- number of times the terms appear in docs 
term_count = X_all.getnnz(axis=0)

indices = np.where(y_all == 1)[0]
# tf_c1 -- the fraction of times the terms appear in class 1
frac_in_class1 = 1e0*X_all.tocsc()[indices].getnnz(axis=0)/term_count

indcs = np.where((frac_in_class1 <= 0.001) & (term_count >= 1000))[0]
print(terms[indcs]) # terms in sincere questions

indcs = np.where((frac_in_class1 >= 0.7) & (term_count >= 100))[0]
print(terms[indcs]) # terms in insincere questions

# plotting: there aren't so many terms in the upper right quadrant of the plot

plt.scatter(term_count,frac_in_class1)
plt.semilogx()
plt.xlabel('n - # times term in corpus')
plt.ylabel('tf_c1 - class-specific frequency')
plt.title('scatter plot')
plt.show()

xbins = 10**np.linspace(2,5,31)
ybins = np.linspace(0,1,41)
counts, _, _ = np.histogram2d(term_count,frac_in_class1,bins=(xbins,ybins))
counts[counts == 0] = 0.5 # so log10(0) is not nan
plt.pcolormesh(xbins, ybins, np.log10(counts.T))
plt.semilogx()
plt.xlabel('n - # times term in corpus')
plt.ylabel('tf_c1 - class-specific frequency')
cbar = plt.colorbar(label='count',ticks=[0,1,2,3])
cbar.ax.set_yticklabels([1,10,100,1000])
plt.title('heatmap')
plt.show()
