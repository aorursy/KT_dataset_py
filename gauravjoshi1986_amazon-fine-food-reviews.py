import pandas as pd

import numpy as np



np.random.seed(0)



def read_text_file(f):

    df_complete = pd.read_csv(f)

    #df = df_complete.loc[:,["Text","Score"]]

    #df.dropna(how="any", inplace=True)    

    return df_complete



df = read_text_file("../input/Reviews.csv")

print (df.head())
df.shape



def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



Score = df['Score']

Score = Score.map(partition)



df['Review'] = df['Score'].map(partition)



print(df.head())
df_count_prcnt = df.Score.value_counts()



def compute_percentage(x):

    pct = float(x/df_count_prcnt.sum()) * 100

    return round(pct, 2)



df_count_prcnt = df_count_prcnt.apply(compute_percentage)



df_count_prcnt.plot(kind="bar", colormap='jet')



print(df_count_prcnt)
# frequency counts for users|product etc.



def top_n_counts (n, col, col_1):

    gb = df.groupby(col)[col_1].count()

    gb = gb.sort_values(ascending=False)

    return gb.head(n)

     

top_n_counts(15, ['ProductId','UserId'], 'ProductId')
top_n_counts(15, ['UserId'], 'UserId').plot(kind='bar', figsize=(20,10), colormap='winter')



print (top_n_counts(15, ['UserId'], 'UserId'))
df[(df['ProductId'] == 'B000WFN0VO') & (df['UserId'] == 'A29JUMRL1US6YP')][['Text','Score']]
# Time series for monthly review counts

df['datetime'] = pd.to_datetime(df["Time"], unit='s')

df_grp = df.groupby([df.datetime.dt.year, df.datetime.dt.month, df.Score]).count()['ProductId'].unstack().fillna(0)





df_grp.plot(figsize=(20,10), rot=45, colormap='jet')
# Time series for monthly review counts

df['datetime'] = pd.to_datetime(df["Time"], unit='s')

df_grp = df.groupby([df.datetime.dt.year, df.datetime.dt.month, df.Score]).count()['ProductId'].unstack()



df_grp.plot(kind="bar",figsize=(30,10), stacked=True, colormap='jet')
pos = df.loc[df['Review'] == 'positive']

pos = pos[0:25000]



neg = df.loc[df['Review'] == 'negative']

neg = neg[0:25000]
import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud

import string

import matplotlib.pyplot as plt



def create_Word_Corpus(df):

    words_corpus = ''

    for val in pos["Summary"]:

        text = val.lower()

        #text = text.translate(trantab)

        tokens = nltk.word_tokenize(text)

        tokens = [word for word in tokens if word not in stopwords.words('english')]

        for words in tokens:

            words_corpus = words_corpus + words + ' '

    return words_corpus

        

# Generate a word cloud image

pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))

neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))



# Plot cloud

def plot_Cloud(wordCloud):

    plt.figure( figsize=(20,10), facecolor='k')

    plt.imshow(wordCloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.show()

    plt.savefig('wordclouds.png', facecolor='k', bbox_inches='tight')
plot_Cloud(pos_wordcloud)
plot_Cloud(neg_wordcloud)
def sampling_dataset(df):

    count = 5000

    class_df_sampled = pd.DataFrame(columns = ["Score","Text", "Review"])

    temp = []

    for c in df.Score.unique():

        class_indexes = df[df.Score == c].index

        random_indexes = np.random.choice(class_indexes, count, replace=False)

        temp.append(df.loc[random_indexes])

        

    for each_df in temp:

        class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)

    

    return class_df_sampled



df_Sample = sampling_dataset(df.loc[:,["Score","Text","Review"]])

df_Sample.reset_index(drop=True,inplace=True)

print (df_Sample.head())

print (df_Sample.shape)
from nltk import clean_html

from nltk import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

import re



stop = stopwords.words('english')



def Tokenizer(df):

    comments = []

    for index, datapoint in df.iterrows():

        # Strips punctuation/abbr, converts to lowercase

        text = re.sub('<[^>]*>', '', datapoint["Text"])

        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())

        text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

        # Tokenizes into sentences

        tokenized_words = [w for w in text.split() if w not in stop]

        comments.append(tokenized_words)

    df["clean_reviews"] = comments

    return df



def stemming(df):

    # Stem all words with stemmer

    comments = []

    for index, datapoint in df.iterrows():

        # Strips punctuation/abbr, converts to lowercase

        text = re.sub('<[^>]*>', '', datapoint["Text"])

        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())

        text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

        # Stemming

        porter = PorterStemmer()

        stem_words = [porter.stem(w) for w in text.split() if w not in stop]

        comments.append(stem_words)

    df["clean_reviews"] = comments

    return df
df_Sample = stemming(df_Sample)



print (df_Sample.head())

print (df_Sample.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  

# min_df=5, discard words appearing in less than 5 documents

# max_df=0.8, discard words appering in more than 80% of the documents

# sublinear_tf=True, use sublinear weighting

# use_idf=True, enable IDF

vectorizer = TfidfVectorizer(min_df=5,

                             max_df = 0.8,

                             sublinear_tf=True,

                             use_idf=True)



train_vectors = vectorizer.fit_transform(df_Sample["Text"])

feature_names = vectorizer.get_feature_names()
# Take a look at the words in the vocabulary

vocab = vectorizer.get_feature_names()

print (vocab[1:200])


print ("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_vectors, df_Sample["Review"] )



# prediction_rbf = classifier_rbf.predict(test_vectors)
df_TestSample = sampling_dataset(df.loc[:,["Score","Text","Review"]])

df_TestSample.reset_index(drop=True,inplace=True)



print (df_TestSample.head())

print (df_TestSample.shape)
test_vectors = vectorizer.transform(df_TestSample["Review"])

prediction_rbf = forest.predict(test_vectors)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
Classifiers = [

    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

    #KNeighborsClassifier(3),

    #SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    #RandomForestClassifier(n_estimators=200),

    #AdaBoostClassifier(),

    #GaussianNB()

]
dense_features=train_vectors.toarray()

dense_test= test_vectors.toarray()



Accuracy=[]

Model=[]

for classifier in Classifiers:

    print('training '+classifier.__class__.__name__)

    try:

        fit = classifier.fit(train_vectors,df_Sample["Review"])

        pred = fit.predict(test_vectors)

    except Exception:

        fit = classifier.fit(dense_features,df_Sample["Review"])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,df_TestSample["Review"])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))    
#Compare the model performances

Index = [1,2]

plt.bar(Index,Accuracy)

plt.xticks(Index, Model,rotation=45)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')