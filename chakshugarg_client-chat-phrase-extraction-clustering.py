import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk,collections
from nltk.util import ngrams
import string
import re
import spacy
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud

data = pd.read_csv("../input/question2.csv")
print(data['transcript'].head())

stop_words = set(stopwords.words('english')) 
concatinated_data = data.to_string(header=False,index=False,index_names=False)
word_tokens = word_tokenize(concatinated_data) 
def cleanDataForNLP(TextData):
    TextData.lower()
    TextData = re.sub('[^A-Za-z]+', ' ', TextData)    
    word_tokens = word_tokenize(TextData)
    filteredText = ""
    for w in word_tokens:
        if w not in stop_words and len(w) > 2 and not w.isnumeric() and w not in string.punctuation:
            filteredText = filteredText + " " + w
    
    return filteredText.strip()
textData=cleanDataForNLP(concatinated_data)
tokenized = textData.split()
ngram = ngrams(tokenized, 3)
ngramFreq = collections.Counter(ngram)
mostcommonngrams=ngramFreq.most_common(10)
print(mostcommonngrams)
nlp = spacy.load('en')
nlpdata=nlp(textData)
posTags=[(x.text, x.pos_) for x in nlpdata if x.pos_ != u'SPACE']
nounPhrases=[np.text for np in nlpdata.noun_chunks]
nounPhrasesFiltered=[]
for nounPhrase in nounPhrases:
    if(len(nounPhrase.split(" "))>1):
        nounPhrasesFiltered.append(nounPhrase)

df = pd.DataFrame({'nounPhrases':nounPhrasesFiltered})
nounPhrasesCount=df['nounPhrases'].value_counts()
print("Extracted Phrases")
print(nounPhrasesCount.head(10))
wordcloud = WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['nounPhrases']))

print(wordcloud)
fig = plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
vectorizer = TfidfVectorizer()
vectorizedText = vectorizer.fit_transform(df['nounPhrases'])
words = vectorizer.get_feature_names()

kmeans = KMeans(n_clusters = 5, n_init = 20, n_jobs = 1) 
kmeans.fit(vectorizedText)
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
for index, row in df.iterrows():
    y = vectorizer.transform([row['nounPhrases']])
    print (row['nounPhrases'] + " - " + str(kmeans.predict(y)))