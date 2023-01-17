import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
%matplotlib inline
train_data = pd.read_csv('../input/moviereviewskaggla/train.tsv',sep='\t')
test_data = pd.read_csv('../input/test-data/test.tsv',sep='\t')
train_data.head(10)
Sentiment_words=[]
for row in train_data['Sentiment']:
    if row ==0:
        Sentiment_words.append('negative')
    elif row == 1:
        Sentiment_words.append('somewhat negative')
    elif row == 2:
        Sentiment_words.append('neutral')
    elif row == 3:
        Sentiment_words.append('somewhat positive')
    elif row == 4:
        Sentiment_words.append('positive')
    else:
        Sentiment_words.append('Failed')
train_data['Sentiment_words'] = Sentiment_words
train_data.head()
word_count =pd.value_counts(train_data['Sentiment_words'].values, sort=True)
word_count
Index = [1,2,3,4,5]
plt.figure(figsize=(15,5))
plt.bar(Index,word_count,color = 'blue')
plt.xticks(Index,['neutral','somewhat positive','somewhat negative','positive','negative'],rotation=45)
plt.ylabel('word_count')
plt.xlabel('word')
plt.title('Count of Moods')
plt.bar(Index, word_count)
for a,b in zip(Index, word_count):
    plt.text(a, b, str(b) ,color='green', fontweight='bold')
#Clean the column Phrase
def review_to_words(raw_review): 
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))
#Create a list using the review_to_words function to clean the phrases
#and convert them to a list
corpus= []
for i in range(0, 156060):
    corpus.append(review_to_words(train_data['Phrase'][i]))
train_data['new_Phrase']=corpus
train_data.drop(['Phrase'],axis=1,inplace=True)
train_data.head(10)
from  sklearn.feature_extraction.text  import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x_train = cv.fit_transform(corpus).toarray()
#y = array of sentiments corresponding to the sentiment column
y = train_data.iloc[:, 2].values
from sklearn.naive_bayes import MultinomialNB

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y, test_size = 0.40, random_state = 0)


classifier = MultinomialNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

mse = ((y_pred - y_test) ** 2).mean()
mse
rmse = sqrt(mse)
rmse
mat = pd.crosstab(y_test, y_pred)
mat
sommeTot=0
sommeDiago=0
sommeFlex1=0
sommeFlex2orMore=0

for i in range(0,5):
    for j in range(0,5):
        sommeTot+=int(mat.iloc[i,j])
        if i==j:
            sommeDiago+=int(mat.iloc[i,j])
        elif i-j==1 or i-j==-1:
            sommeFlex1+=int(mat.iloc[i,j])
        else :  
            sommeFlex2orMore+=int(mat.iloc[i,j])
qualiteParfaite=sommeDiago/sommeTot
qualitePresqueParfaite=sommeFlex1/sommeTot
qualiteBof=sommeFlex2orMore/sommeTot

print("Parfaitement classés:")
print(qualiteParfaite)
print("Presque parfaitement classés:")
print(qualitePresqueParfaite+qualiteParfaite)
print("Mal classés:")
print(qualiteBof)