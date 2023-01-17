import nltk



paragraph =  """I have three visions for India. In 3000 years of our history, people from all over 

               the world have come and invaded us, captured our lands, conquered our minds. 

               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,

               the French, the Dutch, all of them came and looted us, took over what was ours. 

               Yet we have not done this to any other nation. We have not conquered anyone. 

               We have not grabbed their land, their culture, 

               their history and tried to enforce our way of life on them. """
#Cleaning the texts.



import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



ps=PorterStemmer()

lemm=WordNetLemmatizer()



sentences=nltk.sent_tokenize(paragraph)

corpus=[]



for i in range(len(sentences)):

    review = re.sub('[^a-zA-Z]', ' ',sentences[i])

    review=review.lower()

    review=review.split()

    review=[lemm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]

    review=' '.join(review)

    corpus.append(review)

    

print(corpus)
#creating TF IDF.

from sklearn.feature_extraction.text import TfidfVectorizer

cv=TfidfVectorizer()

x=cv.fit_transform(corpus).toarray()
x.shape