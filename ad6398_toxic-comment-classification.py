import pandas as pd

import string

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

from tqdm import tqdm
data = pd.read_csv("../input/train.csv")
def pre_process(text):

 

    text = text.translate(str.maketrans('', '', string.punctuation))

   

#     text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

   

    return text
y= data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
len(y)
x=  data['comment_text'].apply(pre_process)
x= x.tolist()
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=300)

vectorizer.fit(x)

X= vectorizer.transform(x)
Y= y.values
from skmultilearn.problem_transform import ClassifierChain

from sklearn.naive_bayes import MultinomialNB

classifier = ClassifierChain(MultinomialNB(alpha=0.7))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.08, random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)

# from skmultilearn.adapt import MLkNN

# classifier1 = MLkNN(k=6)

# classifier1.fit(X_train, y_train)

# predictions1 = classifier1.predict(X_test)





# accuracy_score(y_test,predictions1)

# from sklearn.ensemble import RandomForestClassifier

# randclf = RandomForestClassifier(n_estimators=100,random_state=2)
# randclf.fit(X_train, y_train)
# randclf.score(X_test, y_test)