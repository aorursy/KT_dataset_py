import pandas as pd
# Let's first import the data
messages = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding = "ISO-8859-1")
messages
messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
messages
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
  review = re.sub('[^a-zA-Z]', ' ', messages['v2'][i])
  review = review.lower()
  review = review.split()

  review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features = 2500)
x = tf.fit_transform(corpus).toarray()
y = pd.DataFrame(messages['v1'])
y
## now for converting all string values into categorial values
for labels, contents in y.items():
    if pd.api.types.is_string_dtype(contents):
        y['v1'] = contents.astype('category').cat.as_ordered()
y.v1.cat.codes
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x , y, test_size = 0.2, random_state = 2)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(xtrain, ytrain)
ypreds = spam_detect_model.predict(xtest)
ypreds
ytest
from sklearn.metrics import classification_report
classification_report(ytest, ypreds)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypreds)
cm
from sklearn.metrics import accuracy_score
ac_s = accuracy_score(ytest, ypreds)
ac_s
