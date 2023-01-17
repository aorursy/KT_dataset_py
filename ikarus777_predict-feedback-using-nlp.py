import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

TARGET = 'feedback'
NFOLDS = 5
SEED = 8
NROWS = None
DATA_DIR = '../input'

DATA_FILE = f'{DATA_DIR}/amazon_alexa.tsv'
data = pd.read_csv(DATA_FILE, sep='\t')
print(data.shape)
data.head()
list(data['feedback'].unique())
list(data['rating'].unique())
data.isnull().sum()
rating = data['rating']
bins = np.arange(7) - 0.5
plt.hist(rating, bins=bins, rwidth=0.85)
plt.xticks(range(6))
plt.ylabel('Reviews')
plt.xlabel('Rating')
plt.title('Amazon rating distribution')
plt.xlim([0, 6])
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def cleanText(text):
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    porter = PorterStemmer()
    
    text = text.lower()
    words = text.split()
    words = [w.translate(table) for w in words]
    words = [porter.stem(w) for w in words if not w in stop_words]
    return ' '.join(words)

data['verified_reviews'] = data['verified_reviews'].apply(cleanText)
data['verified_reviews'].head(10)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1000)
count_vect = cv.fit_transform(data['verified_reviews'])
X = pd.DataFrame(count_vect.todense(), columns=cv.get_feature_names())
y = data['feedback']

# Merge multiple features with the same name
X = X.groupby(X.columns, axis=1).agg('mean')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
print(confusion_matrix(y_test, y_pred))