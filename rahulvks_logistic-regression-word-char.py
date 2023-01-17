import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import pandas as pd
import re
import plotly as py
import plotly.graph_objs as go
%matplotlib inline
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
#import lightgbm as lgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import tqdm
import os
%matplotlib inline
import collections
import spacy #load spacy
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
from nltk.corpus import stopwords
stops = stopwords.words("english")
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")

train_data['ciphertext'] = train_data['ciphertext'].str.lower()
train_data['ciphertext'] = train_data['ciphertext'].astype(str)

test_data['ciphertext'] = test_data['ciphertext'].str.lower()
test_data['ciphertext'] = test_data['ciphertext'].astype(str)

X = train_data[['ciphertext']]
Y = train_data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.25)
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
# build TFIDF Vectorizer

tokens= ((u'(?ui)\\b\\w*[a-z]+\\w*\\b'))

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #stop_words = 'english',
    strip_accents='ascii',
    analyzer='word',
    token_pattern=tokens,
    ngram_range=(1,2),
    dtype=np.float32,
    max_features=7500
)


# Character Stemmer
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #stop_words = 'english',
    strip_accents='ascii',
    analyzer='char',
    token_pattern=tokens,
    ngram_range=(2, 4),
    dtype=np.float32,
    max_features=12000
)

word_vectorizer.fit(train_data['ciphertext'])

char_vectorizer.fit(train_data['ciphertext'])


train_word_features = word_vectorizer.transform(train_data['ciphertext'])
train_char_features = char_vectorizer.transform(train_data['ciphertext'])

train_features = hstack([
    train_char_features,
    train_word_features])
Target = train_data["target"]

%time
print("Modeling..")
loss = []
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, test_size=0.33)

lr = LogisticRegression(solver="sag", max_iter=100,class_weight='balanced',C=2.65,penalty='l2')
lr.fit(train_features,Target)
lr_pred=lr.predict(X_test_tfidf)

accuracy_tfidf =accuracy_score(y_test_tfidf,lr_pred)

print(accuracy_tfidf)
print("Auc Score: ",np.mean(cross_val_score(lr, train_features, Target, cv=3,)))

print(classification_report(y_test_tfidf,lr_pred))
# test
test_word_features = word_vectorizer.transform(test_data['ciphertext'])
test_char_features = char_vectorizer.transform(test_data['ciphertext'])

test_features = hstack([
    test_char_features,
    test_word_features])


lr = LogisticRegression(solver="sag", max_iter=100,class_weight='balanced',C=2.65,penalty='l2')
lr.fit(train_features,Target)
Predicted=lr.predict(test_features)

df_submission['Predicted'] = final_prediction
df_submission.to_csv('submission.csv', index = False)
