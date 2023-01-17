# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# libraries for dataset preparation, feature engineering, model training
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn import model_selection, preprocessing, metrics
from sklearn import decomposition, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk.corpus import stopwords

# Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

# Configuring visualizations
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
df_train = pd.read_csv("../input/drugsComTrain_raw.csv")
df_train.info()
df_test = pd.read_csv("../input/drugsComTest_raw.csv")
df_test.info()
print("Total conditions in train data:", df_train.condition.nunique())
print("Total conditions in test data:", df_test.condition.nunique())
print("Conditions in test but not in train data:", len(set(df_test.condition).difference(set(df_train.condition))))
print("Total drugs in train data:", df_train.drugName.nunique())
print("Total drugs in test data:", df_test.drugName.nunique())
print("Drugs in test but not in train data:", len(set(df_test.drugName).difference(set(df_train.drugName))))
df_train.drugName.value_counts().plot(kind="hist", 
                                      logy=True, 
                                      title="Histogram of drug counts (Log Frequency)")
plt.show()
df_train.condition.value_counts().plot(kind="hist", 
                                      logy=True, 
                                      title="Histogram of condition counts (Log Frequency)")
plt.show()
df_train.rating.plot(kind="hist",
                     title="Histogram of ratings")
plt.show()
rating_to_categ = {1:'Low',2:'Low',3:'Low',4:'Medium',5:'Medium',6:'Medium',7:'Medium', 8:'Top',9:'Top', 10:'Top'}

df_train['category'] = df_train['rating'].map(rating_to_categ)
df_test['category'] = df_train['rating'].map(rating_to_categ)
X_train = df_train.review
y_train = df_train.category
X_test = df_test.review
y_test = df_test.category
# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Define additional stopwords in a string
additional_stopwords = """I&#039;ve 039"""

# Split the the additional stopwords string on each word and then add
# those words to the NLTK stopwords list
stoplist += additional_stopwords.split()
# count level tf-idf 
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]{1,}', 
                             ngram_range=(1,2), stop_words=stoplist, 
                             min_df=1, max_features=5000)
vectorizer.fit(X_train)

x_vectors = vectorizer.transform(X_train)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver="adam", activation="logistic", alpha=1e-5, hidden_layer_sizes=(100, 20),
                    random_state=42)

print("Score: {:.4f}".format(cross_val_score(mlp, x_vectors, y_train, cv=3).mean()))
print(metrics.classification_report(y_test, mlp.predict(X_test)))