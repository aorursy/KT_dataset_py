import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns



from PIL import Image

from wordcloud import WordCloud



import spacy

import nltk

from nltk.util import ngrams



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report, mean_absolute_error, r2_score

from imblearn.over_sampling import RandomOverSampler, SMOTE



import lime

from lime import lime_text

from lime.lime_text import LimeTextExplainer

from sklearn.pipeline import make_pipeline
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.shape
data.head()
# Looks like there are 16 objects with no description.

data[data['name'].isnull()]
# Fill NaN with an empty string and check missing values again

data['name'].fillna('', inplace=True)

data['name'].isnull().sum()
def remove_punct(line):

    return re.sub('[^A-Za-z]+', ' ', line).lower()



data['clean_name'] = data['name'].apply(remove_punct)
# Let's compare raw and cleaned texts.

data[['name', 'clean_name']]
nlp = spacy.load("en")

stopwords = nlp.Defaults.stop_words
def tokenize_no_stopwords(line):

    tokens = nltk.tokenize.word_tokenize(line)

    tokens_no_stop = [w for w in tokens if w not in stopwords]

    return " ".join(tokens_no_stop)
data['final_name'] = data['clean_name'].apply(tokenize_no_stopwords)
# Well, looks about right

data[['clean_name', 'final_name']]
# first, we need to concatenate all descriptions in one string

text = ""

for i in data['final_name']:

    text += " " + i

    

# next, we tokenize it into separate words

tokenized_text = nltk.tokenize.word_tokenize(text)



# finally, create a frequency dictionary with the help of nltk

freq_dict = nltk.FreqDist(w for w in tokenized_text)
def plot_most_common(dict_data ,title):

    df = pd.DataFrame(dict_data)

    df.columns = ['word', 'count']

    plt.figure(figsize=(8, 8))

    sns.set(style="darkgrid")

    sns.barplot(x="count", y="word", data=df, palette='twilight')

    plt.title(title)

    plt.show()

    

plot_most_common(freq_dict.most_common(20), 'Top 20 frequent words for NYC Airbnb titles')
freq_dict_bigrams = nltk.FreqDist(nltk.bigrams(w for w in tokenized_text))

plot_most_common(freq_dict_bigrams.most_common(20), 'Top 20 frequent bigrams for NYC Airbnb titles')
freq_dict_trigrams = nltk.FreqDist(nltk.trigrams(w for w in tokenized_text))

plot_most_common(freq_dict_trigrams.most_common(20), 'Top 20 frequent trigrams for NYC Airbnb titles')
# First, we are going to take a look at the price distribution

plt.figure(figsize=(8, 8))

sns.distplot(data['price'])

plt.show()
data['price'].describe()
costly = data[data['price']>1000]

costly.shape
# first, we need to concatenate all descriptions in one string

costly_text = ""

for i in costly['final_name']:

    costly_text += " " + i

    

# next, we tokenize it into separate words

tokenized_costly_text = nltk.tokenize.word_tokenize(costly_text)



# finally, create a frequency dictionary with the help of nltk

freq_dict_costly = nltk.FreqDist(w for w in tokenized_costly_text)
plot_most_common(freq_dict_costly.most_common(20), "Top 20 words in pricy apartments' titles")
freq_dict_bigrams_costly = nltk.FreqDist(nltk.bigrams(w for w in tokenized_costly_text))

plot_most_common(freq_dict_bigrams_costly.most_common(20), "Top 20 bigrams in pricy apartments' titles")
freq_dict_trigrams_costly = nltk.FreqDist(nltk.trigrams(w for w in tokenized_costly_text))

plot_most_common(freq_dict_trigrams_costly.most_common(20), "Top 20 trigrams in pricy apartments' titles")
def new_target(line):

    if line > 500:

        return 1

    else:

        return 0

        

data['target'] = data['price'].apply(new_target)

data['target'].value_counts()
train, test = train_test_split(data, test_size=0.2, random_state=315, stratify=data['target'])



X_train, y_train = train['final_name'], train['target']

X_test, y_test = test['final_name'], test['target']



X_train.shape, y_train.shape, X_test.shape, y_test.shape
vect = TfidfVectorizer()

X_train = vect.fit_transform(X_train)

X_test = vect.transform(X_test)
ros = RandomOverSampler(sampling_strategy='minority', random_state=1)



X_train_ros, y_train_ros = ros.fit_sample(X_train, y_train)

np.bincount(y_train_ros)
lr = LGBMClassifier(random_state=315)

lr.fit(X_train_ros, y_train_ros)

preds = lr.predict(X_test)
print(classification_report(y_test, preds))
def draw_cm(y_test, y_pred):

  cm = confusion_matrix(y_test, y_pred)

  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  df_cm = pd.DataFrame(cm_norm)

  plt.figure(figsize = (6,4))

  sns.heatmap(df_cm, annot=True, cmap="Blues")

  plt.xlabel("Predicted class")

  plt.ylabel("True class")

  print("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)))

  print("Recall: {0:.3f}".format(recall_score(y_test, y_pred)))

  plt.show()
draw_cm(y_test, preds)
# create a pipeline and a LIME's explainer object

c = make_pipeline(vect, lr)

class_names=['cheaper', 'expensive']

explainer = LimeTextExplainer(class_names=class_names)



# use them to explain an individual prediction



ind = 48050

exp = explainer.explain_instance(test.loc[ind]['final_name'], c.predict_proba, labels=[1])

exp.show_in_notebook(text=True)



ind_pred = c.predict_proba([test.loc[ind]['final_name']])



print("True class: {}".format(class_names[test.loc[ind]['target']]))

print("Predicted class: {}".format(class_names[np.argmax(ind_pred)]))
ind = 19427

exp = explainer.explain_instance(test.loc[ind]['final_name'], c.predict_proba, labels=[1])

exp.show_in_notebook(text=True)



ind_pred = c.predict_proba([test.loc[ind]['final_name']])



print("True class: {}".format(class_names[test.loc[ind]['target']]))

print("Predicted class: {}".format(class_names[np.argmax(ind_pred)]))