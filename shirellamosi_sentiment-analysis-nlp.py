import pandas as pd

import numpy as np

import scipy

import re

import string



import seaborn as sns

import matplotlib.pyplot as plt

import scikitplot as skplt

from wordcloud import WordCloud





from sklearn.model_selection import train_test_split as split

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import lightgbm as lgb



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 

from nltk.stem import PorterStemmer, LancasterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



from textblob import TextBlob

import warnings

warnings.filterwarnings('ignore') 



from IPython.display import Image



%matplotlib inline
df = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv", index_col=0)

print(df.shape)

df.head(3)
df.groupby(['Rating', 'Recommended IND'])['Recommended IND'].count()
df.loc[(df.Rating==5) & (df['Recommended IND']==0)]['Review Text'].iloc[1]
text_df = df[['Title', 'Review Text', 'Recommended IND']]

text_df.head()
text_df['Review'] = text_df['Title'] + ' ' + text_df['Review Text']

text_df = text_df.drop(labels=['Title','Review Text'] , axis=1)

text_df.head()
text_df.Review.isna().sum()
text_df = text_df[~text_df.Review.isna()]

text_df = text_df.rename(columns={"Recommended IND": "Recommended"})

print("My data's shape is:", text_df.shape)

text_df.head()
text_df['Recommended'].unique()
text_df['Recommended'].value_counts(normalize=True)
text_df['Review_length'] = text_df['Review'].apply(len)

print(text_df.shape)

text_df.head()
text_df['Review_length'].describe()
sns.set(rc={'figure.figsize':(11,5)})

sns.distplot(text_df['Review_length'] ,hist=True, bins=100)
df_zero = text_df[text_df['Recommended']==0]

df_one = text_df[text_df['Recommended']==1]
sns.distplot(df_zero[['Review_length']] ,hist=False)

sns.distplot(df_one[['Review_length']], hist=False)
def count_exclamation_mark(string_text):

    count = 0

    for char in string_text:

        if char == '!':

            count += 1

    return count
text_df['count_exc'] = text_df['Review'].apply(count_exclamation_mark)

text_df.head(5)
text_df['count_exc'].describe(np.arange(0.2, 1.0, 0.2))
text_df['count_exc'].value_counts().sort_index().plot(kind='bar')
text_df[text_df['count_exc']== 41].index
text_df['Review'][3301]
text_df['Polarity'] = text_df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)

text_df.head(5)
text_df['Polarity'].plot(kind='hist', bins=100)
text_prep = text_df.copy()
string.punctuation
def punctuation_removal(messy_str):

    clean_list = [char for char in messy_str if char not in string.punctuation]

    clean_str = ''.join(clean_list)

    return clean_str
text_prep['Review'] = text_prep['Review'].apply(punctuation_removal)

text_prep['Review'].head()
Image(url= "http://josecarilloforum.com/imgs/longnounphrase_schematic-1B.png", width=600, height=10)
def adj_collector(review_string):

    new_string=[]

    review_string = word_tokenize(review_string)

    tup_word = nltk.pos_tag(review_string)

    for tup in tup_word:

        if 'VB' in tup[1] or tup[1]=='JJ':  #Verbs and Adjectives

            new_string.append(tup[0])  

    return ' '.join(new_string)
text_prep['Review'] = text_prep['Review'].apply(adj_collector)

text_prep['Review'].head(7)
print(stopwords.words('english')[::12])
stop = stopwords.words('english')

stop.append("i'm")
stop_words = []



for item in stop: 

    new_item = punctuation_removal(item)

    stop_words.append(new_item) 

print(stop_words[::12])
clothes_list =['dress', 'top','sweater','shirt',

               'skirt','material', 'white', 'black',

              'jeans', 'fabric', 'color','order', 'wear']
def stopwords_removal(messy_str):

    messy_str = word_tokenize(messy_str)

    return [word.lower() for word in messy_str 

            if word.lower() not in stop_words and word.lower() not in clothes_list ]
text_prep['Review'] = text_prep['Review'].apply(stopwords_removal)

text_prep['Review'].head()
print(text_prep['Review'][3301])



#'Beautiful and unique. Love this top, just received it today.

# \nit is a very artistic interpretation for a casual top.

# \nthe blue is gorgeous!

# \nthe unique style of the peplm and the details on the front set this apart!

# \nruns a little shorter, but i feel the length enhances it;s beauty, and is appropriate for the overall design.

# \nlove !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nordered my usual size and it fits perfectly.'
print(text_prep['Review'][267]) 
def drop_numbers(list_text):

    list_text_new = []

    for i in list_text:

        if not re.search('\d', i):

            list_text_new.append(i)

    return ' '.join(list_text_new)
text_prep['Review'] = text_prep['Review'].apply(drop_numbers)

text_prep['Review'].head()
print(text_prep['Review'][267]) 
print(text_prep['Review'][2293])
porter = PorterStemmer()
text_prep['Review'] = text_prep['Review'].apply(lambda x: x.split())

text_prep['Review'].head()
def stem_update(text_list):

    text_list_new = []

    for word in text_list:

        word = porter.stem(word)

        text_list_new.append(word) 

    return text_list_new
text_prep['Review'] = text_prep['Review'].apply(stem_update)

text_prep['Review'].head()
text_prep['Review'] = text_prep['Review'].apply(lambda x: ' '.join(x))

text_prep['Review'].head()
print(text_prep['Review'][2293])
pos_df = text_prep[text_prep.Recommended== 1]

neg_df = text_prep[text_prep.Recommended== 0]

pos_df.head(3)
pos_words =[]

neg_words = []



for review in pos_df.Review:

    pos_words.append(review) 

pos_words = ' '.join(pos_words)

pos_words[:40]



for review in neg_df.Review:

    neg_words.append(review)

neg_words = ' '.join(neg_words)

neg_words[:400]
wordcloud = WordCloud().generate(pos_words)



wordcloud = WordCloud(background_color="white",max_words=len(pos_words),\

                      max_font_size=40, relative_scaling=.5, colormap='summer').generate(pos_words)

plt.figure(figsize=(13,13))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(neg_words)



wordcloud = WordCloud(background_color="white",max_words=len(neg_words),\

                      max_font_size=40, relative_scaling=.5, colormap='gist_heat').generate(neg_words)

plt.figure(figsize=(13,13))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
text_prep['Review'].head()
def text_vectorizing_process(sentence_string):

    return [word for word in sentence_string.split()]
bow_transformer = CountVectorizer(text_vectorizing_process)
bow_transformer.fit(text_prep['Review'])
print(text_prep['Review'].iloc[3])
example = bow_transformer.transform([text_prep['Review'].iloc[3]])

print(example)

#3507=Love

#4438=petit
Reviews = bow_transformer.transform(text_prep['Review'])

Reviews
print('Shape of Sparse Matrix', Reviews.shape)

print('Amount of Non-Zero occurences:', Reviews.nnz)
tfidf_transformer = TfidfTransformer().fit(Reviews)



tfidf_example = tfidf_transformer.transform(example)

print (tfidf_example)

#3507=Love

#4438=petit
[i for i in bow_transformer.vocabulary_.items() if i[1]==3507]
[i for i in bow_transformer.vocabulary_.items()][6:60:10]
messages_tfidf = tfidf_transformer.transform(Reviews)

messages_tfidf.shape
print(messages_tfidf[:1]) 

#tuple(index_num, word_num), tfidf_proba
messages_tfidf = messages_tfidf.toarray()

messages_tfidf = pd.DataFrame(messages_tfidf)

print(messages_tfidf.shape)

messages_tfidf.head()
df_all = pd.merge(text_prep.drop(columns='Review'),messages_tfidf, 

                  left_index=True, right_index=True )

df_all.head()
X = df_all.drop('Recommended', axis=1)

y = df_all.Recommended



X.head()
X.shape
X.describe()
X_train, X_test, y_train, y_test = split(X,y, test_size=0.3, stratify=y, random_state=111)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
pd.DataFrame(X_train_scaled,columns= X_train.columns).describe()
pca_transformer = PCA(n_components=2).fit(X_train_scaled)

X_train_scaled_pca = pca_transformer.transform(X_train_scaled)

X_test_scaled_pca = pca_transformer.transform(X_test_scaled)

X_train_scaled_pca[:1]
plt.figure(figsize=(15,7))

sns.scatterplot(x=X_train_scaled_pca[:, 0], 

                y=X_train_scaled_pca[:, 1], 

                hue=y_train, 

                sizes=100,

                palette="inferno") 
X_train_scaled = scipy.sparse.csr_matrix(X_train_scaled)

X_test_scaled = scipy.sparse.csr_matrix(X_test_scaled)



X_train = scipy.sparse.csr_matrix(X_train.values)

X_test = scipy.sparse.csr_matrix(X_test.values)

X_test
def report(y_true, y_pred, labels):

    cm = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred), 

                                        index=labels, columns=labels)

    rep = classification_report(y_true=y_true, y_pred=y_pred)

    return (f'Confusion Matrix:\n{cm}\n\nClassification Report:\n{rep}')
svc_model = SVC(C=1.0, 

             kernel='linear',

             class_weight='balanced', 

             probability=True,

             random_state=111)

svc_model.fit(X_train_scaled, y_train)
test_predictions = svc_model.predict(X_test_scaled)

print(report(y_test, test_predictions, svc_model.classes_ ))
skplt.metrics.plot_roc(y_test, svc_model.predict_proba(X_test_scaled)) 
lr_model = LogisticRegression(class_weight='balanced', 

                              random_state=111, 

                              solver='lbfgs',

                              C=1.0)



gs_lr_model = GridSearchCV(lr_model, 

                           param_grid={'C': [0.01, 0.1, 0.5, 1.0, 5.0]}, 

                           cv=5, 

                           scoring='roc_auc')



gs_lr_model.fit(X_train_scaled, y_train)
gs_lr_model.best_params_
test_predictions = gs_lr_model.predict(X_test_scaled)

print(report(y_test, test_predictions, gs_lr_model.classes_ ))
skplt.metrics.plot_roc(y_test, gs_lr_model.predict_proba(X_test_scaled),

                      title='ROC Curves - Logistic Regression') 
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=555)



ada_model = AdaBoostClassifier(base_estimator=dt, learning_rate=0.001, n_estimators=1000, random_state=222)

ada_model.fit(X_train ,y_train)
test_predictions = ada_model.predict(X_test)

print(report(y_test, test_predictions, ada_model.classes_ ))
skplt.metrics.plot_roc(y_test, ada_model.predict_proba(X_test), 

                       title='ROC Curves - AdaBoost') 
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=5, 

                                  class_weight='balanced', random_state=3)

rf_model.fit(X_train, y_train)
test_predictions = rf_model.predict(X_test)

print(report(y_test, test_predictions, rf_model.classes_ ))
skplt.metrics.plot_roc(y_test, rf_model.predict_proba(X_test), 

                       title='ROC Curves - Random Forest') 
my_list = list(zip(rf_model.feature_importances_ ,X.columns))

my_list.sort(key=lambda tup: tup[0],reverse=True)

my_list[:7]
bow_list = [i for i in bow_transformer.vocabulary_.items()]



for i in my_list:

    for j in bow_list:

        if i[1] == j[1] and i[0]> 0.005:

            print(f'Importance: {i[0]:.4f}   Word num: {i[1]}   Word:  { j[0]}')
probs = rf_model.predict_proba(X_train)

fpr, tpr, thresholds = metrics.roc_curve(y_train, probs[:,1])
#Train

plt.subplots(figsize=(10, 6))

plt.plot(fpr, tpr, '-', label="ROC curve")

plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label="diagonal")

for x, y, txt in zip(fpr[::100], tpr[::100], thresholds[::100]):

    plt.annotate(np.round(txt,3), (x, y-0.03), fontsize='x-small')

rnd_idx = 700

plt.annotate('this point refers to the tpr and the fpr\n at a probability threshold of {}'\

             .format(np.round(thresholds[rnd_idx], 4)), 

             xy=(fpr[rnd_idx], tpr[rnd_idx]), xytext=(fpr[rnd_idx]+0.2, tpr[rnd_idx]-0.25),

             arrowprops=dict(facecolor='black', lw=2, arrowstyle='->',color='r'),)

plt.legend(loc="upper left")

plt.xlabel("FPR")

plt.ylabel("TPR")
probs = rf_model.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:,1])
#Test

plt.subplots(figsize=(10, 6))

plt.plot(fpr, tpr, '-', label="ROC curve")

plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label="diagonal")

for x, y, txt in zip(fpr[::70], tpr[::70], thresholds[::70]):

    plt.annotate(np.round(txt,4), (x, y-0.01))



plt.legend(loc="upper left")

plt.xlabel("FPR")

plt.ylabel("TPR")
X_train = pd.DataFrame(X_train.toarray(), columns=X.columns)

X_train.head()
X_test = pd.DataFrame(X_test.toarray(), columns=X.columns)

X_test.head()
rf_model.classes_
arr= rf_model.predict_proba(X_test)

print(arr)
arr_list = arr.tolist()
arr_list[1][1]
proba_list = []

for i in arr_list:

    proba_list.append(i[0])

proba_list[:5]
X_test['Proba0'] = proba_list

X_test.head()
prediction_list = []

for i in X_test['Proba0']:

    if i > 0.4998:

        prediction_list.append(0)

    else:

        prediction_list.append(1)

prediction_list[:5]
X_test['Predictions'] = prediction_list

X_test.head()
print(report(y_test, X_test['Predictions'], rf_model.classes_))