import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
path = '../input/imdb-large-movie-dataset/aclImdb/train/neg'

data2 = []

files = [path+'/'+f for f in os.listdir(path) if os.path.isfile(path+'/'+f)]

for f in files:

      with open (f, "r",encoding="utf8") as myfile:

               data2.append(myfile.read())

df_3=pd.DataFrame(data2,columns=['review'])

df_3['label']='neg'

path = '../input/imdb-large-movie-dataset/aclImdb/train/pos'

data3 = []

files = [path+'/'+f for f in os.listdir(path) if os.path.isfile(path+'/'+f)]

for f in files:

      with open (f, "r",encoding="utf8") as myfile:

               data3.append(myfile.read())

df_4=pd.DataFrame(data3,columns=['review'])

df_4['label']='pos'

train_df = pd.concat([df_3,df_4])

train_df.head()
path = '../input/imdb-large-movie-dataset/aclImdb/test/neg'

data = []

files = [path+'/'+f for f in os.listdir(path) if os.path.isfile(path+'/'+f)]

for f in files:

      with open (f, "r",encoding="utf8") as myfile:

               data.append(myfile.read())

df_1=pd.DataFrame(data,columns=['review'])

df_1['label']='neg'

path = '../input/imdb-large-movie-dataset/aclImdb/test/pos'

data1 = []

files = [path+'/'+f for f in os.listdir(path) if os.path.isfile(path+'/'+f)]

for f in files:

      with open (f, "r",encoding="utf8") as myfile:

               data1.append(myfile.read())

df_2=pd.DataFrame(data1,columns=['review'])

df_2['label']='pos'

df_2.head()

test_df = pd.concat([df_1,df_2])

test_df.head()
print(test_df.shape)

print(train_df.shape)
test_df['review'] = test_df['review'].str.lower().str.split()

train_df['review'] = train_df['review'].str.lower().str.split()
#####Remove stop words from the data

import nltk

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

test_df['review'] = test_df['review'].apply(lambda x: [word for word in x if word not in stop])

test_df.head()
train_df['review'] = train_df['review'].apply(lambda x: [word for word in x if word not in stop])

train_df.head()
###Removing punctuations, HTML tags (like br) etc

def remove_html_tags(review):

    """Remove html tags from a string"""

    import re

    clean = re.compile('<.*?>')

    return re.sub(clean, '', review)

test_df["review"] = test_df["review"].astype(str).map(remove_html_tags)

train_df['review'] = train_df['review'].astype(str).map(remove_html_tags)

train_df.head()
import string

test_df["review"] = test_df["review"].astype(str).apply(lambda x : x.translate(str.maketrans('','', string.punctuation)))

train_df["review"] = train_df["review"].astype(str).apply(lambda x : x.translate(str.maketrans('','', string.punctuation)))

train_df.head()
test_df["review"] = test_df["review"].astype(str).apply(lambda x : x.translate(str.maketrans('','', string.digits)))

train_df["review"] = train_df["review"].astype(str).apply(lambda x : x.translate(str.maketrans('','', string.digits)))

train_df.head()
####Apply Stemming and Lemmatization

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

test_df['review'] = test_df['review'].astype(str).str.split().apply(lambda x: ' '.join([ps.stem(word) for word in x]))

train_df['review'] = train_df['review'].astype(str).str.split().apply(lambda x: ' '.join([ps.stem(word) for word in x]))

train_df.head()
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

test_df['review'] = test_df['review'].astype(str).str.split().apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word) for word in x]))

train_df['review'] = train_df['review'].astype(str).str.split().apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word) for word in x]))

train_df.head()
####Apply feature selection to select most important words/features

import operator

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_Train = cv.fit_transform(train_df['review'])



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y_Train = le.fit_transform(train_df['label'])



from sklearn.ensemble import ExtraTreesClassifier

tree_clf = ExtraTreesClassifier()

tree_clf.fit(X_Train, Y_Train)



importances = tree_clf.feature_importances_

feature_names = cv.get_feature_names()

feature_imp_dict = dict(zip(feature_names, importances))

sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)

from sklearn.feature_selection import SelectFromModel



model = SelectFromModel(tree_clf, prefit=True)

X_Train_updated = model.transform(X_Train)

print('Total features count', X_Train.shape[1])

print('Selected features', X_Train_updated.shape[1])
df_freq = pd.concat([train_df, test_df], ignore_index = True)

df =  pd.DataFrame(df_freq)

df.fillna("",inplace = True)

df.head()
# Vectorizing negative reviews set

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

vect_pos = vect.fit_transform(df[df.label.isin(['neg'])].review)



# Visualising the high frequency words for negative set

df1 = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

df1.nlargest(10, 'frequency')
# Vectorizing positive reviews set

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

vect_pos = vect.fit_transform(df_freq[df_freq.iloc[:,1].isin(['pos'])].review)



# Visualising the high frequency words for positive set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

df_freq.nlargest(10, 'frequency')
########Discover the lowest frequency and highest frequency words

# Vectorizing complete review set

vect = CountVectorizer()

vect_pos = vect.fit_transform(df.review)



# Visualising the high and low frequency words for complete set

df2 = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

print(df2.nlargest(5, 'frequency'), sep='\n')

print(df2.nsmallest(5, 'frequency'), sep='\t')
### Read unlabeled data from respective folder (unsup) and store in unsup_df

path = '../input/imdb-large-movie-dataset/aclImdb/train/unsup'

data2 = []

files = [path+'/'+f for f in os.listdir(path) if os.path.isfile(path+'/'+f)]

for f in files:

      with open (f, "r",encoding="utf8") as myfile:

               data2.append(myfile.read())

unsup_df=pd.DataFrame(data2,columns=['review'])

unsup_df.head()
unsup_df['review'] = unsup_df['review'].str.lower().str.split()

import nltk

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

unsup_df['review'] = unsup_df['review'].apply(lambda x: [word for word in x if word not in stop])

def remove_html_tags(review):

    """Remove html tags from a string"""

    import re

    clean = re.compile('<.*?>')

    return re.sub(clean, '', review)

unsup_df['review'] = unsup_df['review'].astype(str).map(remove_html_tags)

import string

unsup_df['review'] = unsup_df['review'].astype(str).apply(lambda x : x.translate(str.maketrans('','', string.punctuation)))

unsup_df['review'] = unsup_df['review'].astype(str).apply(lambda x : x.translate(str.maketrans('','', string.digits)))

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

unsup_df['review'] = unsup_df['review'].astype(str).str.split().apply(lambda x: ' '.join([ps.stem(word) for word in x]))

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

unsup_df['review'] = unsup_df['review'].astype(str).str.split().apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word) for word in x]))

unsup_df.head()
#####Create a cluster to separate positive and negative words (bonus) using k-means algorithm

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(unsup_df['review'])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)
unsup_df['label'] = kmeans.labels_

unsup_df.head()
unsup_df["label"] = np.where(unsup_df["label"]==0, 'neg', 'pos')

unsup_df.tail()
####Create a word cloud with positive and negative words after cleansing

# Creating a list of train and test data to analyse

from wordcloud import WordCloud

imdb_list = df["review"][df.label.isin(['pos'])].unique().tolist()

imdb = " ".join(imdb_list)



# Create a word cloud for psitive words

imdb_wordcloud = WordCloud().generate(imdb)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
# Creating a list of train and test data to analyse

imdb_list = df["review"][df.label.isin(['neg'])].unique().tolist()

imdb = " ".join(imdb_list)



# Create a word cloud for negative words

imdb_wordcloud = WordCloud().generate(imdb)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
# Creating an object for Count vectorizer and fitting it to positive dataset

from sklearn.feature_extraction.text import CountVectorizer

hist_cv = CountVectorizer()

hist_pos = hist_cv.fit_transform(df[df.label.isin(['pos'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_pos.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data[0], bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Creating an object for Count vectorizer and fitting it to negative dataset

hist_cv = CountVectorizer()

hist_neg = hist_cv.fit_transform(df[df.label.isin(['neg'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_neg.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data, bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
####Repeat visualization step 1 & 2 after feature selection and note the impact (Bonus)

import operator

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(df['review'])



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y = le.fit_transform(df['label'])



from sklearn.ensemble import ExtraTreesClassifier

tree_clf = ExtraTreesClassifier()

tree_clf.fit(X, Y)



importances = tree_clf.feature_importances_

feature_names = cv.get_feature_names()

feature_imp_dict = dict(zip(feature_names, importances))

sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)

from sklearn.feature_selection import SelectFromModel



model = SelectFromModel(tree_clf, prefit=True)

X_updated = model.transform(X)

print('Total features count', X.shape[1])

print('Selected features', X_updated.shape[1])
# Creating a list of train and test data to analyse

from wordcloud import WordCloud

imdb_list = df["review"][df.label.isin(['pos'])].unique().tolist()

imdb = " ".join(imdb_list)



# Create a word cloud for psitive words

imdb_wordcloud = WordCloud().generate(imdb)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
# Creating a list of train and test data to analyse

from wordcloud import WordCloud

imdb_list = df['review'][df.label.isin(['neg'])].unique().tolist()

imdb = " ".join(imdb_list)



# Create a word cloud for negative words

imdb_wordcloud = WordCloud().generate(imdb)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
# Creating an object for Count vectorizer and fitting it to positive dataset

from sklearn.feature_extraction.text import CountVectorizer

hist_cv = CountVectorizer()

hist_pos = hist_cv.fit_transform(df[df.label.isin(['pos'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_pos.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data[0], bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Creating an object for Count vectorizer and fitting it to positive dataset

hist_cv = CountVectorizer()

hist_neg = hist_cv.fit_transform(df[df.label.isin(['neg'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_neg.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data, bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
df_algo = pd.concat([train_df,test_df], keys=['train', 'test'])

df_algo = df_algo.reset_index(col_level=1).drop(['level_1'], axis=1)

df_algo.fillna("",inplace = True)

df_algo.head()
###Phase 4 Hypothesis testing  and Feature Selection

import operator

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(df_algo['review'])



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y = le.fit_transform(df_algo['label'])



from sklearn.ensemble import ExtraTreesClassifier

tree_clf = ExtraTreesClassifier()

tree_clf.fit(X, Y)



importances = tree_clf.feature_importances_

feature_names = cv.get_feature_names()

feature_imp_dict = dict(zip(feature_names, importances))

sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)

from sklearn.feature_selection import SelectFromModel



model = SelectFromModel(tree_clf, prefit=True)

X_updated = model.transform(X)

print('Total features count', X.shape[1])

print('Selected features', X_updated.shape[1])
from sklearn.feature_extraction.text import CountVectorizer

# Vectorizing complete review set

vect = CountVectorizer()

vect_pos = vect.fit_transform(df_algo.review)



# Visualising the high and low frequency words for complete set

df1 = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

print(df1.nlargest(15, 'frequency').index)
###Hypothesis testing

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()

vect_n = vect.fit_transform(df_algo[df_algo.label.isin(['neg'])].review)

df_x = pd.DataFrame(vect_n.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T
vect_p = vect.fit_transform(df_algo[df_algo.label.isin(['pos'])].review)

df_y = pd.DataFrame(vect_p.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T
import scipy.stats

result = scipy.stats.ttest_ind(df_x, df_y, equal_var=False)

print (result)



# Reject or accept null hypothesis

def validation_null_hypothesis(p_value):

    if p_value <= 0.05:

        print ("There is a significant difference")

        return False

    else:

        print ("No significant difference")

        return True



validation_null_hypothesis(result[1])
from sklearn.feature_extraction.text import TfidfVectorizer

vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')

vect_algo.fit(df_algo.review)

Xf_train = vect_algo.transform(df_algo[df_algo['level_0'].isin(['train'])].review)

Xf_test = vect_algo.transform(df_algo[df_algo['level_0'].isin(['test'])].review)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

yf_train = le.fit_transform(df_algo[df_algo['level_0'].isin(['train'])].label)

yf_test = le.fit_transform(df_algo[df_algo['level_0'].isin(['test'])].label)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(Xf_train, yf_train, train_size = 0.75, random_state = 0)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_pred)

print(accuracy)



from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)

predictions = clf.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

print(accuracy)



from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

y_pred = tree.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print(accuracy)



from sklearn.cluster import KMeans

model = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)

model.fit(X_train)

prediction = model.predict(X_val)

accuracy = accuracy_score(y_val, prediction)

print(accuracy)

###Supervised Learning: Build a sentiment analysis model to predict positive and negative classes 



# Fit the logistic regression model to the object

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0)

clf.fit(Xf_train, yf_train)



# predict the outcome for testing data

predictions = clf.predict(Xf_test)

from sklearn.metrics import accuracy_score



# check the accuracy of the model

accuracy = accuracy_score(yf_test, predictions)

print("Observation: logistic regression model gives an accuracy of %.2f%% on the testing data" %(accuracy*100))
#Feature Selection for unsupervised data

from sklearn.feature_extraction.text import CountVectorizer

# Vectorizing unlabelled reviews set

vect = CountVectorizer(stop_words = 'english', analyzer='word')

vect_pos = vect.fit_transform(unsup_df.review)



# Creating a dataframe for the high frequency words for unlabelled reviews set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T



# Removing high frequency and low frequency data for more accuracy

word_list = df_freq.nlargest(100, 'frequency').index

word_list = word_list.append(df_freq.nsmallest(43750, 'frequency').index)



# Removing unwanted words based on word_list from unlabelled data

count = 0

for sentence in unsup_df['review']:

    sentence = [word for word in sentence.lower().split() if word not in word_list]

    sentence = ' '.join(sentence)

    unsup_df.loc[count, 'review'] = sentence

    count+=1
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(unsup_df.review)

Y = vectorizer.transform(df_algo[df_algo['level_0'].isin(['test'])].review)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df_algo[df_algo['level_0'].isin(['test'])].label)
###Unsupervised Learning: Build a clustering model consisting of 2 clusters based on positive and negative reviews

from sklearn.cluster import KMeans

true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

model.fit(X)



# Visualising the 2 clusters

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :10]:

        print(' %s' % terms[ind])
# Prediction for test set using Kmeans clusters

Y = vectorizer.transform(df_algo[df_algo['level_0'].isin(['test'])].review)

prediction = model.predict(Y)



# Actual results of test sets for comparison

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df_algo[df_algo['level_0'].isin(['test'])].label)

# check the accuracy of the model



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y, prediction)

if accuracy < 0.5:

    accuracy = 1 - accuracy

print("Observation: The unsupervised learning gives an accuracy of %.2f%% on the testing data" %(accuracy*100))
##Supervised Learning: Compare the performance of different machine learning models, at least 2 



#Fitting logistic regression to training set

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(Xf_train, yf_train)



predictions = clf.predict(Xf_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(yf_test, predictions)

print(accuracy)
##Random forest classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=5)

rf.fit(Xf_train, yf_train)

y_pred = rf.predict(Xf_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(yf_test, y_pred)

accuracy
####Unsupervised Learning: Compare the performance of different models, at least 2

from sklearn.cluster import KMeans

true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=1)

model.fit(X)

prediction = model.predict(Y)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y, prediction)

accuracy
###Hierarchical clustering

from sklearn.cluster import AgglomerativeClustering

k = 2

Wclustering = AgglomerativeClustering(n_clusters = k, linkage='ward')

Wclustering_pred = Wclustering.fit_predict(X[:500].todense())
##accuracy of hierarchical clustering

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y[:500], Wclustering.labels_)

accuracy
####Divide the data into 4 clusters to enable finding more classes and analye each clster.

# Creating a k-means object and fitting it to target variable

from sklearn.cluster import KMeans

true_k = 4

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=13)

model.fit(Xf_train)

 

# Visualising the clusters

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vect_algo.get_feature_names()

for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :12]:

        print(' %s' % terms[ind])
frame = pd.DataFrame(model.labels_,columns = ['cluster'])

frame['cluster'].value_counts() #no of review per cluster
####Active Learning: Cluster the training dataset and try and find the genre

# Creating a k-means object and fitting it to target variable

from sklearn.cluster import KMeans

true_k = 9

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=13)

model.fit(Xf_train)

 

# Visualising the clusters

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vect_algo.get_feature_names()

for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :12]:

        print(' %s' % terms[ind])
print('''From above we get the following genres:-

Cluster 0 - Television Series

Cluster 1 - Action, entertainment

Cluster 2 - comedy,fun,romantic

Cluster 3 - Dancing,singing,music

Cluster 4 - Cartoon, Animation, Disney

Cluster 5 - terribl

Cluster 6 - novel

Cluster 7 - horror

Cluster 8 - war, documentari''')
###verify with testing data

from sklearn.cluster import KMeans

true_k = 9

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=13)

model.fit(Xf_test)

 

# Visualising the clusters

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vect_algo.get_feature_names()

for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :12]:

        print(' %s' % terms[ind])