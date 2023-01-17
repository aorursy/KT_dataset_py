import numpy as np

import pandas as pd 

from sklearn import model_selection, preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer



np.random.seed(123) #for reprodicible results

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filename = '/kaggle/input/walmart-product-dataset-usa/walmart_com-ecommerce_product_details.csv'

data = pd.read_csv(filename)

data
def get_category(full_cat):

    if '|' in str(full_cat):

        return full_cat.split('|')[0]

    return 'NA'

    

data['General_Category'] = data.apply(lambda x: get_category(x['Category']), axis=1)



data['General_Category'].value_counts()
selected_categories = data['General_Category'].value_counts()[:10].index.tolist()

selected_categories
data = data[data['General_Category'].isin(selected_categories)]

data
data = data[~data['Description'].isnull()]

data
# split the dataset into training and test datasets 

train_x, test_x, train_y, test_y = model_selection.train_test_split(data['Description'], data['General_Category'])



# label encode the target variable, encode labels to 0, 1, 2

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

test_y = encoder.fit_transform(test_y)



categories_df = pd.DataFrame({"category": selected_categories}, index=encoder.transform(selected_categories))

categories_df
import nltk

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

stopwords
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words = stopwords,

                             max_features=5000, ngram_range=(1,2))

tfidf_vect.fit(data['Description'])



# output of tfidf transform is sparse matrix, which is not allow us to apply normal matrix calculation

# => we need to convert to normal matrix

xtrain_tfidf =  tfidf_vect.transform(train_x).toarray()

xtest_tfidf =  tfidf_vect.transform(test_x).toarray()



# Getting transformed training and testing dataset

print('Number of training documents: %s' %str(xtrain_tfidf.shape[0]))

print('Number of testing documents: %s' %str(xtest_tfidf.shape[0]))

print('Number of features of each document: %s' %str(xtrain_tfidf.shape[1]))

print('xtrain_tfidf shape: %s' %str(xtrain_tfidf.shape))

print('train_y shape: %s' %str(train_y.shape))

print('xtest_tfidf shape: %s' %str(xtest_tfidf.shape))

print('test_y shape: %s' %str(test_y.shape))
def top_tfidf_feats(row, features, top_n=10):

    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature', 'tfidf']

    return df



def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):

    ''' Return the top n features that on average are most important amongst documents in rows

        indentified by indices in grp_ids. '''

    if grp_ids:

        D = Xtr[grp_ids]

    else:

        D = Xtr



    D[D < min_tfidf] = 0

    tfidf_means = np.mean(D, axis=0)

    return top_tfidf_feats(tfidf_means, features, top_n)



def top_feats_by_topic(Xtr, y, topics_df, features, min_tfidf=0.1, top_n=10):

    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value

        calculated across documents with the same topic. '''

    dfs = pd.DataFrame(index = range(0,top_n))

    for i in topics_df.index:

        ids = np.where(y==i)

        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)

        dfs[topics_df.loc[i,][0]] = feats_df["feature"]

    return dfs
tfidf_vect.inverse_transform(xtrain_tfidf[13])
from sklearn import linear_model

from sklearn.metrics import accuracy_score



# train

logreg = linear_model.LogisticRegression(C=1e5, 

        solver = 'sag', multi_class = 'multinomial')

logreg.fit(xtrain_tfidf, train_y)



# test

y_pred = logreg.predict(xtest_tfidf)

print("Accuracy: %.2f %%" %(100*accuracy_score(test_y, y_pred.tolist())))
my_test = ["""Ginkgo Biloba Leaf Extract - Naturally Supports Brain, Nervous System and Memory

120 MG Extract Per Serving, 60 Servings, 60 Capsules - By Revive Herbs



Ginkgo Biloba Leaf Extract Naturally Supports Brain, Nervous System and Memory Made Using High Quality Ingredients Unconditional 90-Day Return Policy



REVIVE HERBS has been founded with one principle in mind -

we want to do our part in the revolution that is taking place. We

are referring to the revolution in the use of herbs and other natural

ingredients. Allopathic medicine has its place, but in many cases,

have side effects. Compared to that products with natural products

have been increasing shown in clinical studies to produce similar

or better results without the associated side effects. We care about

our customers, humanity and our earth. A part of our profits goes to

charities to help the under privileged lead a healthy and better life.



TAKE ACTION NOW: We encourage you to take control of your health and

give our product a chance. We promise we will do our best to support

you in your journey to a better health.""",

                

"""AmazonBasics Glass Electric Kettle

Have hot water ready in an instant with the AmazonBasics Glass Electric Kettle. This 1.7 liter, 1500 watt kettle quickly brings water to a boil, allowing you to make herbal tea, cocoa, French press coffee, and other hot beverages in a fraction of the time. Perfect for serving friends, family, or yourself, the kettle smoothly detaches from its heating base for cord-free convenience. Enjoy a hot beverage minus the fuss with this modern, space-saving glass kettle.""",

"""The Russell Athletic Men’s Essential Tee delivers the comfort, style, and performance to fit your active lifestyle. This t-shirt features our signature Dri-Power moisture wicking technology, odor protection to keep the fabric fresh, and a UPF 30+ rating to protect you from harmful UV rays. This tee is a wardrobe essential, offering both the style and comfort of cotton with the benefits of performance.

"""]



x_tfidf =  tfidf_vect.transform(my_test).toarray()



y_log = logreg.predict(x_tfidf)

print('prediction of logistic regression with SAG: {}'.format([categories_df.loc[i,][0] for i in y_log.tolist()]))

# train

logreg = linear_model.LogisticRegression(C=1e5, 

        solver = 'sag', multi_class = 'multinomial')

logreg.fit(xtrain_tfidf, train_y)



# test

y_pred = logreg.predict(xtest_tfidf)

print("Accuracy: %.2f %%" %(100*accuracy_score(test_y, y_pred.tolist())))
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17)

forest.fit(xtrain_tfidf, train_y)

# test

y_pred = forest.predict(xtest_tfidf)

print("Accuracy: %.2f %%" %(100*accuracy_score(test_y, y_pred.tolist())))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(xtrain_tfidf, train_y)

# test

y_pred = knn.predict(xtest_tfidf)

print("Accuracy: %.2f %%" %(100*accuracy_score(test_y, y_pred.tolist())))