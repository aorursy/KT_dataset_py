# load packages

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer

from sklearn import decomposition, ensemble

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns

import pandas, xgboost, numpy, textblob, string

import pandas as pd
# load the dataset

trainDF = pd.read_csv('../input/bbc-text.csv') # encoding = "latin"
trainDF.head(10)
trainDF.shape
trainDF['category'].unique()
trainDF['category'].value_counts()
sns.set(rc={'figure.figsize':(10,10)})

sns.countplot(trainDF['category'])
# split the dataset into training and validation datasets 

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['category'])



train_labels = train_y

valid_labels = valid_y

# label encode the target variable 

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
# Count Vectors as features

# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(trainDF['text'])



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(train_x)

xvalid_count =  count_vect.transform(valid_x)
# plot the train features

pca = PCA(n_components=2).fit(xtrain_count.toarray())

data2D = pca.transform(xtrain_count.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")
# plot the validation features

pca = PCA(n_components=2).fit(xvalid_count.toarray())

data2D = pca.transform(xvalid_count.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(trainDF['text'])

xtrain_tfidf =  tfidf_vect.transform(train_x)

xvalid_tfidf =  tfidf_vect.transform(valid_x)
# plot the train features

pca = PCA(n_components=2).fit(xtrain_tfidf.toarray())

data2D = pca.transform(xtrain_tfidf.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")
# plot the validation features

pca = PCA(n_components=2).fit(xvalid_tfidf.toarray())

data2D = pca.transform(xvalid_tfidf.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(trainDF['text'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
# plot the train features

pca = PCA(n_components=2).fit(xtrain_tfidf_ngram.toarray())

data2D = pca.transform(xtrain_tfidf_ngram.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")
# plot the validation features

pca = PCA(n_components=2).fit(xvalid_tfidf_ngram.toarray())

data2D = pca.transform(xvalid_tfidf_ngram.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['text'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
# plot the train features

pca = PCA(n_components=2).fit(xtrain_tfidf_ngram_chars.toarray())

data2D = pca.transform(xtrain_tfidf_ngram_chars.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")
# plot the validation features

pca = PCA(n_components=2).fit(xvalid_tfidf_ngram_chars.toarray())

data2D = pca.transform(xvalid_tfidf_ngram_chars.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")
# getting train features

hash_vectorizer = HashingVectorizer(n_features=5000)

hash_vectorizer.fit(trainDF['text'])

xtrain_hash_vectorizer =  hash_vectorizer.transform(train_x) 

xvalid_hash_vectorizer =  hash_vectorizer.transform(valid_x)
# plot the train features

pca = PCA(n_components=2).fit(xtrain_hash_vectorizer.toarray())

data2D = pca.transform(xtrain_hash_vectorizer.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")
# plot the validation features

pca = PCA(n_components=2).fit(xvalid_hash_vectorizer.toarray())

data2D = pca.transform(xvalid_hash_vectorizer.toarray())

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

ax = sns.scatterplot(data2D[:,0], data2D[:,1],

hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    

    return metrics.accuracy_score(predictions, valid_y)
# Naive Bayes on Count Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)

print("NB, Count Vectors: ", accuracy)



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

print("NB, WordLevel TF-IDF: ", accuracy)



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print("NB, N-Gram Vectors: ", accuracy)



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print("NB, CharLevel Vectors: ", accuracy)
# Linear Classifier on Count Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_count, train_y, xvalid_count)

print("LR, Count Vectors: ", accuracy)



# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_tfidf, train_y, xvalid_tfidf)

print("LR, WordLevel TF-IDF: ", accuracy)



# Linear Classifier on Ngram Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print("LR, N-Gram Vectors: ", accuracy)



# Linear Classifier on Character Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print("LR, CharLevel Vectors: ", accuracy)



# Linear Classifier on Hash Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_hash_vectorizer, train_y, xvalid_hash_vectorizer)

print("LR, Hash Vectors: ", accuracy)
# RF on Count Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_count, train_y, xvalid_count)

print("RF, Count Vectors: ", accuracy)



# RF on Word Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf, train_y, xvalid_tfidf)

print("RF, WordLevel TF-IDF: ", accuracy)



# RF on Ngram Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print("RF, N-Gram Vectors: ", accuracy)



# RF on Character Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print("RF, CharLevel Vectors: ", accuracy)



# RF on Hash Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_hash_vectorizer, train_y, xvalid_hash_vectorizer)

print("RF, Hash Vectors: ", accuracy)
# Extreme Gradient Boosting on Count Vector

accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())

print("Xgb, Count Vectors: ", accuracy)



# Extreme Gradient Boosting on Word Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())

print("Xgb, WordLevel TF-IDF: ", accuracy)



# Extreme Gradient Boosting on Ngram Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print("Xgb, N-Gram Vectors: ", accuracy)



# Extreme Gradient Boosting on Character Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())

print("Xgb, CharLevel Vectors: ", accuracy)



# Extreme Gradient Boosting on Hash Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_hash_vectorizer, train_y, xvalid_hash_vectorizer)

print("Xgb, Hash Vectors: ", accuracy)