# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Set up matplotlib style 

plt.style.use('ggplot')



# Libraries for wordcloud making and image importing

from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image



# And libraries for data transformation

import datetime

from string import punctuation
# Import data and transform tsv file

data = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter='\t')
# Data overlook

data.head()
# Transform string data and remove punctuation

data['verified_reviews'] = data.verified_reviews.apply(lambda x: x.lower())

data['verified_reviews'] = data.verified_reviews.apply(lambda x: ''.join([c for c in x if c not in punctuation]))
# Get length of review for EDA

data['review_length'] = data.verified_reviews.apply(lambda x: len(x))
# Check the data again

data.head()
# Take a look at the mean, standard deviation, and maximum

print('The mean for the length of review:',data['review_length'].mean())

print('The standard deviation for the length of reviews:',data['review_length'].std())

print('The maximum for the length of reviews:',data['review_length'].max())
# And take a look at the distribution of the length

data['review_length'].hist(bins=20)

plt.title('Distribution of review length')
# Transform date to datetime data type

data['date'] = data.date.apply(lambda x:  datetime.datetime.strptime(x, '%d-%b-%y'))
data.info()
A = np.array(Image.open('../input/amazon-logo/amazon-logo-white.png'))
np.random.seed(321)

sns.set(rc={'figure.figsize':(14,8)})

reviews = ' '.join(data['verified_reviews'].tolist())



wordcloud = WordCloud(mask=A,background_color="black").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Reviews',size=20)

plt.show()
sns.set(rc={'figure.figsize':(10,6)})

sns.countplot(data.variation,

              order = data['variation'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Counts of each variation')
data.rating.value_counts()
data5 = data[data.rating == 5]

data_not_5 = data[data.rating != 5]

data1 = data[data.rating == 1]
sns.set(rc={'figure.figsize':(14,8)})

reviews = ' '.join(data5['verified_reviews'].tolist())



wordcloud = WordCloud(mask=A,background_color="black").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Reviews of rating 5',size=20)

plt.show()
pd.options.display.max_colwidth = 200



data5[data5['verified_reviews'].str.contains('prime')]['verified_reviews'][:3]
data5[data5['verified_reviews'].str.contains('time ')]['verified_reviews'][:3]
data5[data5['verified_reviews'].str.contains('easy')]['verified_reviews'][:3]
sns.set(rc={'figure.figsize':(14,8)})

reviews = ' '.join(data1['verified_reviews'].tolist())



wordcloud = WordCloud(mask=A,background_color="black").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Reviews of rating 1',size=20)

plt.show()
data1[data1['verified_reviews'].str.contains('useless')]['verified_reviews'][:3]
reviews = ' '.join(data_not_5['verified_reviews'].tolist())



wordcloud = WordCloud(mask=A,background_color="black").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Reviews of negative reviews',size=20)

plt.show()
sns.boxplot(data.variation, data.rating)

plt.xticks(rotation = 90)
data.head()
data_date = data.groupby('date').count()
data_date.rating.plot()
sns.boxplot('rating','review_length',data=data)
data['log_review_length'] = data.review_length.apply(lambda x: (np.log(x)+1))
sns.boxplot('rating','log_review_length',data=data)
sns.boxplot('variation','log_review_length',data=data)

plt.xticks(rotation = 90)
from collections import Counter



text = ' '.join(data['verified_reviews'].tolist())

review_word = text.split(' ')

all_reviews = ' '.join(review_word)

words = all_reviews.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



reviews_ints = []

for review in review_word:

    reviews_ints.append([vocab_to_int[word] for word in review.split()])
print('Unique words: ', len((vocab_to_int)))
counts.most_common(20)
data1['rating'].value_counts()
text1 = ' '.join(data1['verified_reviews'].tolist())

review_word1 = text1.split(' ')

all_reviews1 = ' '.join(review_word1)

words1 = all_reviews1.split()



# words wrong datatype

counts1 = Counter(words1)

vocab1 = sorted(counts1, key=counts1.get, reverse=True)

vocab_to_int1 = {word: ii for ii, word in enumerate(vocab1, 1)}



reviews_ints1 = []

for review in review_word1:

    reviews_ints1.append([vocab_to_int1[word] for word in review.split()])
counts1.most_common(20)
data1 = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter='\t')
import spacy

nlp = spacy.load('en')



def explain_text_entities(text):

    doc = nlp(text)

    for ent in doc.ents:

        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
for i in range(15,20):

    one_sentence = data1['verified_reviews'][i]

    doc = nlp(one_sentence)

    spacy.displacy.render(doc, style='ent',jupyter=True)
for i in ['en_core_web_sm','en_core_web_lg']:

    nlp = spacy.load(i)

    print('This is model:',i)

    for i in range(17,22):

        one_sentence = data1['verified_reviews'][i]

        doc = nlp(one_sentence)

        spacy.displacy.render(doc, style='ent',jupyter=True)
data.head()
data['positive'] = 0

data.loc[data['rating'] ==5, 'positive'] = 1



y = data['positive']
from nltk.tokenize import word_tokenize 



word_tokenize(data.verified_reviews[0])
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



stop_words = set(stopwords.words('english')) 



data['cleaned_reviews'] = data.verified_reviews.apply(lambda x: word_tokenize(x))



data['cleaned_reviews'] = data.cleaned_reviews.apply(lambda x: [w for w in x if w not in stop_words])



data['cleaned_reviews'] = data.cleaned_reviews.apply(lambda x: ' '.join(x))
data.head()
data.info()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X_train, X_test, y_train, y_test = train_test_split(data["cleaned_reviews"], y, test_size=0.33

                                    ,random_state=53)



# Initialize a CountVectorizer object: count_vectorizer

count_vectorizer = CountVectorizer(stop_words="english")



# Transform the training data using only the 'text' column values: count_train 

count_train = count_vectorizer.fit_transform(X_train)



y_train = np.asarray(y_train.values)



ch2 = SelectKBest(chi2, k = 300)



X_new = ch2.fit_transform(count_train, y_train)



# Transform the test data using only the 'text' column values: count_test 

count_test = count_vectorizer.transform(X_test)



X_test_new = ch2.transform(X=count_test)
from sklearn.feature_extraction.text import TfidfVectorizer



# Initialize a TfidfVectorizer object: tfidf_vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)



# Transform the training data: tfidf_train 

tfidf_train = tfidf_vectorizer.fit_transform(X_train)



# Transform the test data: tfidf_test 

tfidf_test = tfidf_vectorizer.transform(X_test)
# Create the CountVectorizer DataFrame: count_df

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())



# Create the TfidfVectorizer DataFrame: tfidf_df

tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())



# Print the head of count_df

print(count_df.head())



# Print the head of tfidf_df

print(tfidf_df.head())



# Calculate the difference in columns: difference

difference = set(count_df.columns) - set(tfidf_df.columns)

print(difference)



# Check whether the DataFrames are equal

print(count_df.equals(tfidf_df))

# Import the necessary modules

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB





# Instantiate a Multinomial Naive Bayes classifier: nb_classifier

nb_classifier = MultinomialNB()



# Fit the classifier to the training data

nb_classifier.fit(X_new, y_train)



# Create the predicted tags: pred

pred = nb_classifier.predict(X_test_new)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)
sns.heatmap(metrics.confusion_matrix(pred,y_test),annot=True,fmt='2.0f')
# Create a Multinomial Naive Bayes classifier: nb_classifier

nb_classifier = MultinomialNB()



# Fit the classifier to the training data

nb_classifier.fit(tfidf_train,y_train)



# Create the predicted tags: pred

pred = nb_classifier.predict(tfidf_test)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)
sns.heatmap(metrics.confusion_matrix(pred,y_test),annot=True,fmt='2.0f')
from sklearn.feature_extraction.text import TfidfTransformer

def TextPreprocessing(train,train_y,test):

    

    CountVectorizer(stop_words="english")

    count = count_vectorizer.fit_transform(train)

    count_test = count_vectorizer.transform(X_test)

    

    tfidf_vectorizer = TfidfTransformer()

    tfidf = tfidf_vectorizer.fit_transform(count)

    tfidf_test = tfidf_vectorizer.transform(count_test)

    

    ch2 = SelectKBest(chi2, k = 300)

    

    train_new = ch2.fit_transform(tfidf, y_train)

    test_new = ch2.transform(tfidf_test)

    

    return train_new, test_new
train, test = TextPreprocessing(X_train,y_train,X_test)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=51)

# Fit the classifier to the training data

clf.fit(train, y_train)



# Create the predicted tags: pred

pred = clf.predict(test)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=51)

# Fit the classifier to the training data

clf.fit(X_new, y_train)



# Create the predicted tags: pred

pred = clf.predict(X_test_new)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)

sns.heatmap(metrics.confusion_matrix(pred,y_test),annot=True,fmt='2.0f')
clf = RandomForestClassifier(random_state=51)

# Fit the classifier to the training data

clf.fit(tfidf_train, y_train)



# Create the predicted tags: pred

pred = clf.predict(tfidf_test)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)

sns.heatmap(metrics.confusion_matrix(pred,y_test),annot=True,fmt='2.0f')
from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier()



scorer = metrics.make_scorer(metrics.fbeta_score, beta=0.5)



parameters = {'n_estimators': [150, 180, 250], 'max_features': [120,150], 'max_depth': [120,135,150],

              'min_samples_split':[3,5],'min_samples_leaf':[1,3,5]}



grid_obj = GridSearchCV(clf, parameters, scoring=scorer)



grid_fit = grid_obj.fit(X_new, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



best_predictions = best_clf.predict(X_test_new)



score = metrics.accuracy_score(y_test, best_predictions)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, best_predictions)

print('F score is:',f1)
best_clf
sns.heatmap(metrics.confusion_matrix(pred,best_predictions),annot=True,fmt='2.0f')
from itertools import compress



features = count_vectorizer.get_feature_names()

mask = ch2.get_support()

features = list(compress(features, mask))

importances = best_clf.feature_importances_

indices = np.argsort(importances)



sns.set(rc={'figure.figsize':(11,50)})



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
import eli5

from eli5.sklearn import PermutationImportance



X_test_new = X_test_new.toarray()



perm = PermutationImportance(best_clf, random_state=1).fit(X_test_new, y_test)

eli5.show_weights(perm, feature_names = features)
best_clf.fit(count_train,y_train)
eli5.show_prediction(best_clf, doc=X_train[20], vec=count_vectorizer)
from sklearn.tree import DecisionTreeClassifier



tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_new, y_train)
from sklearn import tree

import graphviz



tree_graph = tree.export_graphviz(tree_model, feature_names=features)

graphviz.Source(tree_graph)
df = pd.DataFrame(X_test_new,columns = features)

df.head()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=df, model_features=features, feature='love')



# plot it

pdp.pdp_plot(pdp_goals, 'love')

plt.show()
feature_to_plot = 'time'

pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=df, model_features=features, feature=feature_to_plot)



pdp.pdp_plot(pdp_dist, feature_to_plot)

plt.show()
best_clf.fit(X_new, y_train)
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=best_clf, dataset=df, model_features=features, feature='use')



# plot it

pdp.pdp_plot(pdp_goals, 'use')

plt.show()
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot

features_to_plot = ['love', 'good']

inter2  =  pdp.pdp_interact(model=best_clf, dataset=df, model_features=features, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=features_to_plot, plot_type='contour')

plt.show()
row_to_show = 5

data_for_prediction = df.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





best_clf.predict_proba(data_for_prediction_array)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(best_clf)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# Create object that can calculate shap values

explainer = shap.TreeExplainer(best_clf)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(df)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values[1], df)
shap.dependence_plot('love', shap_values[1], df, interaction_index="great")