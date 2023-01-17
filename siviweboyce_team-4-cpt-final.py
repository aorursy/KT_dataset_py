!pip install comet_ml
import comet_ml
!pip install spacy

!pip install NLTK
import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from numpy import arange

import seaborn as sns



#Natural Language Processing

import re

import spacy.cli

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer





#Matrix measurement

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, precision_score, recall_score



#Resampling techniques

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



#Machine Learning Models

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import SGDClassifier, LinearRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import  KFold





from wordcloud import WordCloud 



# from google.colab import drive
spacy.cli.download('en_core_web_sm')

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('stopwords')

stop = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
train = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/train%20(1).csv')

print(train.head())
test = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/test%20(1).csv')

print(test.head())
def clean_text(df):

  i = 0

  for tweet in df['message']:

    tweet = tweet.lower()

    tweet = re.sub(r'http\S+', 'LINK', tweet)

    tweet = re.sub(r'@\S+', 'USER_REF', tweet)

    tweet = re.sub(r'[^\w\s]', '', tweet)

    tweet = tweet.lstrip()

    tweet = tweet.rstrip()

    tweet = tweet.replace('  ', ' ')

    df.loc[i, 'message'] = tweet

    i += 1



clean_text(train)

train
clean_text(test)

test
def remove_stopwords(df):

    my_stop_words = stopwords.words('english')

    my_stop_words.append('LINK')

    my_stop_words.append('USER_REF')



    df_index = 0



    for tweet in df['message']:

      tweet = word_tokenize(tweet)

      tweet = [word for word in tweet if not word in my_stop_words]

      tweet = ' '.join(tweet)



      df.loc[df_index, 'message'] = tweet

      df_index += 1



    return df
remove_stopwords(train)
remove_stopwords(test)
def entities(df):

    df_index = 0



    for tweet in df['message']:

      tweet = nlp(tweet)



      for entity in tweet.ents:

        df.loc[df_index, 'message'] = df.loc[df_index, 'message'].replace(str(entity.text), str(entity.label_))



      df_index += 1



      return df
entities(train)
entities(test)
def lem_text(df):

    df_index = 0



    for tweet in df['message']:

      tweet = nlp(tweet)

      

      for token in tweet:

        df.loc[df_index, 'message'] = df.loc[df_index, 'message'].replace(str(token.text), str(token.lemma_))



      df_index += 1



      return df
lem_text(train)
lem_text(test)
train.isnull().sum()

test.isnull().sum()
train.sentiment.value_counts()
counts = train["sentiment"].value_counts()

plt.bar(range(len(counts)), counts)

plt.xticks([0, 1, 2, 3], ['Pro', 'News', 'Neutral', 'Anti'])



plt.ylabel("Total per class")

plt.xlabel("Sentiment Classes")

plt.show()
#Percentage of the major class

len(train[train.sentiment==1])/len(train.sentiment)
#word clouds

news = train[train['sentiment'] == 2]['message']

pro = train[train['sentiment'] == 1]['message']

neutral =train[train['sentiment'] == 0]['message']

Anti = train[train['sentiment'] ==-1]['message']





news = [word for line in news for word in line.split()]

pro = [word for line in pro for word in line.split()]

neutral = [word for line in neutral for word in line.split()]

Anti= [word for line in Anti for word in line.split()]



news = WordCloud(

    background_color='white',

    max_words=20,

    max_font_size=40,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(news))



pro = WordCloud(

    background_color='white',

    max_words=20,

    max_font_size=40,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(pro))







neutral = WordCloud(

    background_color='white',

    max_words=20,

    max_font_size=40,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(neutral))





Anti = WordCloud(

    background_color='white',

    max_words=20,

    max_font_size=40,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(Anti))





fig, axs = plt.subplots(2, 2, figsize = (20, 10))

fig.tight_layout(pad = 0)



axs[0, 0].imshow(news)

axs[0, 0].set_title('Words from news tweets', fontsize = 20)

axs[0, 0].axis('off')



axs[0, 1].imshow(pro)

axs[0, 1].set_title('Words from pro tweets', fontsize = 20)

axs[0, 1].axis('off')





axs[1, 0].imshow(Anti)

axs[1, 0].set_title('Words from anti tweets', fontsize = 20)

axs[1, 0].axis('off')



axs[1, 1].imshow(neutral)

axs[1, 1].set_title('Words from neutral tweets', fontsize = 20)

axs[1, 1].axis('off')



plt.savefig('joint_cloud.png')
X = train['message']

X
y = train['sentiment']

y
tf_vecto = TfidfVectorizer(lowercase = True,stop_words = 'english',ngram_range=(1, 2))

X = tf_vecto.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#X_test = test['message']
#X_test = tf_vect.transform(X_test)
def accuracy(model):

    features = train['message']

    target = train['sentiment']



    train_scores = []

    test_scores = []



    #tf_vect = TfidfVectorizer(ngram_range=(1, 2))

    tf_vecto = TfidfVectorizer(lowercase = True,stop_words = 'english',ngram_range=(1, 2))





    folds = KFold(n_splits=5, shuffle=True)



    for train_index, test_index in folds.split(features):

        x_train, x_test = features.iloc[train_index], features.iloc[test_index]    

        y_train, y_test = target.iloc[train_index], target.iloc[test_index]



        x_train = tf_vecto.fit_transform(x_train)

        x_test = tf_vecto.transform(x_test)

     

        model.fit(x_train, y_train)

        train_predictions = model.predict(x_train)

        test_predictions = model.predict(x_test)



        train_score = accuracy_score(y_train, train_predictions)

        train_scores.append(train_score)



        test_score = accuracy_score(y_test, test_predictions)

        test_scores.append(test_score)



    avg_train_accuracy = np.mean(train_scores)

    avg_test_accuracy = np.mean(test_scores)



    return [avg_train_accuracy, avg_test_accuracy]

sv = SVC()



sv_accuracy = accuracy(sv)

sv_accuracy
bernoulli = BernoulliNB()



bernoulli_accuracy = accuracy(bernoulli)

bernoulli_accuracy
mnb = MultinomialNB()



mnb_accuracy = accuracy(mnb)

mnb_accuracy
sgd = SGDClassifier()



sgd_accuracy = accuracy(sgd)

sgd_accuracy
rand_forest = RandomForestClassifier()



rand_forest_accuracy = accuracy(rand_forest)

rand_forest_accuracy
knn = KNeighborsClassifier()



knn_accuracy = accuracy(knn)

knn_accuracy
grad_booster = GradientBoostingClassifier()



grad_booster_accuracy = accuracy(grad_booster)

grad_booster_accuracy
extra_trees = ExtraTreesClassifier()



extra_trees_accuracy = accuracy(extra_trees)

extra_trees_accuracy
bagging = BaggingClassifier()



bagging_accuracy = accuracy(bagging)

bagging_accuracy
dec_tree = DecisionTreeClassifier()



dec_tree_accuracy = accuracy(dec_tree)

dec_tree_accuracy
linear_sv = LinearSVC()



linear_sv_accuracy = accuracy(linear_sv)

linear_sv_accuracy
models = ['SVC', 'Bernoulli', 'Multinomial Naive Bayes', 'SGDClassifier', 'Random Forest', 'KNearestNeighbours', 'Gradient Booster', 'Extra Trees', 'Bagging', 'Decision Tree', 'Linear SV']

bar_widths = [sv_accuracy[1], bernoulli_accuracy[1], mnb_accuracy[1], sgd_accuracy[1], rand_forest_accuracy[1], knn_accuracy[1], grad_booster_accuracy[1], extra_trees_accuracy[1], bagging_accuracy[1], dec_tree_accuracy[1], linear_sv_accuracy[1]]

bar_positions = arange(11) + 0.75

tick_positions = range(1,12)



fig, ax = plt.subplots()

ax.barh(bar_positions, bar_widths, 0.5)

ax.set_yticks(tick_positions)

ax.set_yticklabels(models)



ax.set_ylabel('Model')

ax.set_xlabel('Accuracy')

ax.set_title('Accuracy For Each Model Trained')



plt.show()
linear_sv.fit(X_train, y_train)
#confusion matrix and classification_report

y_pred = linear_sv.predict(X_test)



print(confusion_matrix(y_test,y_pred))



print('\n\nAccuracy score: ' + str(accuracy_score(y_test, y_pred)))

print("\n\nClassification Report:\n\n", classification_report(y_test,y_pred,target_names=['Anti', 'Neutral','Pro','News']))
sentiment_code = {1:'Pro', 2:'News', 0:'Neutral', -1:'Anti'}
train['sentiment_code'] = train['sentiment'].map(sentiment_code)
aux_train = train[['sentiment', 'sentiment_code']].drop_duplicates().sort_values('sentiment_code')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12.8,6))

sns.heatmap(conf_matrix, 

            annot=True,

            cbar=False,

            fmt='g',

            xticklabels=aux_train['sentiment'].values, 

            yticklabels=aux_train['sentiment'].values,

            cmap="Blues")

plt.ylabel('Predicted')

plt.xlabel('Actual')

plt.title('Confusion matrix')

plt.show()
sgd.fit(X_train,y_train)
y_predict = sgd.predict(X_test)



print(confusion_matrix(y_test,y_predict))



print('\n\nAccuracy score: ' + str(accuracy_score(y_test, y_pred)))

print("\n\nClassification Report:\n\n", classification_report(y_test,y_predict,target_names=['Anti', 'Neutral','Pro','News']))



sentiment_code = {1:'Pro', 2:'News', 0:'Neutral', -1:'Anti'}
train['sentiment_code'] = train['sentiment'].map(sentiment_code)
aux_train = train[['sentiment', 'sentiment_code']].drop_duplicates().sort_values('sentiment_code')

conf_matrix = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(12.8,6))

sns.heatmap(conf_matrix, 

            annot=True,

            cbar=False,

            fmt='g',

            xticklabels=aux_train['sentiment'].values, 

            yticklabels=aux_train['sentiment'].values,

            cmap="Blues")

plt.ylabel('Predicted')

plt.xlabel('Actual')

plt.title('Confusion matrix')

plt.show()
train = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/train%20(1).csv')

print(train.head())
test = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/test%20(1).csv')

print(test.head())
clean_text(train)

remove_stopwords(train)

entities(train)

lem_text(train)
train_majority = train[train.sentiment== 1]

train_0 = train[train.sentiment== 0]

train_2 = train[train.sentiment== 2]



train_minority = train[train.sentiment==-1]



# Downsample majority classes

train_majority_downsampled = resample(train_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=1296,     # to match minority class

                                 random_state=123) # reproducible results





train_0_downsampled = resample(train_0, 

                                 replace=False,    

                                 n_samples=1296,     

                                 random_state=123) 



train_2_downsampled = resample(train_2, 

                                 replace=False,    

                                 n_samples=1296,     

                                 random_state=123) 



                      







# Combine minority class with downsampled majority class

train_downsampled1 = pd.concat([train_0_downsampled,train_2_downsampled])



train_downsampled2 = pd.concat([train_majority_downsampled, train_minority])



train_downsampled =  pd.concat([train_downsampled1, train_downsampled2])
train_downsampled
train_downsampled['sentiment'].value_counts()
counts = train["sentiment"].value_counts()

counti = train_downsampled['sentiment'].value_counts()



plt.bar(range(len(counts)), counts)

plt.bar(range(len(counts)),counti,color='red')

plt.xticks([0, 1, 2, 3], ['Pro', 'News', 'Neutral', 'Anti'])



plt.ylabel("Total per class")

plt.xlabel("Sentiment Classes")

plt.legend(['original','resampled'])

plt.show()
X_down = train['message']

X_down
y_down = train['sentiment']

y_down
X_down = tf_vecto.fit_transform(X_down)

X_down
#train_test_split

X_train1,X_test1,y_train1,y_test1 = train_test_split(X_down,y_down,test_size=0.2,random_state=0)


lsvm = LinearSVC()

lsvm.fit(X_train1, y_train1)
#confusion matrix and classification_report

y_pred1 = lsvm.predict(X_test1)



print(confusion_matrix(y_test1,y_pred1))



print('\n\nAccuracy score: ' + str(accuracy_score(y_test1, y_pred1)))

print("\n\nClassification Report:\n\n",classification_report(y_test1,y_pred1,target_names=['Anti', 'Neutral','Pro','News']))
sentiment_code = {1:'Pro', 2:'News', 0:'Neutral', -1:'Anti'}
train['sentiment_code'] = train['sentiment'].map(sentiment_code)
aux_train = train[['sentiment', 'sentiment_code']].drop_duplicates().sort_values('sentiment_code')

conf_matrix = confusion_matrix(y_test1, y_pred1)

plt.figure(figsize=(12.8,6))

sns.heatmap(conf_matrix, 

            annot=True,

            cbar=False,

            fmt='g',

            xticklabels=aux_train['sentiment'].values, 

            yticklabels=aux_train['sentiment'].values,

            cmap="Blues")

plt.ylabel('Predicted')

plt.xlabel('Actual')

plt.title('Confusion matrix')

plt.show()
print(X_train.shape,y_train.shape)
smote = SMOTE("minority")

X_sm , y_sm = smote.fit_resample(X_train,y_train)
print(X_sm.shape,y_sm.shape)
ls= LinearSVC()

ls.fit(X_sm, y_sm)





#confusion matrix and classification_report

y_predsm = ls.predict(X_test)



print(confusion_matrix(y_test,y_predsm))



print('\n\nAccuracy score: ' + str(accuracy_score(y_test, y_predsm)))

print("\n\nClassification Report:\n\n",classification_report(y_test,y_predsm,target_names=['Anti', 'Neutral','Pro','News']))
sentiment_code = {1:'Pro', 2:'News', 0:'Neutral', -1:'Anti'}
train['sentiment_code'] = train['sentiment'].map(sentiment_code)
aux_train = train[['sentiment', 'sentiment_code']].drop_duplicates().sort_values('sentiment_code')

conf_matrix = confusion_matrix(y_test, y_predsm)

plt.figure(figsize=(12.8,6))

sns.heatmap(conf_matrix, 

            annot=True,

            cbar=False,

            fmt='g',

            xticklabels=aux_train['sentiment'].values, 

            yticklabels=aux_train['sentiment'].values,

            cmap="Blues")

plt.ylabel('Predicted')

plt.xlabel('Actual')

plt.title('Confusion matrix')

plt.show()
train = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/train%20(1).csv')

print(train.head())
test = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/test%20(1).csv')

print(test.head())

sample = pd.read_csv('https://raw.githubusercontent.com/Stephane-Masamba/Team_4_CPT_ML-Classification/Mikael/sample_submission.csv')

print(sample.head())

clean_text(train)

remove_stopwords(train)
clean_text(test)

remove_stopwords(test)
X_min = train['message']

X_min

y_min = train['sentiment']

y_min

X_min = tf_vecto.fit_transform(X_min)

#train_test_split

X_train2,X_test2,y_train2,y_test2 = train_test_split(X_min,y_min,test_size=0.2,random_state=0)
sgd = SGDClassifier()



sgd_accuracy = accuracy(sgd)

sgd_accuracy



linear_sv = LinearSVC()



linear_sv_accuracy = accuracy(linear_sv)

linear_sv_accuracy

linear_sv.fit(X_train2, y_train2)

#confusion matrix and classification_report

y_pred2 = linear_sv.predict(X_test2)





print(confusion_matrix(y_test1,y_pred1))



print('\n\nAccuracy score: ' + str(accuracy_score(y_test2, y_pred1)))

print("\n\nClassification Report:\n\n",classification_report(y_test2,y_pred2,target_names=['Anti', 'Neutral','Pro','News']))
sentiment_code = {1:'Pro', 2:'News', 0:'Neutral', -1:'Anti'}
train['sentiment_code'] = train['sentiment'].map(sentiment_code)
aux_train = train[['sentiment', 'sentiment_code']].drop_duplicates().sort_values('sentiment_code')

conf_matrix = confusion_matrix(y_test1, y_pred1)

plt.figure(figsize=(12.8,6))

sns.heatmap(conf_matrix, 

            annot=True,

            cbar=False,

            fmt='g',

            xticklabels=aux_train['sentiment'].values, 

            yticklabels=aux_train['sentiment'].values,

            cmap="Blues")

plt.ylabel('Predicted')

plt.xlabel('Actual')

plt.title('Confusion matrix')

plt.show()
# import comet_ml in the top of your file

from comet_ml import Experiment

    

# Add the following code anywhere in your machine learning file

experiment = Experiment(api_key="kyaDe1YHDUV60KbpzF3dVpIuk",

                        project_name="general", workspace="rachel-ramonyai")
f1 = f1_score(y_test, y_predsm,average='macro')

precision = precision_score(y_test, y_pred,average='macro')

recall = recall_score(y_test, y_pred,average='macro')
params = {"kernel": 'linear',

          "model_type": "SVC",

          "stratify": True

          }
params = {

          "model_type": "Best LinearSVC",

          "stratify": True

          }



metrics = {"f1": f1,

           "recall": recall,

           "precision": precision

           }
# Log our parameters and results

experiment.log_parameters(params)

experiment.log_metrics(metrics)
experiment.end()