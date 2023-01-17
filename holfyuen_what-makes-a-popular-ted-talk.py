# Loading data
import numpy as np
import pandas as pd
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

ted = pd.read_csv("../input/ted_main.csv")
transcripts = pd.read_csv('../input/transcripts.csv')
print (ted.shape, transcripts.shape)
ted.head()
# Categorize events into TED and TEDx; exclude those that are non-TED events
ted = ted[ted['event'].str[0:3]=='TED'].reset_index()
ted.loc[:,'event_cat'] = ted['event'].apply(lambda x: 'TEDx' if x[0:4]=='TEDx' else 'TED')

print ("No. of talks remain: ", len(ted))
ted['film_date'] = ted['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
ted['published_date'] = ted['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
ted['film_month'] = ted['film_date'].apply(lambda x: x.month)
ted['pub_month'] = ted['published_date'].apply(lambda x: x.month)
ted['film_weekday'] = ted['film_date'].apply(lambda x: x.weekday()) # Monday: 0, Sunday: 6
ted['pub_weekday'] = ted['published_date'].apply(lambda x: x.weekday())
ted[['film_date','published_date']].head()
ted['ratings']=ted['ratings'].str.replace("'",'"')
ted=ted.merge(ted.ratings.apply(lambda x: pd.Series(pd.read_json(x)['count'].values,index=pd.read_json(x)['name'])), 
            left_index=True, right_index=True)
Positive = ['Beautiful', 'Courageous', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', 'Persuasive']
Negative = ['Confusing', 'Longwinded', 'Obnoxious', 'Unconvincing']
ted['positive']=ted.loc[:,Positive].sum(axis=1)+1
ted['negative']=ted.loc[:,Negative].sum(axis=1)+1
ted['pop_ratio']=ted['positive']/ted['negative']
ted.loc[:,'Popular'] = ted['pop_ratio'].apply (lambda x: 1 if x >5 else 0)

print ("No. of Not Popular talks: ", len(ted[ted['Popular']==0]))
# print ("Ratio of Popular talks: {:.4f}".format(len(ted[ted['Popular']==1])/ float(len(ted))))
overall_mean_popular = np.mean(ted.Popular)
print ("Ratio of Popular talks: {:.4f}".format(overall_mean_popular))
nums = ['comments', 'duration', 'languages', 'num_speaker', 'views']
sns.pairplot(ted, vars=nums, hue='Popular', hue_order = [1,0], diag_kind='kde', size=3);
ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded', 'Confusing', 'Informative', 'Fascinating', 'Unconvincing', 
           'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring', 'Popular']
plt.figure(figsize=(10,8))
sns.heatmap(ted[ratings].corr(), annot=True, cmap='RdBu');
plt.figure(figsize=(9,6))
lw = ted.Longwinded / (ted.positive + ted.negative)
plt.scatter(ted.duration, lw, s=7)
plt.show()
lw_talks_id = lw[lw>0.2].index
ted.loc[lw_talks_id,['title','main_speaker','speaker_occupation','event','duration','Longwinded','positive','negative','Popular']]
fm = ted.groupby('film_month')['Popular'].mean().round(4) - overall_mean_popular
pm = ted.groupby('pub_month')['Popular'].mean().round(4) - overall_mean_popular
by_month = pd.concat([fm, pm], axis=1)
by_month.columns = ['Filmed','Published']
by_month.plot(kind='bar', figsize=(9,6))
plt.title('Ratio of Popular Talks by Month (Net of Overall Mean)', fontsize=14)
plt.xticks(rotation=0)
plt.show()
fw = ted.groupby('film_weekday')['Popular'].mean().round(4) - overall_mean_popular
pw = ted.groupby('pub_weekday')['Popular'].mean().round(4) - overall_mean_popular
by_weekday = pd.concat([fw, pw], axis=1)
by_weekday.columns = ['Filmed', 'Published']
by_weekday.index = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
by_weekday.plot(kind='bar', figsize=(9,6))
plt.title('Ratio of Popular Talks by Day of Week', fontsize=14)
plt.xticks(rotation=0)
plt.show()
x = ted.groupby('num_speaker').mean()['Popular'].rename('pct_popular')
pd.concat([x, ted.num_speaker.value_counts().rename('talks_count')], axis=1)
count_vector = CountVectorizer(stop_words='english',min_df=20/len(ted)) # Only keep those with 20 or more occurrences
tag_array = count_vector.fit_transform(ted.tags).toarray()
tag_matrix = pd.DataFrame(tag_array, columns = count_vector.get_feature_names())
all_tags = tag_matrix.columns
tag_matrix = pd.concat([tag_matrix, ted.Popular], axis=1)
by_tag = dict()
for col in all_tags:
    by_tag[col]=tag_matrix.groupby(col)['Popular'].mean()[1] - overall_mean_popular
tag_rank = pd.DataFrame.from_dict(by_tag, orient='index')
tag_rank.columns = ['pop_rate_diff']

plt.figure(figsize=(16,7))
plt.subplot(121)
bar_2 = tag_rank.sort_values(by='pop_rate_diff', ascending=False)[:15]
sns.barplot(x=bar_2.pop_rate_diff, y=bar_2.index, color='blue')
plt.title('15 Most Popular Tags')
plt.xlabel('Ratio of Popular Talk (Net of Mean)', fontsize=14)
plt.yticks(fontsize=12)
plt.subplot(122)
bar_1 = tag_rank.sort_values(by='pop_rate_diff')[:15]
sns.barplot(x=bar_1.pop_rate_diff, y=bar_1.index, color='red')
plt.title('15 Most Unpopular Tags')
plt.xlabel('Ratio of Popular Talk (Net of Mean)', fontsize=14)
plt.yticks(fontsize=12)
plt.show()
ted.loc[:,'occ'] = ted.speaker_occupation.copy()
ted.occ = ted.occ.fillna('Unknown')
ted.occ = ted.occ.str.replace('singer/songwriter', 'singer, songwriter')
ted.occ = ted.occ.str.replace('singer-songwriter', 'singer, songwriter')
count_vector2 = CountVectorizer(stop_words='english', min_df=20/len(ted))
occ_array = count_vector2.fit_transform(ted.occ).toarray()
occ_matrix = pd.DataFrame(occ_array, columns = count_vector2.get_feature_names())
all_occ = occ_matrix.columns
occ_matrix = pd.concat([occ_matrix, ted.Popular], axis=1)
by_occ = dict()
for col in all_occ:
    by_occ[col]=occ_matrix.groupby(col)['Popular'].mean()[1] - overall_mean_popular
occ_rank = pd.DataFrame.from_dict(by_occ, orient='index')
occ_rank.columns = ['pop_rate_diff']

plt.figure(figsize=(16,7))
plt.subplot(121)
bar_2 = occ_rank.sort_values(by='pop_rate_diff', ascending=False)[:10]
sns.barplot(x=bar_2.pop_rate_diff, y=bar_2.index, color='blue')
plt.title('10 Most Popular Occupation Keywords', fontsize=14)
plt.xlabel('Ratio of Popular Talk (Net of Mean)')
plt.yticks(fontsize=12)
plt.subplot(122)
bar_1 = occ_rank.sort_values(by='pop_rate_diff')[:10]
sns.barplot(x=bar_1.pop_rate_diff, y=bar_1.index, color='red')
plt.title('10 Most Unpopular Occupation Keywords', fontsize=14)
plt.xlabel('Ratio of Popular Talk (Net of Mean)')
plt.yticks(fontsize=12)
plt.show()
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist, word_tokenize
stopwords = set(STOPWORDS)

plt.figure(figsize=(15,12))

plt.subplot(121)
word_pos = FreqDist(w for w in word_tokenize(' '.join(ted.loc[ted.Popular==1, 'title']).lower()) if (w not in stopwords) & (w.isalpha()))
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_pos)
plt.imshow(wordcloud)
plt.title("Wordcloud for Title - Popular", fontsize=16)
plt.axis("off")

plt.subplot(122)
word_neg = FreqDist(w for w in word_tokenize(' '.join(ted.loc[ted.Popular==0, 'title']).lower()) if (w not in stopwords) & (w.isalpha()))
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_neg)
plt.imshow(wordcloud)
plt.title("Wordcloud for Title - Unpopular", fontsize=16)
plt.axis("off")
plt.show()
from nltk.stem import WordNetLemmatizer
import string

wnl = WordNetLemmatizer()

extrasw = set(['say', 'says', 'talk', 'us', 'world', 'make'])
stopwords2 = stopwords.union(extrasw)

pos_str = ' '.join(ted.loc[ted.Popular==1, 'description']).lower().translate(str.maketrans('','',string.punctuation))
neg_str = ' '.join(ted.loc[ted.Popular==0, 'description']).lower().translate(str.maketrans('','',string.punctuation))

plt.figure(figsize=(15,12))

plt.subplot(121)
word_pos = FreqDist(wnl.lemmatize(wnl.lemmatize(w), pos='v') for w in word_tokenize(pos_str) if w not in stopwords2)
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_pos)
plt.imshow(wordcloud)
plt.title("Wordcloud for Description - Popular", fontsize=16)
plt.axis("off")

plt.subplot(122)
word_neg = FreqDist(wnl.lemmatize(wnl.lemmatize(w), pos='v') for w in word_tokenize(neg_str) if w not in stopwords2)
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_neg)
plt.imshow(wordcloud)
plt.title("Wordcloud for Description - Unpopular", fontsize=16)
plt.axis("off")
plt.show()
y = ted.Popular
x = pd.concat([occ_matrix.drop('Popular', axis=1), tag_matrix.drop('Popular', axis=1)], axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=36)
# Write function on training and testing
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from time import time

def train_predict(clf, parameters):
        
    clf.fit(x_train, y_train)
    scorer = make_scorer(fbeta_score, beta=1)

    # 5-fold cross validation
    start = time()

    grid_obj = GridSearchCV(clf, parameters, cv=5, scoring=scorer)
    grid_fit = grid_obj.fit(x_train, y_train)
    best_clf = grid_fit.best_estimator_
    best_prob_train = best_clf.predict_proba(x_train)
    best_prob = best_clf.predict_proba(x_test)
    best_pred_train = (best_prob_train[:,1]>0.65)*1
    best_pred = (best_prob[:,1]>0.65)*1

    end = time()

    run_time = end - start

    # Report results
    print (clf.__class__.__name__ + ":")
    print ("Accuracy score on training data (optimized by grid-search CV): {:.4f}".format(best_clf.score(x_train, y_train)))
    print ("Accuracy score on testing data (optimized by grid-search CV): {:.4f}".format(best_clf.score(x_test, y_test)))
    print ("F1-score on training data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_train, best_pred_train, beta = 1)))
    print ("F1-score on testing data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_test, best_pred, beta = 1)))
    print ("Parameters: ", grid_fit.best_params_)
    # print (confusion_matrix(y_test, best_predictions))
    print ("Total runtime: {:.4f} seconds".format(run_time))
    return best_prob_train, best_prob, best_pred_train, best_pred
# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 108)
parameters = {'n_estimators': range(10,21,2), 'max_features': ['auto', 'log2'], 'min_samples_split': range(3, 7)}
rf_prob_train, rf_prob, rf_pred_train, rf_pred = train_predict(clf, parameters)
# train_predict(clf, parameters)
pd.DataFrame(confusion_matrix(y_train, rf_pred_train))
pd.DataFrame(confusion_matrix(y_test, rf_pred))
from scipy.sparse import hstack

y = ted.Popular

cv_t = CountVectorizer(stop_words='english', max_features=10000, lowercase=True) # Title
cv_d = CountVectorizer(stop_words='english', max_features=1000, lowercase=True) # Description
x_t = cv_t.fit_transform(ted.title)
x_d = cv_d.fit_transform(ted.description)
x_all = hstack([x_t, x_d])

x_train, x_test, y_train, y_test = train_test_split(x_all, y, test_size=0.25, random_state=36)
from sklearn.naive_bayes import MultinomialNB

clf2 = MultinomialNB()
parameters={}
nb_prob_train, nb_prob, nb_pred_train, nb_pred = train_predict(clf2, parameters)
pd.DataFrame(confusion_matrix(y_train, nb_pred_train))
pd.DataFrame(confusion_matrix(y_test, nb_pred))
m3_prob_train = (rf_prob_train + nb_prob_train)/2
m3_prob = (rf_prob + nb_prob)/2
m3_pred_train = (m3_prob_train[:,1]>0.65)*1
m3_pred = (m3_prob[:,1]>0.65)*1
from sklearn.metrics import accuracy_score
print ("Model 3:")
print ("Accuracy score on training data (optimized by grid-search CV): {:.4f}".format(accuracy_score(y_train, m3_pred_train)))
print ("Accuracy score on testing data (optimized by grid-search CV): {:.4f}".format(accuracy_score(y_test, m3_pred)))
print ("F1-score on training data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_train, m3_pred_train, beta = 1)))
print ("F1-score on testing data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_test, m3_pred, beta = 1)))
pd.DataFrame(confusion_matrix(y_train, m3_pred_train))
pd.DataFrame(confusion_matrix(y_test, m3_pred))