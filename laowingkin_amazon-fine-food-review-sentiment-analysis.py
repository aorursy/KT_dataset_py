import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier

from string import punctuation

from sklearn import svm

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import nltk

from nltk import ngrams

from itertools import chain

from wordcloud import WordCloud

odf = pd.read_csv('../input/Reviews.csv')

odf['Helpful %'] = np.where(odf['HelpfulnessDenominator'] > 0, odf['HelpfulnessNumerator'] / odf['HelpfulnessDenominator'], -1)

odf['% Upvote'] = pd.cut(odf['Helpful %'], bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = ['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest = True)

odf.head()
df_s = odf.groupby(['Score', '% Upvote']).agg({'Id': 'count'})

df_s = df_s.unstack()

df_s.columns = df_s.columns.get_level_values(1)

fig = plt.figure(figsize=(15,10))



sns.heatmap(df_s[df_s.columns[::-1]].T, cmap = 'YlGnBu', linewidths=.5, annot = True, fmt = 'd', cbar_kws={'label': '# reviews'})

plt.yticks(rotation=0)

plt.title('How helpful users find among user scores')
df = odf[odf['Score'] != 3]

X = df['Text']

y_dict = {1:0, 2:0, 4:1, 5:1}

y = df['Score'].map(y_dict)
c = CountVectorizer(stop_words = 'english')



def text_fit(X, y, model,clf_model,coef_show=1):

    

    X_c = model.fit_transform(X)

    print('# features: {}'.format(X_c.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X_c, y, random_state=0)

    print('# train records: {}'.format(X_train.shape[0]))

    print('# test records: {}'.format(X_test.shape[0]))

    clf = clf_model.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)

    print ('Model Accuracy: {}'.format(acc))

    

    if coef_show == 1: 

        w = model.get_feature_names()

        coef = clf.coef_.tolist()[0]

        coeff_df = pd.DataFrame({'Word' : w, 'Coefficient' : coef})

        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])

        print('')

        print('-Top 20 positive-')

        print(coeff_df.head(20).to_string(index=False))

        print('')

        print('-Top 20 negative-')        

        print(coeff_df.tail(20).to_string(index=False))

    

    

text_fit(X, y, c, LogisticRegression())
text_fit(X, y, c, DummyClassifier(),0)
tfidf = TfidfVectorizer(stop_words = 'english')

text_fit(X, y, tfidf, LogisticRegression())

tfidf_n = TfidfVectorizer(ngram_range=(1,2),stop_words = 'english')

text_fit(X, y, tfidf_n, LogisticRegression())
df = df[df['Score'] == 5]

df = df[df['% Upvote'].isin(['0-20%', '20-40%', '60-80%', '80-100%'])]

df.shape



X = df['Text']

y_dict = {'0-20%': 0, '20-40%': 0, '60-80%': 1, '80-100%': 1}

y = df['% Upvote'].map(y_dict)



print('Class distribution:')

print(y.value_counts())
df_s = pd.DataFrame(data = [X,y]).T



Downvote_records = len(df_s[df_s['% Upvote'] == 0])

Downvote_indices = np.array(df_s[df_s['% Upvote'] == 0].index)



Upvote_indices = df_s[df_s['% Upvote'] == 1].index



random_upvote_indices = np.random.choice(Upvote_indices, Downvote_records, replace = False)

random_upvote_indices = np.array(random_upvote_indices)



under_sample_indices = np.concatenate([Downvote_indices,random_upvote_indices])



under_sample_data = df_s.ix[under_sample_indices, :]

X_u = under_sample_data['Text']

under_sample_data['% Upvote'] = under_sample_data['% Upvote'].astype(int)

y_u = under_sample_data['% Upvote']





print("Percentage of upvote transactions: ", len(under_sample_data[under_sample_data['% Upvote'] == 1])/len(under_sample_data))

print("Percentage of downvote transactions: ", len(under_sample_data[under_sample_data['% Upvote'] == 0])/len(under_sample_data))

print("Total number of records in resampled data: ", len(under_sample_data))
c = CountVectorizer(stop_words = 'english')



text_fit(X_u, y_u, c, LogisticRegression())
tfidf_n = TfidfVectorizer(ngram_range=(1,2),stop_words = 'english')



text_fit(X_u, y_u, tfidf_n, LogisticRegression())
#pd.set_option('display.max_colwidth', -1)

print('Downvote score 5 comments examples:')

print(under_sample_data[under_sample_data['% Upvote']==0]['Text'].iloc[:100:20])

print('Upvote score 5 comments examples')

print(under_sample_data[under_sample_data['% Upvote']==1]['Text'].iloc[:100:20])
under_sample_data['word_count'] = under_sample_data['Text'].apply(lambda x: len(x.split()))

under_sample_data['capital_count'] = under_sample_data['Text'].apply(lambda x: sum(1 for c in x if c.isupper()))

under_sample_data['question_mark'] = under_sample_data['Text'].apply(lambda x: sum(1 for c in x if c == '?'))

under_sample_data['exclamation_mark'] = under_sample_data['Text'].apply(lambda x: sum(1 for c in x if c == '!'))

under_sample_data['punctuation'] = under_sample_data['Text'].apply(lambda x: sum(1 for c in x if c in punctuation))



print(under_sample_data.groupby('% Upvote').agg({'word_count': 'mean', 'capital_count': 'mean', 'question_mark': 'mean', 'exclamation_mark': 'mean', 'punctuation': 'mean'}).T)



X_num = under_sample_data[under_sample_data.columns.difference(['% Upvote', 'Text'])]

y_num = under_sample_data['% Upvote']

X_train, X_test, y_train, y_test = train_test_split(X_num, y_num, random_state=0)



clf_lr = LogisticRegression().fit(X_train, y_train)

acc_lr = clf_lr.score(X_test, y_test)

print('Logistic Regression accuracy: {}'.format(acc_lr))



clf_svm = svm.SVC().fit(X_train, y_train)

acc_svm = clf_svm.score(X_test, y_test)

print('SVM accuracy: {}'.format(acc_svm))
df_user = odf.groupby(['UserId', 'ProfileName']).agg({'Score':['count', 'mean']})

df_user.columns = df_user.columns.get_level_values(1)

df_user.columns = ['Score count', 'Score mean']

df_user = df_user.sort_values(by = 'Score count', ascending = False)

print(df_user.head(10))

def plot_user(UserId):

    df_1user = odf[odf['UserId'] == UserId]['Score']

    df_1user_plot = df_1user.value_counts(sort=False)

    ax = df_1user_plot.plot(kind = 'bar', figsize = (15,10), title = 'Score distribution of user {} review'.format(odf[odf['UserId'] == UserId]['ProfileName'].iloc[0]))



plot_user('A3OXHLG6DIBRW8')
print(df_user[(df_user['Score mean']<3.5) & (df_user['Score mean']>2.5)].head())
plot_user('A2M9D9BDHONV3Y')
def get_token_ngram(score, benchmark, userid='all'):



    if userid != 'all':

        df = odf[(odf['UserId'] == userid) & (odf['Score'] == score)]['Text']

    else:

        df = odf[odf['Score'] == score]['Text']

        

    count = len(df)

    total_text = ' '.join(df)

    total_text = total_text.lower()

    stop = set(stopwords.words('english'))

    total_text = nltk.word_tokenize(total_text)

    total_text = [word for word in total_text if word not in stop and len(word) >= 3]

    lemmatizer = WordNetLemmatizer()

    total_text = [lemmatizer.lemmatize(w,'v') for w in total_text]

    bigrams = ngrams(total_text,2)

    trigrams = ngrams(total_text, 3)



    # look at 2-gram and 3-gram together

    combine = chain(bigrams, trigrams)

    text = nltk.Text(combine)

    fdist = nltk.FreqDist(text)

    

    # return only phrase occurs more than benchmark of his reviews

    return sorted([(w,fdist[w],str(round(fdist[w]/count*100,2))+'%') for w in set(text) if fdist[w] >= count*benchmark], key=lambda x: -x[1])



# score 1-5 reviews with this user

index = ['Phrase', 'Count', 'Occur %']



for j in range(1,6):

    test = pd.DataFrame()

    d = get_token_ngram(j, 0.25, 'A2M9D9BDHONV3Y')

    print('score {} reviews most popular 2-gram / 3-gram:'.format(j))

    for i in d:

        test = test.append(pd.Series(i, index = index), ignore_index = True)

    test = test.sort_values('Count', ascending=False)

    print(test)



# score 1-5 reviews with all users

index = ['Phrase', 'Count', 'Occur %']



for j in range(1,6):

    test = pd.DataFrame()

    # easier benchmark since we have many different users here, thus different phrase

    d = get_token_ngram(j, 0.03)

    print('score {} reviews most popular 2-gram / 3-gram:'.format(j))

    for i in d:

        test = test.append(pd.Series(i, index = index), ignore_index = True)

    test = test.sort_values('Count', ascending=False)

    print(test)
def get_token_adj(score, benchmark, userid='all'):

    

    if userid != 'all':

        df = odf[(odf['UserId'] == userid) & (odf['Score'] == score)]['Text']

    else:

        df = odf[odf['Score'] == score]['Text']

        

    count = len(df)

    total_text = ' '.join(df)

    total_text = total_text.lower()

    stop = set(stopwords.words('english'))

    total_text = nltk.word_tokenize(total_text)

    total_text = [word for word in total_text if word not in stop and len(word) >= 3]

    lemmatizer = WordNetLemmatizer()

    total_text = [lemmatizer.lemmatize(w,'a') for w in total_text]

    # get adjective only

    total_text = [word for word, form in nltk.pos_tag(total_text) if form == 'JJ']

    

    text = nltk.Text(total_text)

    fdist = nltk.FreqDist(text)

    

    # return only phrase occurs more than benchmark of his reviews

    return sorted([(w,fdist[w],str(round(fdist[w]/count*100,2))+'%') for w in set(text) if fdist[w] >= count*benchmark], key=lambda x: -x[1])
# score 1-5 reviews with this user

index = ['Phrase', 'Count', 'Occur %']



for j in range(1,6):

    test = pd.DataFrame()

    d = get_token_adj(j, 0.25, 'A2M9D9BDHONV3Y')

    print('score {} reviews most popular adjectives word:'.format(j))

    for i in d:

        test = test.append(pd.Series(i, index = index), ignore_index = True)

    test = test.sort_values('Count', ascending=False)

    print(test)
# score 1-5 reviews with all users

index = ['Phrase', 'Count', 'Occur %']



for j in range(1,6):

    test = pd.DataFrame()

    d = get_token_adj(j, 0.05)

    print('score {} reviews most popular adjectives word:'.format(j))

    for i in d:

        test = test.append(pd.Series(i, index = index), ignore_index = True)

    test = test.sort_values('Count', ascending=False)

    print(test)