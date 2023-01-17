# Packages



import pandas as pd

import numpy as np

import string

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords





# Visualisation



import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS





# Model Building



    #classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier



    #vectorizers

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



    #training features

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV



    #performance measures

from sklearn.metrics import accuracy_score,log_loss

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import make_scorer



    #filter future warnings

#two futre warnings occured multiple times when running cross validation and GridSearchCV have been removed

#FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.

#FutureWarning: The default value of cv will change from 3 to 5 in version 0.22

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# view the data

train.head()
test.head()
print('Number of rows and columns in train data:{}' .format(train.shape))

print('Number of rows and columns in test data:{}' .format(test.shape))
train.isnull().sum()
test.isnull().sum()
type_sum = train.groupby(['type']).count()

type_sum.sort_values('posts', ascending=False, inplace=True)

type_sum
train['word_count'] = train['posts'].apply(lambda x: len(str(x).split(" ")))

word_count = train.groupby('type').sum()

word_count.sort_values('word_count', ascending=False, inplace=True)

word_count
#drop word_count column

train = train.drop(['word_count'], axis=1)
dim = (15.0, 4.0)

fig, ax = plt.subplots(figsize=dim)

cmrmap = sns.color_palette('CMRmap', 16)

sns.set_palette(cmrmap)

sns.countplot(x='type', data=train,

              order=['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ',

                     'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP',

                     'ISTJ', 'ISTP'])

plt.title('Distribution of Myers-Briggs Types in the Dataset', fontsize=16)

plt.xlabel('Personality Type')

plt.ylabel('Count of Posts')

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
# Create a binary column for each of the 4 dimension types

train['Mind'] = train['type'].map(lambda x: 'Extroverted'

                                  if x[0] == 'E' else 'Introverted')

train['Energy'] = train['type'].map(lambda x: 'Intuitive'

                                    if x[1] == 'N' else 'Sensing')

train['Nature'] = train['type'].map(lambda x: 'Thinking'

                                    if x[2] == 'T' else 'Feeling')

train['Tactics'] = train['type'].map(lambda x: 'Judging'

                                     if x[3] == 'J' else 'Perceiving')
# Countplot of the Introverted - Extroverted variable

IEcolors = sns.xkcd_palette(['red', 'soft pink'])

sns.set_palette(IEcolors)

sns.countplot(x='Mind', data=train, order=['Introverted', 'Extroverted'])

plt.ylim(0, 8000)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('Introverted vs Extroverted')

plt.ylabel('Count of each Personality Type')

plt.title('Introversion vs. Extroversion', fontsize=14)

plt.show()

# Start with one review:

def generate_wordcloud(text, title):

    # Create and generate a word cloud image:

    wordcloud = WordCloud(background_color='white').generate(text)



    # Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')

    plt.title(title, fontsize=40)

    plt.show()
# Group together posts written by those under the mind variable

words_of_mind = train.groupby('Mind')['posts'].apply(' '.join).reset_index()
for i, t in enumerate(words_of_mind['Mind']):

    text = words_of_mind.iloc[i,1]

    generate_wordcloud(text, t)
# Countplot of the Intuitive - Sensing variable

NScolors = sns.xkcd_palette(['blue', 'light blue'])

sns.set_palette(NScolors)

sns.countplot(x='Energy', data=train, order=['Intuitive', 'Sensing'])

plt.title('Intuitive vs. Sensing', fontsize=14)

plt.ylim(0, 8000)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
words_of_energy = train.groupby('Energy')['posts'].apply(' '.join).reset_index()

for i, t in enumerate(words_of_energy['Energy']):

    text = words_of_energy.iloc[i, 1]

    generate_wordcloud(text, t)
# Countplot of the Tinking - Feeling variable

TFcolors = sns.xkcd_palette(['green', 'pale green'])

sns.set_palette(TFcolors)

sns.countplot(x='Nature', data=train, order=['Thinking', 'Feeling'])

plt.title('Thinking vs. Feeling', fontsize=14)

plt.ylim(0, 8000)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
words_of_nature = train.groupby('Nature')['posts'].apply(' '.join).reset_index()

for i, t in enumerate(words_of_nature['Nature']):

    text = words_of_nature.iloc[i, 1]

    generate_wordcloud(text, t)
# Countplot of Judging - Perceiving

JPcolors = sns.xkcd_palette(['purple', 'lavender'])

sns.set_palette(JPcolors)

sns.countplot(x='Tactics', data=train, order=['Judging', 'Perceiving'])

plt.title('Judging vs. Perceiving', fontsize=14)

plt.ylim(0, 8000)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
words_of_tactics = train.groupby('Tactics')['posts'].apply(' '.join).reset_index()

for i, t in enumerate(words_of_tactics['Tactics']):

    text = words_of_tactics.iloc[i, 1]

    generate_wordcloud(text, t)
def remove_delimiters (post):

    new = post.replace('|||',' ')

    return ' '.join(new.split())



train['posts'] = train['posts'].apply(remove_delimiters)

test['posts'] = test['posts'].apply(remove_delimiters)
## Remove urls

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'



#apply to train set

train['posts'] = train['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)



#apply to test set

test['posts'] = test['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)
train['posts'] = train['posts'].str.lower()



test['posts'] = test['posts'].str.lower()
#Remove punctuation & numbers

def remove_punctuation(post):

    punc_numbers = string.punctuation + '0123456789'

    return ''.join([l for l in post if l not in punc_numbers])



train['posts'] = train['posts'].apply(remove_punctuation)



test['posts'] = test['posts'].apply(remove_punctuation)
train.head()
# Lematise posts

lemmatizer = WordNetLemmatizer()

train['lemma'] = [' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])for text in train['posts']]

test['lemma'] = [' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])for text in test['posts']]
train.head()
#Check for stopwords train

stop = stopwords.words('english')

train['stopwords'] = train['lemma'].apply(lambda x: len([x for x in x.split() if x in stop]))

train[['lemma','stopwords']].head()
#Check for stopwords test

stop = stopwords.words('english')

test['stopwords'] = test['lemma'].apply(lambda x: len([x for x in x.split() if x in stop]))

test[['lemma','stopwords']].head()
def remove_stop_words(word):

    if word not in stop:

        return word

    else:

        return ''
test['lemma_no_stop'] = [' '.join([remove_stop_words(word) for word in text.split(' ')])for text in test['lemma']]
test.head()
#Create binary classes for each of the personality characteristics

train['E'] = train['type'].apply(lambda x: x[0] == 'E').astype('int')

train['N'] = train['type'].apply(lambda x: x[1] == 'N').astype('int')

train['T'] = train['type'].apply(lambda x: x[2] == 'T').astype('int')

train['J'] = train['type'].apply(lambda x: x[3] == 'J').astype('int')
train.head()
mind_df = train[['lemma','E']]
vect_mind = TfidfVectorizer(lowercase=True, 

                            stop_words='english', 

                            max_features=250,

                            min_df=4,

                            max_df=0.5

                           )
vect_mind.fit(mind_df['lemma'])

X_count_mind = vect_mind.transform(mind_df['lemma'])
X_count_mind.shape
vect_mind.get_feature_names()
X = X_count_mind

y = mind_df['E']

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y,

                                                    test_size =0.3,

                                                   random_state = 42)
def scoring_function_log_loss(y_test, y_pred_test):

    return log_loss(y_test, y_pred_test)
score_log_loss = make_scorer(scoring_function_log_loss, greater_is_better = False)
def tune_LogReg_model(X_train, y_train): 

    C_list = [0.001, 0.01, 0.1, 0.5, 0.75, 1, 5, 10, 25, 100]

    penalty_list = ['l1','l2']



    score = make_scorer(scoring_function_log_loss, greater_is_better = False)

    

    logreg = LogisticRegression()

    

    parameters = {'C':C_list,

                  'penalty': penalty_list}

    tune = GridSearchCV(logreg, parameters, scoring = score)

    tune.fit(X_train,y_train)

    

    return tune
best_mind_model = tune_LogReg_model(X_train, y_train)
best_mind_model.best_params_
mind_model = LogisticRegression(C=best_mind_model.best_params_['C'], penalty = best_mind_model.best_params_['penalty'])

mind_model.fit(X_train, y_train)
y_pred_train = mind_model.predict(X_train)
accuracy_score(y_train, y_pred_train)
y_pred_test = mind_model.predict(X_test)
accuracy_score(y_test, y_pred_test)
confusion_matrix(y_train, y_pred_train)
confusion_matrix(y_test, y_pred_test)
print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))
log_loss(y_train, y_pred_train)
log_loss(y_test, y_pred_test)
mind_log_loss = cross_val_score(mind_model, X, y, scoring=score_log_loss,cv=4,)

print('Log Loss %2f' %(-1 * mind_log_loss.mean()))



mind_acc = cross_val_score(mind_model, X, y, scoring='accuracy',cv=4,)

print('Accuracy %2f' %(mind_acc.mean()))
energy_df = train[['lemma','N']]
vect_energy = TfidfVectorizer(lowercase=True, 

                            stop_words='english', 

                            max_features=195,

                            min_df=4,

                            max_df=0.5

                           )

vect_energy.fit(energy_df['lemma'])

X_count_energy = vect_energy.transform(energy_df['lemma'])



X_count_energy.shape



vect_energy.get_feature_names()
X = X_count_energy

y = energy_df['N']

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y,

                                                    test_size =0.3,

                                                   random_state = 42)
best_energy_model = tune_LogReg_model(X_train, y_train)
best_energy_model.best_params_
energy_model = LogisticRegression(C=best_energy_model.best_params_['C'], penalty = best_energy_model.best_params_['penalty'])

energy_model.fit(X_train, y_train)
y_pred_train = energy_model.predict(X_train)



accuracy_score(y_train, y_pred_train)
y_pred_test = energy_model.predict(X_test)



accuracy_score(y_test, y_pred_test)
confusion_matrix(y_train, y_pred_train)
confusion_matrix(y_test, y_pred_test)
print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))
log_loss(y_train, y_pred_train)
log_loss(y_test, y_pred_test)
energy_log_loss = cross_val_score(energy_model, X, y, scoring=score_log_loss,cv=4)

print('Log Loss %2f' %(-1 * energy_log_loss.mean()))



energy_acc = cross_val_score(energy_model, X, y, scoring='accuracy',cv=4,)

print('Accuracy %2f' %(energy_acc.mean()))
nature_df = train[['lemma','T']]
vect_nature = TfidfVectorizer(lowercase=True, 

                            stop_words='english', 

                            max_features=3900,

                            min_df=4,

                            max_df=0.5

                            #ngram_range=(3,3)

                           )

vect_nature.fit(nature_df['lemma'])

X_count_nature = vect_nature.transform(nature_df['lemma'])



X_count_nature.shape



vect_nature.get_feature_names()
X = X_count_nature

y = nature_df['T']

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y,

                                                    test_size =0.3,

                                                   random_state = 42)
best_nature_model = tune_LogReg_model(X_train, y_train)
best_nature_model.best_params_
nature_model = LogisticRegression(C=best_nature_model.best_params_['C'], penalty = best_nature_model.best_params_['penalty'])

nature_model.fit(X_train, y_train)
y_pred_train = nature_model.predict(X_train)



accuracy_score(y_train, y_pred_train)
y_pred_test = nature_model.predict(X_test)



accuracy_score(y_test, y_pred_test)
confusion_matrix(y_train, y_pred_train)
confusion_matrix(y_test, y_pred_test)
print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))
log_loss(y_train, y_pred_train)
log_loss(y_test, y_pred_test)
nature_log_loss = cross_val_score(nature_model, X, y, scoring=score_log_loss,cv=4,)

print('Log Loss %2f' %(-1 * nature_log_loss.mean()))



nature_acc = cross_val_score(nature_model, X, y, scoring='accuracy',cv=4,)

print('Accuracy %2f' %(nature_acc.mean()))
tactics_df = train[['lemma','J']]
vect_tactics = TfidfVectorizer(lowercase=True, 

                            stop_words='english', 

                            max_features=260,

                            min_df=4,

                            max_df=0.5

                           )

vect_tactics.fit(tactics_df['lemma'])

X_count_tactics = vect_tactics.transform(tactics_df['lemma'])



X_count_tactics.shape



vect_tactics.get_feature_names()
X = X_count_tactics

y = tactics_df['J']

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y,

                                                    test_size =0.3,

                                                   random_state = 42)
best_tactics_model = tune_LogReg_model(X_train, y_train)
best_tactics_model.best_params_
tactics_model = LogisticRegression(C=best_tactics_model.best_params_['C'], penalty = best_tactics_model.best_params_['penalty'])

tactics_model.fit(X_train, y_train)
y_pred_train = tactics_model.predict(X_train)



accuracy_score(y_train, y_pred_train)
y_pred_test = tactics_model.predict(X_test)



accuracy_score(y_test, y_pred_test)
confusion_matrix(y_train, y_pred_train)
confusion_matrix(y_test, y_pred_test)
print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))
log_loss(y_train, y_pred_train)
log_loss(y_test, y_pred_test)
tactics_log_loss = cross_val_score(tactics_model, X, y, scoring=score_log_loss,cv=4,)

print('Log Loss %2f' %(-1 * tactics_log_loss.mean()))



tactics_acc = cross_val_score(tactics_model, X, y, scoring='accuracy',cv=4,)

print('Accuracy %2f' %(tactics_acc.mean()))
test.head()
pred_mind_count = vect_mind.transform(test['lemma_no_stop'])



pred_mind_count.shape



X = X_count_mind

y = mind_df['E']



final_mind_model = mind_model

final_mind_model.fit(X, y)



final_mind_predictions = final_mind_model.predict(pred_mind_count)



test['E_pred'] = final_mind_predictions



test.head()



pred_mind_df = test[['id', 'E_pred']]



pred_mind_df.head(10)



pred_mind_df.columns



pred_mind_df['E_pred'].value_counts().plot(kind = 'bar',color = ['darkblue','dodgerblue'])



#pred_mind_df









plt.show()



pred_mind_df.head(10)
pred_energy_count = vect_energy.transform(test['lemma_no_stop'])



pred_energy_count.shape



X = X_count_energy

y = energy_df['N']



final_energy_model = energy_model

final_energy_model.fit(X, y)



final_energy_predictions = final_energy_model.predict(pred_energy_count)



test['N_pred'] = final_energy_predictions



pred_energy_df = test[['id', 'N_pred']]



pred_energy_df['N_pred'].value_counts().plot(kind = 'bar', color = ['purple','violet'])

plt.show()



pred_energy_df.head(10)
pred_nature_count = vect_nature.transform(test['lemma_no_stop'])



pred_nature_count.shape



X = X_count_nature

y = nature_df['T']



final_nature_model = nature_model

final_nature_model.fit(X, y)



final_nature_predictions = final_nature_model.predict(pred_nature_count)



test['T_pred'] = final_nature_predictions



pred_nature_df = test[['id', 'T_pred']]



pred_nature_df['T_pred'].value_counts().plot(kind = 'bar', color = ['darkgreen','yellowgreen'])

plt.show()



pred_nature_df.head(10)
pred_tactics_count = vect_tactics.transform(test['lemma_no_stop'])



pred_tactics_count.shape



X = X_count_tactics

y = tactics_df['J']



final_tactics_model = tactics_model

final_tactics_model.fit(X, y)



final_tactics_predictions = final_tactics_model.predict(pred_tactics_count)



test['J_pred'] = final_tactics_predictions



pred_tactics_df = test[['id', 'J_pred']]



pred_tactics_df['J_pred'].value_counts().plot(kind = 'bar', color = ['black','grey'])





plt.show()



pred_tactics_df.head(10)
my_submission = pd.merge(pred_mind_df[['id','E_pred']], pred_energy_df[['id','N_pred']], how ='inner', on ='id') 

my_submission = pd.merge(my_submission[['id','E_pred', 'N_pred']], pred_nature_df[['id','T_pred']], how ='inner', on ='id')

my_submission = pd.merge(my_submission[['id','E_pred', 'N_pred','T_pred']], pred_tactics_df[['id','J_pred']], how ='inner', on ='id') 
my_submission.head(10)
my_submission.rename(columns={'id':'id',

                            'E_pred':'mind',

                            'N_pred': 'energy',

                            'T_pred': 'nature',

                            'J_pred': 'tactics'

                             }, 

                 inplace=True)



my_submission.head()
my_submission.to_csv('Classification_project_final_submission.csv', index=False)
# Create column for the predictions of each of the 4 chracteristics

my_submission['Mind Pred'] = my_submission['mind'].map(lambda x: 'E' if x == 1 else 'I')

my_submission['Energy Pred'] = my_submission['energy'].map(lambda x: 'N' if x == 1 else 'S')

my_submission['Nature Pred'] = my_submission['nature'].map(lambda x: 'T' if x == 1 else 'F')

my_submission['Tactics Pred'] = my_submission['tactics'].map(lambda x: 'J' if x == 1 else 'P')
my_submission.head()
my_submission['Personality Pred'] = my_submission['Mind Pred'] + my_submission['Energy Pred'] + my_submission['Nature Pred']+ my_submission['Tactics Pred']
my_submission[['id','Personality Pred']].head()
mbti_type = ['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP','INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP']

global_p = [ 2.5, 8.1, 1.8, 3.2, 12.3, 8.5, 8.7, 4.3, 1.5, 4.4, 2.1, 3.3, 13.8, 8.8, 11.6, 5.4]
global_types = {'Type':mbti_type,'Percentage':global_p}
global_df = pd.DataFrame(global_types)

global_df
#view posts count of each personality type

# Countplot of the 16 personality types in the dataset

dims1 = (15.0, 4.0)

fig, ax = plt.subplots(figsize=dims1)

cmrmap = sns.color_palette('CMRmap', 16)

sns.set_palette(cmrmap)

sns.countplot(x='type', data=train,\

              order=['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP',\

                    'INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP'])

plt.title('Distribution of Myers-Briggs Types in the test Dataset', fontsize=16)

plt.xlabel('Personality Type')

plt.ylabel('Count of Posts')

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()







#view posts count of each personality type

# Countplot of the 16 personality types in the dataset

dims1 = (15.0, 4.0)

fig, ax = plt.subplots(figsize=dims1)

cmrmap = sns.color_palette("CMRmap", 16)

sns.set_palette(cmrmap)

sns.countplot(x='Personality Pred', data=my_submission,\

              order=['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP',\

                    'INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP'])

plt.title('Distribution of Myers-Briggs Type Predictions', fontsize=16)

plt.xlabel('Personality Type')

plt.ylabel('Count of Posts')

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()







#view posts count of each personality type

# Countplot of the 16 personality types in the dataset

dims1 = (15.0, 4.0)

fig, ax = plt.subplots(figsize=dims1)

cmrmap = sns.color_palette("CMRmap", 16)

sns.set_palette(cmrmap)

sns.barplot(x='Type', y='Percentage', data=global_df, order=['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP',\

                   'INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP'])

plt.title('Approximate Distribution of Global Myers-Briggs Personality Types', fontsize=16)

plt.xlabel('Personality Type')

plt.ylabel('Percentage')

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()




