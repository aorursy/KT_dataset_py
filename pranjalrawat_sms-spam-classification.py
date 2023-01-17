# PRELIMINARIES
'''



link: https://www.kaggle.com/uciml/sms-spam-collection-dataset/home



dataset description: The SMS Spam Collection is a set of SMS 

tagged messages that have been collected for SMS Spam research.

It contains one set of SMS messages in English of 5,574 messages,

tagged acording being ham (legitimate) or spam.



The files contain one message per line. 

Each line is composed by two columns: 

v1 contains the label (ham or spam) 

and v2 contains the raw text.



'''
import numpy as np, pandas as pd

raw_data = pd.read_csv('../input/spam.csv', encoding = "ISO-8859-1")
raw_data['response'] = 0

raw_data.loc[raw_data['v1'] == 'spam', 'response'] = 1

data = raw_data[['response', 'v2']]

data.columns = ['response', 'text']             
# EXPLORATORY DATA ANALYSIS
pd.set_option('display.max_colwidth', -1)

data.head()
# Event Rate - how many spam and how many ham? 

data.response.value_counts()
# Check for Nulls

data.isnull().sum()
# Check Data Types
data['text'].astype('str')

data.dtypes
# FEATURE ENGINEERING
# length of text

data['text_len'] = data['text'].str.len()
# number of words

data['text_tokens'] = data['text'].apply(lambda x: len(str(x).split(" ")))
# average word length

def avg_word(sentence):

    words = sentence.split()

    return (sum(len(word) for word in words)/len(words))



data['text_avg_word_len'] = data['text'].apply(lambda x: avg_word(str(x)))
# number of stop words/fillers (a, an, the...)

from nltk.corpus import stopwords

stop = stopwords.words('english')

stop = stopwords.words('english')

data['text_stop_words'] = data['text'].apply(lambda x: len([x for x in str(x).split() if str(x) in stop]))
# number of 'spamy'/suspicious words

data['text_keywords'] = data['text'].apply(lambda x: len([x for x in x.split() if x.lower() in ('free', 'win', 'won', 'exclusive', 'enroll', 'discount', 'prize', 'million')]))
# number of numeric characters

data['text_numerics'] = data['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
# number of titled words

data['text_titles'] = data['text'].apply(lambda x: len([x for x in x.split() if x.istitle()]))
# TRAIN-TEST DATA SPLIT
data_clean = data.drop(['text'], axis = 1)

data_clean = data_clean.dropna()

print(data_clean.dtypes)

y = data_clean['response'].astype('int')

X = data_clean.drop('response', axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=19)
from catboost import Pool, CatBoostClassifier

cat_feature_index = np.where(X.dtypes == 'object')[0]

train_pool = Pool(X_train, y_train, cat_features = cat_feature_index)

test_pool = Pool(X_test, y_test, cat_features = cat_feature_index)
from catboost import Pool, CatBoostClassifier



cat_model = CatBoostClassifier(

    depth = 6,

    random_seed = 3, 

    learning_rate = 0.1, 

    eval_metric = 'AUC',

    #iterations = 500,

    verbose = True,

    loss_function= 'Logloss',

    od_type='Iter', # overfitting detector - by iterations

    od_wait=50 # prevent overfitting by ending training after 1 rounds without best value

     )



cat_model.fit(

    train_pool,

    eval_set = test_pool, 

    use_best_model = True

    )
#CAT FEATURE IMPORTANCE



feature_importance = cat_model.get_feature_importance(train_pool)

feature_names = X_train.columns

feature_imp = pd.DataFrame([feature_names, feature_importance])

final = feature_imp.transpose()

final.sort_values(by = 1, ascending = False, inplace = True)

pd.set_option('display.max_colwidth', -1)

final.head(10)
# CAT MODEL EVALUATION



# CAT PREDICTIONS

cat_predictions_probs = cat_model.predict_proba(test_pool)

cat_predictions = np.where(cat_predictions_probs[:,1] > 0.5, 1, 0)

print(cat_predictions[:5]) # predicted class

print(cat_predictions_probs[:5]) # probability scores



print('CAT MODEL EVALUATION')

print(y.describe())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

print('\nAccuracy: ', str(accuracy_score(y_test, cat_predictions)))

print('Precision: ', str(precision_score(y_test, cat_predictions)))

print('Recall: ', str(recall_score(y_test, cat_predictions)))

print('F1: ', str(f1_score(y_test, cat_predictions)))

print('Area under ROC Curve: ', str(roc_auc_score(y_test, cat_predictions_probs[:,1])))

print('GINI: ', str(-1 + 2*roc_auc_score(y_test, cat_predictions_probs[:,1])))



tn, fp, fn, tp = confusion_matrix(y_test, cat_predictions).ravel()



print('True Negatives: ', str(tn))

print('True Positives: ', str(tp))

print('False Negatives: ', str(fn))

print('False Positives: ', str(fp))



print('\nTotal SMS: ', str(tn+fp+fn+tp))

print('No. of SMS the Model Declares as Spam: ', str(fp+tp))

print('No. of SMS that were actually SPAM: ', str(tp+fn))

print('No. of Spam SMS caught by Model: ', str(tp))



print('\nProportion of SMS Declared as Spam: ', str((fp+tp)/(tn+fp+fn+tp)))

print('Proportion of Spam SMS Caught by Model: ', str(tp/(tp+fn)))