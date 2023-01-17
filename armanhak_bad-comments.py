import numpy as np

import pandas as pd

# train = pd.read_csv('../input/train.csv').sample(frac=0.1).reset_index(drop=True)

# test = pd.read_csv('../input/test.csv').sample(frac=0.1).reset_index(drop=True)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# train = return_preproc_data(train)

# test = return_preproc_data(test)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack



class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 

train_text = train['comment_text']

test_text = test['comment_text']

all_text = pd.concat([train_text, test_text])



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)

test_word_features = word_vectorizer.transform(test_text)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 5),#2-6

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)

test_char_features = char_vectorizer.transform(test_text)



train_features = hstack([train_char_features, train_word_features])

test_features = hstack([test_char_features, test_word_features])



from sklearn.svm import LinearSVC

from sklearn.ensemble import AdaBoostClassifier

# import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV

scores = []

submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:

    train_target = train[class_name]

    classifier = LogisticRegression(solver='sag')

#     svm = LinearSVC(dual=False)

#     classifier = AdaBoostClassifier(cv, algorithm='SAMME', n_estimators=100)

#     svm = LinearSVC()

#     classifier = CalibratedClassifierCV(svm) 

#     classifier = xgb.XGBClassifier(objective='multi:softprob', learning_rate=1,

#                                    tree_method='auto',

#                                    max_depth = 12,    

#                                    silent=True, 

#                                    n_estimators=100, 

#                                    num_class=2, )



#     print('class')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))

    scores.append(cv_score)

    print('CV score for class {} is {}'.format(class_name, cv_score))



    classifier.fit(train_features, train_target)

    submission[class_name] = classifier.predict_proba(test_features)[:, 1]



print('Total CV score is {}'.format(np.mean(scores)))



# # submission.to_csv('submission.csv', index=False)
# def return_preproc_data(data):

#     import string     

#     from nltk.stem import WordNetLemmatizer

#     from nltk.corpus import stopwords

    

#     #Удаление знаки припинания, цифры

#     to_exclude = set(string.punctuation)

#     to_exclude.update(set(string.digits))

#     to_exclude.update("\n")    



#     data['comment_text'] = [''.join([ch for ch in x.strip().lower() if ch not in to_exclude]).strip() for x in data['comment_text']]

#     #Лемматизация 

#     lemmatizer = WordNetLemmatizer()

#     data['comment_text'] = [' '.join([lemmatizer.lemmatize(lemma) for lemma in x.split()]) for x in data['comment_text']]

#     #Удалим стоп слова 

#     stopWords = set(stopwords.words('english'))

#     data['comment_text'] = [' '.join([word for word in x.split() if word not in stopWords]) for x in data['comment_text']]



#     #удалим слова, у которых длина больше 13

#     data['comment_text'] = [' '.join([word for word in x.split() if len(word) <14]) for x in data['comment_text']]



# #     Выполним стемминг(приведение к начальной форме)

#     from nltk.stem import PorterStemmer

#     ps = PorterStemmer()

#     data['comment_text'] = [' '.join([ps.stem(word) for word in x.split()]) for x in data['comment_text']]

#     return data

# #     Токенизация комментариев 

# #     from keras.preprocessing.text import Tokenizer

# #     tokenizer = Tokenizer()

# #     tokenizer.fit_on_texts(data['comment_text'])

# #     comment_matrix = tokenizer.texts_to_matrix(data['comment_text'])
