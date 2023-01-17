# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import expit,logit

# Read the train and test dataset

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

train_text = train['comment_text']
test_text  =test['comment_text']

combined_df = pd.concat([train,test],axis=0)
combined_df.head(n=5)

word_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1,1),max_features=30000)
word_vectorizer.fit(combined_df['comment_text'])

train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='char',ngram_range=(1,3),max_features=50000)
char_vectorizer.fit(combined_df['comment_text'])

train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features_matrix = hstack([train_char_features,train_word_features])
test_features_matrix = hstack([test_char_features,test_word_features])

losses = []

#predictions = {'id',test['id']}

#print(predictions)
columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
preds = np.zeros((test.shape[0],len(columns)))
index  = 0
for column in columns:
    train_target_result = train[column]
    classifier = LogisticRegression(solver='sag')
    cv_loss = np.mean(cross_val_score(classifier,train_features_matrix,train_target_result,cv=4,scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(column,cv_loss))
    classifier.fit(train_features_matrix,train_target_result)
    preds[:,index] = classifier.predict_proba(test_features_matrix)[:,1]
    index = index + 1
    
print('Total CV score is {}'.format(np.mean(losses)))

submission_dataframe = pd.DataFrame({'id':subm['id']})
classification_dataframe = pd.DataFrame(preds,columns=columns)

final_submission_dataframe = pd.concat([submission_dataframe,classification_dataframe],axis=1)
final_submission_dataframe.to_csv('submission.csv',index=False)

# Any results you write to the current directory are saved as output.


