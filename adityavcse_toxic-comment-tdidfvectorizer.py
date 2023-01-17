
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
from tqdm import tqdm

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
#Import stringpunctuations 
import string
string.punctuation

punc=string.punctuation


#Import nltk package
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train.head()
test.head()


#Take only the commment column from train and test into a seperate datafram
train_text=train['comment_text']
test_text=test['comment_text']

#Remove stop words

train_text = train['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
test_text = test['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#Convert everything to a single case. Tried converting to lower case but it did not improve the score. 
#Converting to  upper case improved my submission score
train_text=train_text.str.upper()
test_text=test_text.str.upper()


#set the punctuations to a variable
punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~==:'

#Remove punctuations from train and test data
train_text = train_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (punc)]))
test_text = test_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (punc)]))

#Check the data
train_text.head()
test_text.head()



#Cncatenate both train and test data into a single dataframe
all_text=pd.concat([train_text,test_text])


#Call the TFIDFvectorizer package to get the word frequency matrix
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=15000)

#Fit the dataframe
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


losses = []

predictions = {'id': test['id']}

#Loop through each of the class names and do a cross validation with the train data itself
for class_name in tqdm(class_names):
    train_target=train[class_name]
    classifier=LogisticRegression(solver='sag')
    
#    classifier=RandomForestRegressor(n_estimators=10, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
    
    cv_loss=np.mean(cross_val_score(classifier,train_word_features,train_target,cv=5,scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))
    
    classifier.fit(train_word_features, train_target)
    predictions[class_name]=classifier.predict_proba(test_word_features)[:,1]
    

#Print the total cross validation score
print('Total CV score is {}'.format(np.mean(losses)))


#If satisfied with the score make a submission
submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('../output/sample_submission.csv', index=False)

