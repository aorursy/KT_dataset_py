import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)
print(train.columns)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
tr_ids = train[['id']]
train[class_names] = train[class_names].astype(np.int8)
target = train[class_names]

print("Word vectorizing...")

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
print("Test and train words converted into vectors.")
print("Train text shape after conversion: ", train_word_features.shape)
print("Test text shape after conversion: ", test_word_features.shape)
train_features = train_word_features
test_features = test_word_features
scores = []
scores_classes = np.zeros((len(class_names), 10))

submission = pd.DataFrame.from_dict({'id': test['id']})
submission_oof = train[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

idpred = tr_ids
number_of_folds = 10


from sklearn.model_selection import StratifiedKFold

number_of_folds = 10
kfolder= StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=15)
scores_classes = np.zeros((len(class_names), number_of_folds))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
for j, (class_name) in enumerate(class_names):
    
    #Predict for each class
    print('Class_name : ' + class_name)
    avreal = target[class_name]
    lr_cv_sum = 0
    lr_test_pred = np.zeros(test.shape[0])
    lr_avpred = np.zeros(train.shape[0])
    
    #Perform 10-fold validation
    for i, (train_index, val_index) in enumerate(kfolder.split(train_features, avreal)):
        '''
        print(train_index)
        print(val_index)
        '''
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = target.loc[train_index], target.loc[val_index]

        classifier = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',max_iter=100,normalize=False, random_state=0,  tol=0.0025)
        
        classifier.fit(X_train, y_train[class_name])
        scores_val = classifier.predict(X_val)
        lr_avpred[val_index] = scores_val
        lr_test_pred += classifier.predict(test_features)
        scores_classes[j][i] = roc_auc_score(y_val[class_name], scores_val)
        #print('Fold %02d class %s AUC: %.6f' % ((i+1), class_name, scores_classes[j][i]))
    
    lr_oof_auc = roc_auc_score(avreal, lr_avpred)
    print('\n Average class %s AUC:\t%.6f\n' % (class_name, np.mean(scores_classes[j])))
    submission[class_name] = lr_test_pred / number_of_folds

submission.to_csv('submission.csv', index=False)