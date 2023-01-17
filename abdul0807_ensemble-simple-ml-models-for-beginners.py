from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mode
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv", index_col=None)
test = pd.read_csv("../input/test.csv", index_col=None)

train.shape, test.shape
X, y = train.drop('target', axis=1), train['target']
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
valid_score = []
test_preds = []
tt = WordPunctTokenizer()

for i, (train_idx, valid_idx) in tqdm(enumerate(splits)):
    start_time = time()
    #print(f'Fold {i+1} started')
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
        
    tf_idf = TfidfVectorizer(tokenizer=tt.tokenize, stop_words='english', ngram_range=(1, 3))
    train_tf = tf_idf.fit_transform(X_train.question_text)
    valid_tf = tf_idf.transform(X_valid.question_text)
    test_tf = tf_idf.transform(test.question_text)
    #print("Completed tf-idf transformation")

    # Model 1
    lr_tf1 = LogisticRegression(C=100)
    lr_tf1.fit(train_tf, y_train)
    #print("Completed Logistic Model 1 training")
    
    # Model 2
    lr_tf2 = LogisticRegression(C=100, solver='saga', max_iter=500, tol=0.001)
    lr_tf2.fit(train_tf, y_train)
    #print("Completed Logistic Model 2 training")

    # Model 3
    nb_tf1 = BernoulliNB(alpha=0.011)
    nb_tf1.fit(train_tf, y_train)
    #print("Completed Naive Bayes Model training")

    pred1 = lr_tf1.predict(valid_tf)
    pred2 = lr_tf2.predict(valid_tf)
    pred3 = nb_tf1.predict(valid_tf)
    valid_pred = mode([pred1, pred2, pred3])[0][0]
    
    validation_f1_score = f1_score(y_valid, valid_pred)
    valid_score.append(validation_f1_score)
    
    pred1 = lr_tf1.predict(test_tf)
    pred2 = lr_tf2.predict(test_tf)
    pred3 = nb_tf1.predict(test_tf)
    test_pred = mode([pred1, pred2, pred3])[0][0]

    test_preds.append(list(test_pred))
    elapsed_time = time() - start_time
    
    print('Fold {} \t val_score={:.4f} \t time={:.2f}s'.format(
            i + 1, validation_f1_score, elapsed_time))
print("Cross validation score is ", np.mean(valid_score))
final_prediction = mode(test_preds)[0][0]
submission = pd.DataFrame()
submission['qid'] = test['qid']
submission['prediction'] = final_prediction
submission.to_csv("submission.csv", index=None)
