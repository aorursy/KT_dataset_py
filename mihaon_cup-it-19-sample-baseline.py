import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

%matplotlib inline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



from nltk.corpus import stopwords



from wordcloud import WordCloud

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train_data.csv', encoding='utf', engine='python', index_col=0)

test = pd.read_csv('../input/test_data.csv', encoding='utf', engine='python', index_col=0)
train.score.value_counts().plot.bar()
tf_idf = TfidfVectorizer(ngram_range=(1, 4), stop_words=stopwords.words('russian'), 

                         #tokenizer=None,

                         #preprocessor=None,

                         analyzer='word',

                         max_df=0.8, 

                         min_df=10,

                         #max_features=10000

                        )
%%time

tf_idf_model = tf_idf.fit(np.concatenate([train['text'], test['text']]))
%%time

train_tf_idf_vec = tf_idf_model.transform(train['text'])

test_tf_idf_vec = tf_idf_model.transform(test['text'])
wordcloud = WordCloud().generate_from_frequencies(tf_idf_model.vocabulary_)



# Display the generated image:

plt.figure() 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0) 

plt.tight_layout()

plt.show() 
lm = LogisticRegression(#solver='newton-cg', 

                        #n_jobs=-1,

                        #solver='lbfgs',

                        penalty='l2',

                        #tol=0.000000001,

                        random_state=42,

                        C=10, 

                        max_iter=100000)
lm_params = {'penalty':['l1', 'l2'],

             'C':[0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100],

             #'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

             #'tol' : [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0001]

    

    

}

lm_search = GridSearchCV(estimator=lm, 

                         param_grid=lm_params, 

                         scoring ='roc_auc', 

                         cv=StratifiedKFold(10), 

                         n_jobs=-1,

                         verbose=1)
%%time

lm_search_fitted = lm_search.fit(X=train_tf_idf_vec, y=pd.factorize(train.score)[0])
lm_search_fitted.best_estimator_
pred_scores = cross_val_score(estimator=lm_search_fitted.best_estimator_, X=train_tf_idf_vec, y=pd.factorize(train.score)[0],

                scoring='roc_auc',  

                cv=10, #stratified by default

                n_jobs=-1)

display(np.mean(pred_scores))
predicts = lm_search_fitted.best_estimator_.predict_proba(test_tf_idf_vec)[:, 0]
sub = pd.DataFrame({'index': range(0, len(predicts)),

                    'score':predicts})

sub.to_csv('LRandTFIDF_sample_submission.csv', index=False)
pd.read_csv('LRandTFIDF_sample_submission.csv').head()