import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
final_out=pd.read_csv("../input/sample_submission.csv")
train_data.head()
combined_df=train_data['comment_text'].append(test_data['comment_text'])
from gensim.models import Word2Vec
Vocab_list=(combined_df.apply(lambda x:str(x).strip().split())
                         )
models=Word2Vec(Vocab_list,size=100)

WordVectorz=dict(zip(models.wv.index2word,models.wv.vectors))
class AverageEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim =100 # because we use 100 embedding points 

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

pipe1=Pipeline([("wordVectz",AverageEmbeddingVectorizer(WordVectorz)),("multilabel",OneVsRestClassifier(LinearSVC(random_state=0)))])
y_train=train_data[[i for i in train_data.columns if i not in ["comment_text","id"]]]
pipe1.fit(train_data['comment_text'],y_train)
predicted1=pipe1.predict(test_data['comment_text'])
label_cols=train_data.columns[2:]
submid = pd.DataFrame({'id': final_out["id"]})
submission = pd.concat([submid, pd.DataFrame(predicted1, columns = label_cols)], axis=1)
submission.to_csv('submission_W2v_m1.csv', index=False)
pipe2=Pipeline([('TFidf',TfidfVectorizer()),("multilabel",OneVsRestClassifier(LinearSVC(random_state=0)))])
pipe2.fit(train_data['comment_text'],y_train)
predicted2=pipe2.predict(test_data['comment_text'])
label_cols=train_data.columns[2:]
submid = pd.DataFrame({'id': final_out["id"]})
submission = pd.concat([submid, pd.DataFrame(predicted2, columns = label_cols)], axis=1)
submission.to_csv('submission_TfIdf_m1.csv', index=False)
