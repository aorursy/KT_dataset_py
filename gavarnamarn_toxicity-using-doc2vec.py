import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
train = pd.read_csv('../input/train.csv')[0:1000]
test = pd.read_csv('../input/test.csv')#[0:1000]
subm = pd.read_csv('../input/sample_submission.csv')#[0:1000]
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1)
alldata = np.concatenate((np.asarray(train['comment_text']), np.asarray(test['comment_text'])), axis=0)
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(alldata)]
tagged_data[0]
max_epochs = 10 #change later
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

#model.save("d2v.model")
#print("Model Saved")
print(model.docvecs['3'])
ntrain = len(train['id']) 
ntest = len(test['id'])

X_train = np.zeros((ntrain, vec_size))
for i in range(ntrain):
    X_train[i,:] = model.docvecs[i]
    
X_test = np.zeros((ntest, vec_size))
for i in range(ntest):
    X_test[i,:] = model.docvecs[ntrain+i]
    
y_train = train[label_cols].values
y_train
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

#classifier = BinaryRelevance(classifier = SVC(probability=True), require_dense = [False, True])
classifier = BinaryRelevance(classifier = MLPClassifier(max_iter=1000), require_dense = [False, True])

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict_proba(X_test).toarray()
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(predictions, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

