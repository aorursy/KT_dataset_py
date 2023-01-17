# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model, metrics
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import gensim
from gensim.models import Word2Vec, FastText
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import itertools
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
warnings.filterwarnings(action='ignore') 
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''     
train = pd.read_csv('/kaggle/input/ofeofa63fv3a36k/train.csv')
test = pd.read_csv('/kaggle/input/ofeofa63fv3a36k/test.csv')
sample = pd.read_csv('/kaggle/input/ofeofa63fv3a36k/sample_submission.csv')


lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.label.values)

xtrain1, xvalid1, xtrain2, xvalid2, ytrain, yvalid = train_test_split(train.sentence1.values, train.sentence2.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

opt_embedding = "w2v"
submission = False
classifications = ["xgb", "log" "svc"]
method = "wordDis"

if False and os.path.isfile('embedding.model'):
    embedding = Word2Vec.load('embedding.model')
else:
    sent1 = [row.split() for row in train.sentence1.values]
    sent2 = [row.split() for row in train.sentence2.values]
    sent3 = [row.split() for row in test.sentence1.values]
    sent4 = [row.split() for row in test.sentence2.values]
    
    sentences = sent1 + sent2     
    freq = Counter(list(itertools.chain(*sentences)))
    
    print("Creating embedding/frequency")
    
    if opt_embedding == "w2v":
        embedding = Word2Vec(min_count=1,
                     window=3,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20)
    elif opt_embedding == "fast":
        embedding = FastText(size=4, window=3, min_count=1)  # instantiate
    
    embedding.build_vocab(sentences)
    embedding.train(sentences, total_examples=embedding.corpus_count, epochs=30)  # train
    embedding.init_sims(replace=True)
    embedding.save('embedding.model')
    
    print("Done")


def avgCos (sentences1, sentences2): 
    sims = []
    
    for (sent1, sent2) in zip(sentences1, sentences2):
        
        sent1 = sent1.split()
        sent2 = sent2.split()

        sent1 = [token for token in sent1 if token in embedding]
        sent2 = [token for token in sent2 if token in embedding]

        if len(sent1) == 0 or len(sent2) == 0:
            sims.append(0)
            continue    
        
        weights1 = None
        weights2 = None
        
        embedding1 = np.average([embedding.wv[token] for token in sent1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([embedding.wv[token] for token in sent2], axis=0, weights=weights2).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]

        sims.append(sim)
    
    return sims


def wordDistance (sentences1, sentences2):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        sent1 = sent1.split()
        sent2 = sent2.split()

        sent1 = [token for token in sent1 if token in embedding]
        sent2 = [token for token in sent2 if token in embedding]

        if len(sent1) == 0 or len(sent2) == 0:
            sims.append(0)
            continue
            
        sims.append(-embedding.wmdistance(sent1, sent2))
        
    return sims

def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX

def sifCos (sentences1, sentences2, freqs=freq, a=0.001):
    total_freq = sum(freqs.values())
    embeddings = []        
    # SIF requires us to first collect all sentence embeddings and then perform 
    # common component analysis.
    for (sent1, sent2) in zip(sentences1, sentences2): 
        sent1 = sent1.split()
        sent2 = sent2.split()
        
        sent1 = [token for token in sent1 if token in embedding]
        sent2 = [token for token in sent2 if token in embedding]
        
        weights1 = [a/(a+freqs.get(token)/total_freq) for token in sent1]
        weights2 = [a/(a+freqs.get(token)/total_freq) for token in sent2]

        embedding1 = np.average([w2v_model.wv[token] for token in sent1], axis=0, weights=weights1)
        embedding2 = np.average([w2v_model.wv[token] for token in sent2], axis=0, weights=weights2)
        
        embeddings.append(embedding1)
        embeddings.append(embedding2)
        
    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx*2].reshape(1, -1), 
                              embeddings[idx*2+1].reshape(1, -1))[0][0] 
            for idx in range(int(len(embeddings)/2))]

    return sims

if method == "avgCos":
    sims_train = avgCos(xtrain1, xtrain2)
    sims_valid = avgCos(xvalid1, xvalid2)
    sims_test = avgCos(test["sentence1"], test["sentence2"])
elif method == "wordDis":
    sims_train = wordDistance(xtrain1, xtrain2)
    sims_valid = wordDistance(xvalid1, xvalid2)
    sims_test = wordDistance(test["sentence1"], test["sentence2"])
else:
    sims_train = avgCos(xtrain1, xtrain2)
    sims_valid = avgCos(xvalid1, xvalid2)
    sims_test = avgCos(test["sentence1"], test["sentence2"])

sims_train = np.array(sims_train).reshape(len(sims_train), 1)
sims_valid = np.array(sims_valid).reshape(len(sims_valid), 1)
sims_test = np.array(sims_test).reshape(len(sims_test), 1)
model = LinearSVC()

for classification in classifications:
    if classification == "log":
        model = linear_model.LogisticRegression()
    elif classification == "svc":
        model = LinearSVC()
    elif classification == "knear":
        model = KNeighborsClassifier(n_neighbors = 3)
    elif classification == "randomTree":
        model = RandomForestClassifier(n_estimators=100)
    elif classification == "xgb":
        model = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
    x_train, y_train = (sims_train, ytrain)
    x_valid, y_valid = (sims_valid, yvalid)
    print(classification)
    model.fit(x_train, y_train)
    print(model.predict(x_train))
    print(round(model.score(x_train, y_train) * 100, 2))
    print(round(model.score(x_valid, y_valid) * 100, 2))
    #print(cross_val_score(model, x_train, y_train, cv = 3))

if submission:
    Y_pred = model.predict(sims_test)
    submission = pd.DataFrame({
            "id": test["id"],
            "label": Y_pred
        })
    submission.to_csv('submission.csv', index=False)
    print("submission file created!")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session