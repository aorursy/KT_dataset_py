import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
!pip install -q pymagnitude #tensorflow keras
#glove: !curl -s http://magnitude.plasticity.ai/glove+subword/glove.6B.50d.magnitude --output vectors.magnitude
#word2vec: !curl -s http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude --output vectors.magnitude
#fastText:  !curl -s http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude --output vectors.magnitude
#elmo light: !curl -s http://magnitude.plasticity.ai/elmo/light/elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude --output vectors.magnitude
!curl -s http://magnitude.plasticity.ai/elmo/light/elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude --output vectors.magnitude
from pymagnitude import *
from tqdm import notebook
MAX_WORDS = 30 # The maximum number of words the sequence model will consider
vectors = Magnitude('./vectors.magnitude', pad_to_length = MAX_WORDS)
df=pd.read_csv('/kaggle/input/sentiment-analysis-for-financial-news/FinancialPhraseBank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt',encoding = "ISO-8859-1", names=['text','sentiment'], delimiter= '@')
df.head(2)
def avg_vec(df):
    vctrLs = []
    for txt in notebook.tqdm(df.text.values): vctrLs.append(np.average(vectors.query(txt.split(' ')), axis = 0))
    return np.array(vctrLs)
train=df.sample(frac=0.8,random_state=42)
test=df.drop(train.index)
xTrn,xTest=avg_vec(train),avg_vec(test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
forest = RandomForestClassifier(n_estimators=100, random_state=0, max_features=0.5, 
                                max_depth=4 ,min_samples_split=5,
                                oob_score=True, n_jobs=-1, min_samples_leaf=50)
def oneHot(row):
  if row=='negative': return -1
  if row=='neutral' : return  0
  if row=='positive' : return +1
train.sentiment,test.sentiment=train.sentiment.apply(oneHot),test.sentiment.apply(oneHot)
forest.fit(xTrn, train.sentiment)
print("Accuracy on training set: {:.3f}".format(forest.score(xTrn, train.sentiment)))
oldscore = forest.oob_score_
print(f'OOB score is {oldscore*100:.1f}%')
#print('Out-of-bag score estimate: {:.3}'.format())
y_predict = forest.predict(xTest)
confusion_matrix(test.sentiment, y_predict)


cm = confusion_matrix(test.sentiment, y_predict)
print("Confusion matrix:\n{}".format(cm))


#Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("Accuracy on test set: {:.3f}".format(forest.score(xTest, test.sentiment)))
test.text.values[0]
x=[np.average(vectors.query(test.text.values[0].split(' ')), axis = 0)]
forest.predict(x)
test.sentiment.values[0]