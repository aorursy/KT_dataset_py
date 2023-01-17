# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import spacy
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/train.csv",header=None,low_memory=False)
data_test = pd.read_csv("../input/test.csv",header=None,low_memory=False)
sentences = data[1][1:]
sentences_test = data_test[1][1:]
X = sentences.values
Y = data[2][1:]
Y = Y.values
nlp = spacy.load('en_core_web_lg')
nlp.remove_pipe('ner')
nlp.max_length = 93621305

Y_train = []
sentence_to_vec = []
for i in range(400000):
    ind = np.random.randint(low=0,high=1306122)
    sentence_to_vec.append(nlp(X[ind]).vector)
    Y_train.append(Y[ind])

LR_model = LogisticRegression(solver='lbfgs').fit(sentence_to_vec,Y_train)
    
X_test = []
for a in sentences_test:    
    X_test.append(nlp(a).vector)

preds = LR_model.predict(X_test)
preds = [int(x) for x in preds]
sub = pd.read_csv("../input/sample_submission.csv")
sub["prediction"] = preds
sub.to_csv("submission.csv",index=False)

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
