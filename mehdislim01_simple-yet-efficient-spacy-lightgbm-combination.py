import numpy as np

import pandas as pd

from lightgbm import LGBMClassifier

from tqdm import tqdm
import spacy

spacy_model = spacy.load('en_core_web_lg') # 'lg' means large and 'en' means english so we are saying import the large spacy english model

#there is small model too just change 'lg' to 'sm', note that the large model vectorize the text to 300d whereas the small model vectorize the text to 96d
train = pd.read_csv('../input/nlp-getting-started/train.csv') #load the data

test = pd.read_csv('../input/nlp-getting-started/test.csv')

sb = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
spacy_vectorizer = lambda text: spacy_model(text).vector # vectorize the data

vectorized_train_documents = []

for i in tqdm(range(train.shape[0])):

    vectorized_train_documents.append(spacy_vectorizer(train.iloc[i].text.lower()))

print('Vectorizing the Training documents is DONE!')



vectorized_test_documents = []

for i in tqdm(range(test.shape[0])):

    vectorized_test_documents.append(spacy_vectorizer(test.iloc[i].text.lower()))

print('Vectorizing the Testing documents is DONE!')
xtrain_spacy = np.array(vectorized_train_documents) #put the data in the right format

xtest_spacy = np.array(vectorized_test_documents)

ytrain = train.target.values.reshape(-1, 1)
lgbmc = LGBMClassifier(learning_rate=.05, n_estimators=150,) # train the model

lgbmc.fit(xtrain_spacy, ytrain)
import os #save the predictions on the test set

os.chdir('/kaggle/working/')

sb['target'] = lgbmc.predict(xtest_spacy)

sb.to_csv('SubmitMe!.csv', index=False)