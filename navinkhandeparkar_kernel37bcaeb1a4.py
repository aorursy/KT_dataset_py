# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

from torchtext import data

from sklearn.model_selection import train_test_split





SEED = 1234



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



TEXT = data.Field(tokenize = 'spacy')

LABEL = data.LabelField(dtype = torch.float)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trainData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")
trainData.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from wordcloud import WordCloud, STOPWORDS

stopwords = STOPWORDS.add('amp')



positive_words = ' '.join(trainData[trainData.target == 4].text.values)

negative_words = ' '.join(trainData[trainData.target == 0].text.values)



plt.figure(figsize=(16, 8))



cloud1 = WordCloud(width=400, height=400, background_color='white', stopwords=stopwords).generate(positive_words)

plt.subplot(121)

plt.imshow(cloud1, interpolation="bilinear")

plt.axis("off")

plt.title('positive_words', size=20)



cloud2 = WordCloud(width=400, height=400, background_color='white', stopwords=stopwords).generate(negative_words)

plt.subplot(122)

plt.imshow(cloud2, interpolation="bilinear")

plt.axis("off")

plt.title('Hatred tweets', size=20)

plt.show()
trainData.loc[trainData['target'] == 4, 'target'] = 1
xtrain, xvalid, ytrain, yvalid = train_test_split(trainData.text.values, trainData.target.values, 

                                                  stratify=trainData.target.values, 

                                                  random_state=42, 

                                                  test_size=0.1, shuffle=True)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1,3), stop_words='english')



ctv.fit(list(xtrain)+list(xvalid))

xtrain_ctv = ctv.transform(xtrain)

xvalid_ctv = ctv.transform(xvalid)

def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota
from sklearn.linear_model import LogisticRegression



lf = LogisticRegression(C=1.0)

lf.fit(xtrain_ctv, ytrain)

predict = lf.predict_proba(xvalid_ctv)



print ("logloss: %0.3f " % multiclass_logloss(yvalid, predict))
xtest_ctv = ctv.transform(testData.text.values)

final_predict = lf.predict(xtest_ctv)
submission = pd.DataFrame({"Id":test_data['id'], "target":final_predict})

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)