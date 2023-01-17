import pandas as pd

raw = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

raw = raw.fillna('')

print(raw.info())

raw
from textutils import cleanup

raw['text_clean'] = cleanup(raw['text'])

raw
X = raw['text_clean']

y = raw['target']



import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print('X_train length', len(X_train))

print('X_test length', len(X_test))

print('y_train length', len(y_train))

print('y_test length', len(y_test))
from naivebayes import NaiveBayes

naiveBayes = NaiveBayes(removeStopWords=True)

naiveBayes.train(X_train, y_train)



# Lets inspect some details from the training

print('vocabulary size', len(naiveBayes.vocabulary))

print('classes', naiveBayes.classes)



# Plot prior probs

import math

print('Prior Probabilities')

for cls in naiveBayes.classes:

    print('{} - {:5.2f}%'.format(cls, math.exp(naiveBayes.logPrior[cls]) * 100))



print('Top 20 Words')

# And top 20 words per class

for cls in naiveBayes.classes:

    print('{}:'.format(cls), naiveBayes.getTopWords(20, cls))
# Now lets predict          

y_predict = naiveBayes.predictAll(X_test)
from sklearn.metrics import confusion_matrix

import seaborn as sn

import matplotlib.pyplot as plt



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_predict)

df_cm = pd.DataFrame(confusionMatrix, columns=np.unique(y_test), index=np.unique(y_test))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

sn.set(font_scale=1.4) # for label size

sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')# font size

plt.show()



# F1 Score

from sklearn.metrics import f1_score

f1_naiveBayes_Stanford = f1_score(y_test, y_predict)

print('f1', f1_naiveBayes_Stanford)