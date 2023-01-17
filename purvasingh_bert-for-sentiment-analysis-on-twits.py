# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install transformers
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import torch

import transformers as ppb

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/stockmarket-sentiment-dataset/stock_data.csv', header=None, skiprows=[0])
df.shape
batch_1 = df[:5791]
print(batch_1[0][6])
batch_1[1].value_counts()
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized[:1]
max_len = 0

for i in tokenized.values:

    if len(i)>max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])
padded.shape
attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
input_ids = torch.tensor(padded)

attention_mask = torch.tensor(attention_mask)

with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:, 0, :].numpy()

labels = batch_1[1]
print(features[:10])
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
lr_clf = LogisticRegression()

lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(test_labels, lr_clf.predict(test_features))

fpr, tpr, thresholds = roc_curve(test_labels, lr_clf.predict_proba(test_features)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()