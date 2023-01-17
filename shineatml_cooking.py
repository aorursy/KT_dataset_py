# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
df_train = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')
df_train.head()
#利用词袋对特征进行处理
df_train_ing = df_train.ingredients
df_test_ing = df_test.ingredients

word_train = [' '.join(x) for x in df_train_ing]
word_test = [' '.join(x) for x in df_test_ing]
tfidfvec = TfidfVectorizer(binary=True)
X_train = tfidfvec.fit_transform(word_train)
X_test = tfidfvec.transform(word_test)
Y_train = df_train.cuisine
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.85)
RF = RandomForestClassifier(n_estimators=900, oob_score=True)
RF.fit(X_train,Y_train)
RF.oob_score_, RF.score(X_val, Y_val)
Y_test = RF.predict(X_test)
Y_test
df_output = pd.DataFrame(np.array([df_test.id, Y_test]).T, columns=['id','cuisine']).set_index('id')
df_output.head()
df_output.to_csv('output_RF.csv')