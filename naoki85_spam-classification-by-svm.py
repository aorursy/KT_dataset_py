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
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

df
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df
df.rename(columns={'v1': 'class', 'v2': 'text'}, inplace=True)

df
df['class_num'] = df['class'].map({'ham':0, 'spam':1})

df
X = pd.DataFrame(df['text'])

y = pd.DataFrame(df['class_num'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)
vec_count = CountVectorizer(min_df=3)

vec_count.fit(X_train['text'])
print('word size: ', len(vec_count.vocabulary_))

print('word content: ', dict(list(vec_count.vocabulary_.items())[0:5]))
X_train_vec = vec_count.transform(X_train['text'])

X_test_vec = vec_count.transform(X_test['text'])
pd.DataFrame(X_train_vec.toarray()[0:5], columns=vec_count.get_feature_names())
pd.DataFrame(X_test_vec.toarray()[0:5], columns=vec_count.get_feature_names())
model = LinearSVC()

model.fit(X_train_vec.toarray(), y_train['class_num'].values)
# 訓練データとテストデータのスコア

print('正解率(train):{:.3f}'.format(model.score(X_train_vec.toarray(), y_train['class_num'].values)))

print('正解率(test):{:.3f}'.format(model.score(X_test_vec, y_test['class_num'].values)))
data = np.array(['I HAVE A DATE ON SUNDAY WITH WILL!!',

                 'Nah I don\'t think he goes to usf, he lives around here though',

                 'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030',

                 'FreeMsg Hey there darling it\'s been 3 week\'s now and no word back! I\'d like some fun you up for it still? Tb ok! XxX std chgs to send, å£1.50 to rcv',

                 'Even my brother is not like to speak with me. They treat me like aids patent.',

                 'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info'])

# ham, ham, spam, spam, ham, spam

df_data = pd.DataFrame(data, columns=['text'])

df_data
input_vec = vec_count.transform(df_data['text'])

p = model.predict(input_vec)

p
list(map(lambda x: 'spam' if x == 1 else 'ham', list(p)))