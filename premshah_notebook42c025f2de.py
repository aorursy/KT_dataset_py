# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data.head()
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))

data['len_q2'] = data.question2.apply(lambda x: len(str(x)))

data['diff_len'] = data.len_q1 - data.len_q2

data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ','')))))

data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ','')))))

data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))

data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))

data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)
#from fuzzywuzzy import fuzz



#data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']),str(x['question2'])),axis=1)

#data['fuzz_wratio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']),str(x['question2'])),axis=1)

#data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']),str(x['question2'])),axis=1)

#data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']),str(x['question2'])),axis=1)

#data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']),str(x['question2'])),axis=1)

#data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']),str(x['question2'])),axis=1)

#data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']),str(x['question2'])),axis=1)

#data.head()
train, test = train_test_split(data, test_size = 0.2)

x_test = test

x_train = train

y_test = test['is_duplicate']

y_train = train['is_duplicate']

x_train.drop(['id','qid1','qid2','question1','question2','is_duplicate'],axis=1,inplace=True)

x_test.drop(['id','qid1','qid2','question1','question2','is_duplicate'],axis=1,inplace=True)
model = RandomForestClassifier(100,oob_score=True)

model.fit(x_train,y_train)

model.score(x_test,y_test)
data = pd.read_csv("../input/test.csv")
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))

data['len_q2'] = data.question2.apply(lambda x: len(str(x)))

data['diff_len'] = data.len_q1 - data.len_q2

data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ','')))))

data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ','')))))

data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))

data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))

data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)

data.head()
test = data.drop(['test_id','question1','question2'],axis=1)

pdt = model.predict(test);

pdt
import pandas as pd

sub = pd.DataFrame()

sub['test_id'] = data['test_id']

sub['is_duplicate'] = pdt

sub.to_csv('simple_xgb.csv', index=False)