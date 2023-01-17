# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


data = pd.read_csv("../input/train.csv",low_memory=False)
data.target = data.target.apply(lambda x: int(x))

#data = numpy.array(data)  #convert array to numpy type array


df_train ,df_test = train_test_split(data,test_size=0.2)

# Y_train = df_train.iloc[0:, 2].values
# text_train = df_train.iloc[0:, 1].values
# vect = DictVectorizer()
# #vec = CountVectorizer()
# #X_train = vec.fit_transform(text_train)
# #feature_names = np.asarray(vect.get_feature_names())


# Y_test = df_test.iloc[0:, 2].values
# text_test = df_test.iloc[0:, 1].values
from sklearn.feature_extraction.text import TfidfVectorizer
# import xgboost as xgb
from sklearn.linear_model import LogisticRegression

vect = TfidfVectorizer(ngram_range=(1,3), min_df=2)
vect.fit(data.question_text)
#from sklearn.feature_selection import SelectPercentile, chi2
#selection = SelectPercentile(percentile=5, score_func=chi2)
#X_train_selected = selection.fit_transform(vect.transform(df_train.question_text), df_train.target)
# train classifier
clf = LogisticRegression(solver='lbfgs', class_weight={1:8, 0:1}) # 'balanced'
clf.fit(vect.transform(df_train.question_text), df_train.target)#vect.transform(df_train.question_text)
# evaluate
from sklearn.metrics import classification_report
y_preds = clf.predict(vect.transform(df_test.question_text))
evaluation = classification_report(df_test.target, y_preds, digits=3)
print(evaluation)
# conduct test experiments

test_data = pd.read_csv('../input/test.csv')
test_labels = clf.predict(vect.transform(test_data.question_text))
# output the test results
test_data['prediction'] = test_labels
sub_data = test_data[['qid', 'prediction']] 
sub_data.to_csv('./submission.csv', index=None)
