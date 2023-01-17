# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
train_data=pd.read_csv("/kaggle/input/fakenews/train.csv")
test_data=pd.read_csv("/kaggle/input/fakenews/test.csv")
score_data=pd.read_csv("/kaggle/input/fakenews/submit.csv")
test_data['label']='t'
# replace the null with " " 
test_data=test_data.fillna(" ")
train_data=train_data.fillna(" ")
# conbine the title, auther and text to get single field  
test_data['total']=test_data["title"]+" "+test_data['author']+" "+test_data["text"]
train_data['total']=train_data['title']+" "+train_data["author"]+" "+train_data["text"]
count_vec=CountVectorizer(ngram_range=(1,2))
counts=count_vec.fit_transform(train_data['total'])
trasformer=TfidfTransformer(smooth_idf=False) # allow 0 division

tfidf=trasformer.fit_transform(counts)
test_count=count_vec.transform(test_data['total'])
test_tfidf=trasformer.fit_transform(test_count)
targets=train_data['label']
X_train,X_test,y_train,y_test=train_test_split(tfidf,targets,random_state=0)
Abad=AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
Abad.fit(X_train,y_train)
print('Accuracy of Adaboost classifier on training set: {:.2f}'
     .format(Abad.score(X_train, y_train)))
print('Accuracy of Adaboost classifier on test set: {:.2f}'
     .format(Abad.score(X_test, y_test))) 
Adab=AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
targets=train_data['label']
Abad.fit(counts,targets)
example_counts = count_vec.transform(test_data['total'])
pre=Abad.predict(example_counts)
preddata=pd.DataFrame(pre,columns=['label'])

preddata['id']=test_data['id']
preddata.groupby('label').count()
score_data.groupby('label').count()

x = np.arange(2)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, [preddata.groupby("label").count().id[0],score_data.groupby("label").count().id[0]], width, label='0')
rects2 = ax.bar(x + width/2, [preddata.groupby("label").count().id[1],score_data.groupby("label").count().id[1]], width, label='1')
ax.set_ylabel('news count')
ax.set_title('compare data between give or prdict')
ax.set_xticks(x)
ax.set_xticklabels(["predict","Given"])
ax.legend()

