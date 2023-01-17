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
import pandas as pd

import numpy as np

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
#train 데이터 불러오기



x_y_train=pd.read_csv("/kaggle/input/article-classification-k-yw/new_train1.csv")

x_y_train.head(3)
#데이터 분리

x_data=x_y_train["content"]

y_data=x_y_train["category"]
#data 확인

x_data.head(3)
#test 데이터 불러오기



test=pd.read_csv("/kaggle/input/article-classification-k-yw/x_test1.csv")

test=test["content"] #학습을 위헤 series 형태로 만들어줌

test.head(3)
clf = Pipeline([

    ('vect', TfidfVectorizer()), 

    ('clf', MultinomialNB(alpha=0.01)),

]) #sklearn 에서 제공하는 pipline에 TfidfVectorizer(문서 전처리하는 클래스)와 MultinomialNB는 나이브베이즈 모형 클래스중 다항분포 나이브베이즈를 넣어서 순서대로 처리하게 함
model = clf.fit(x_data.values.astype("str"), y_data) #fit으로 모델 학습
y_pred = model.predict(test) #예측값



y_pred
y_pred=pd.DataFrame(y_pred)



y_pred=y_pred.rename(columns={0:"category"})

y_pred
id=np.array([i for i in range(len(y_pred))]).reshape(-1,1).astype(int)



result=np.hstack([id,y_pred])



df=pd.DataFrame(result,columns=('id','category'))

df.head(3)