# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.feature_extraction.text import  CountVectorizer,TfidfVectorizer

from sklearn import metrics
def processHeadlines(df):

    df['combined_news'] = df.filter(regex=('Top.*')).apply(lambda x:''.join(str(x.values)),axis=1)#横向的值

    return df



def extractFeature(train_df,test_df):

    feature_extraction = TfidfVectorizer()

    train_X = feature_extraction.fit_transform(train_df['combined_news'].values)

    test_X = feature_extraction.transform(test_df["combined_news"].values)

    return train_X,test_X
data = pd.read_csv('../input/Combined_News_DJIA.csv')
data.head(10)
data = pd.read_csv('../input/Combined_News_DJIA.csv')

#分割数据差不多百分之20测试集

data = processHeadlines(data)

train_df = data[data['Date']<'2015-01-01']

test_df = data[data['Date']>'2014-12-31']

#提取

train_X,test_X = extractFeature(train_df,test_df)

train_y = train_df['Label'].values

test_y = test_df['Label'].values
#训练

clf = SVC(probability=True,kernel='rbf')#由于是二分类。我们参数probability设置以后保留概率值不经过sign

clf.fit(train_X,train_y)

predictions = clf.predict_proba(test_X)
print("auc score:",metrics.roc_auc_score(test_y,predictions[:,1]))

print("accuracy score:",metrics.accuracy_score(test_y,clf.predict(test_X)))