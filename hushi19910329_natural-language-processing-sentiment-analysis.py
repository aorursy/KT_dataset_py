
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import os
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline
print(os.listdir("../input"))

df_review = [line.rstrip() for line in open('../input/amazon_alexa.tsv')]
print('the samples length is: ',len(df_review))

df_review = pandas.read_csv('../input/amazon_alexa.tsv', sep='\t')
df_review.head()
df_review.describe()
df_review.groupby('rating').describe()
# 抽取评论这一列的数据的长度
df_review['length'] = df_review['verified_reviews'].apply(len)
df_review.head()

df_review['length'].plot(bins=50, kind='hist')
# 取对数以后再看
df_review['loglength'] = df_review['length'].apply(lambda x:np.log(x+1))
df_review['loglength'].plot(bins=50, kind='hist')
df_review[['length','rating']].groupby('rating').describe()
df_review[['loglength','rating']].groupby('rating').describe()
df_review[df_review['length'] == df_review['length'].max()]['verified_reviews'].iloc[0]
df_review.hist(column='loglength', by='feedback', bins=50,figsize=(10,4))
import numpy as np
import matplotlib.pyplot as plt
# 读取数据集
dataset = pd.read_csv('../input/amazon_alexa.tsv', delimiter = '\t', quoting = 3)

# 初步清理单词
import re
import nltk
from nltk.corpus import stopwords # 加载停用词
from nltk.stem.porter import PorterStemmer # 将单词进行还原
corpus=[] # 处理后的单词列表
ps=PorterStemmer() 
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i] )
    corpus.append(' '.join([ps.stem(word) for word in review.lower().split() if not word in set(stopwords.words('english'))]))


# 建立词袋模型
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
cv.fit(corpus)
X=cv.transform(corpus).toarray()
y=dataset.iloc[:,4].values
X = np.array([np.hstack([X[i], df_review['loglength'].values[i]]) for i in range(3150)])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0,stratify = y)

import xgboost as xgb
# 分类
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# 预测
y_pred = classifier.predict(X_test)

# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# 分类报告
from  sklearn.metrics import classification_report as cr
print(cr(y_test, y_pred))
# 在用lightGBM试一下看看
import lightgbm as lgb
clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4,is_unbalance = True
    )
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(cr(y_test, y_pred))
cm2 = confusion_matrix(y_test, y_pred)
cm2
cm
