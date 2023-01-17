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
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer #将原始文档的集合转换为TF-IDF特性的矩阵
from sklearn.feature_selection import chi2 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB #(多项式)朴素贝叶斯
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC #线性支持向量机
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix #混淆矩阵,用于展示两个向量之间的差异
from IPython.display import display
from sklearn import metrics
%matplotlib inline
df = pd.read_csv('../input/consumer_complaints.csv')
df.head()
df_new = df[df['consumer_complaint_narrative'].notnull()]
df.shape
df_new.shape
df_new.info()
df_new.isnull().sum()
df_new = df_new[['product','consumer_complaint_narrative']]
df_new.info()
le = LabelEncoder()
le.fit(df_new['product'].unique())
df_new['category_id'] = le.transform(df_new['product'])
df_new.head()
df_new['category_id'].unique()
df_new = df_new.reset_index()
category_id_df = df_new[['product','category_id']].drop_duplicates().sort_values('category_id')
category_id_df
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','product']].values)
category_to_id
id_to_category
fig = plt.figure(figsize=(8,6))
df_new.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim=0)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2),stop_words='english')
features = tfidf.fit_transform(df_new.consumer_complaint_narrative).toarray()
features.shape
labels = df_new.category_id
N = 2
for product, category_id in sorted(category_to_id.items()):
    feature_chi2 = chi2(features, labels==category_id)
    indices = np.argsort(features_chi2[0]) #根据value对index重排,返回重排后的index
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' '))==1]
    biggrams = [v for v in feature_names if len(v.split(' '))==2]
    print("# '{}':".format(product))
    print(".Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print(".Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
X_train, X_test, y_train, y_test = train_test_split(df_new['consumer_complaint_narrative'],df_new['product'],random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
#检验预测是否准确
df_new[df_new['Consumer_complaint_narrative'] == "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."]
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV*len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(mode, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((nodel_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx','accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor='gray',linewidth=2)
plt.show()
#输出四种模型的交叉验证最终的平均分数分别是多少
cv_df.groupby('model_name').accuracy.mean()
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df_new.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize==(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', 
           xticklabels=category_id_df.product.values, yticklabels=category_id_df.product.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted!=actual and conf_mat[actual, predicted]>=10:
            print("'{}' predicted as '{}':{} examples.".format(id_to_category[actual],id_to_category[predicted],conf_mat[actual, predicted]))
            display(df_new.loc[indices_test[(y_test==actual)&(y_pred==predicted)]][['product','consumer_complaint_narrative']])
            print('')
model.fit(features, labels)
N =2
for product, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' '))==1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' '))==2][:N]
    print("# '{}':".format(product))
    print(" . Top unigrams:\n . {}".format('\n . '.join(unigrams)))
    print(" . Top bigrams:\n . {}".format('\n . '.join(bigrams)))
#最后，输出每个类别的分类报告
print(metrics.classification_report(y_test, y_pred, target_names=df_new['product'].unique()))