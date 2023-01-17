# 导入各类包
import pandas as pd #数据读入
from sklearn.linear_model import LogisticRegression # LR算法
from sklearn.feature_extraction.text import CountVectorizer # 数据预处理
!ls ../input/new_data/new_data
# 导入数据
df_train = pd.read_csv('../input/new_data/new_data/train_set.csv')
df_test = pd.read_csv('../input/new_data/new_data/test_set.csv')
df_train.drop(columns=['article','id'],inplace=True)
df_test.drop(columns=['article'],inplace=True)
# 数据处理
vectorizer = CountVectorizer(ngram_range=(1,2),min_df=3,max_df=0.9,max_features=100000)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class']-1
# 使用LR算法
# 其他可以尝试的算法：朴素贝叶斯、xgb、svm等等，没尝试，但估计运算时间会爆炸
lg = LogisticRegression(C=4,dual=True)
lg.fit(x_train,y_train)
# 预测
y_test = lg.predict(x_test)
# 输出结果
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:,['id','class']]
df_result.to_csv('./result.csv',index=False)