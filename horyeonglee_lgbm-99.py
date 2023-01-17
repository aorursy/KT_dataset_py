import pandas as pd
import random
true_news = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake_news = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
print(true_news.isnull().sum())
print(fake_news.isnull().sum())
fake_news.head()
true_news.head()
print('true: ',true_news.shape,true_news.columns)
print('fake: ',fake_news.shape,fake_news.columns)
random_index = random.sample(range(true_news.shape[0]),5)
print('true\n',true_news['title'][random_index])
print('fake\n',fake_news['title'][random_index])
print('true\n',true_news['text'][random_index])
print('fake\n',fake_news['text'][random_index])
print('true\n',true_news['subject'].value_counts())
print('fake\n',fake_news['subject'].value_counts())
# 서로 다른 범주를 갖고 있음
true_MON = true_news['date'].apply(lambda x: x[0:3].upper())
fake_MON = fake_news['date'].apply(lambda x: x[0:3].upper())
print(true_MON.value_counts())
print(fake_MON.value_counts())
# 기사 출처 제거를 위해 '-' 를 기준으로
Reuters = true_news['text'].apply(lambda x : x.split('-')[0])
# Reuters를 포함하지 않는 데이터 확인
notReuters = Reuters[Reuters.apply(lambda x: 'Reuters' not in x)]
notReuters[1:20]
# Reuters가 아니더라도 출처를 나타내고 있는 경우가 확인 되기 때문에
# 텍스트의 길이(30)를 기준으로 분류 시도
notReuters[notReuters.apply(lambda x:len(x)>30)]
# 출처가 없이 본문이 시작하는 기사 선별, 해당 기사들은 출처 제거 작업에서 제외 되어야함
no_remove = notReuters[notReuters.apply(lambda x:len(x)>30)].index
# case1) 출처가 있으며, '-'이 존재하는 경우 '-'를 기준으로 텍스트를 분리해 뒷부분 선택
# case2) 출처가 없으며, '-'이 존재하지 않는 경우 텍스트가 분리되지 않아 데이터 그대로 선택
# case3) 출처가 없으며, '-'이 존재하는 경우

for i in range(true_news.shape[0]):
    if i not in no_remove:
        try:
            true_news['text'][i] = '-'.join(true_news['text'][i].split(' - ')[1:])
        except:
            true_news['text'][i] = true_news['text'][i]
    else:
        true_news['text'][i] = true_news['text'][i]
# 내용에 텍스트가 없는 경우 존재 (공백만 있음)
true_news['text'][true_news['text'].apply(lambda x: len(x)<3)]
# 특히 fake_news의 경우 그 수가 매우 많은 편
print(len(fake_news['text'][fake_news['text'].apply(lambda x: len(x)<3)]))
print(len(fake_news['title'][fake_news['title'].apply(lambda x: len(x)<3)]))
# title이 공백인 경우는 없음, 따라서 제목 + 내용을 하나의 변수로 만들어 분석 예정
true_news['content'] = true_news['title'] + true_news['text']
fake_news['content'] = fake_news['title'] + fake_news['text']
# 최소한 한문장은 갖게 됨
print(fake_news['content'][fake_news['content'].apply(lambda x: len(x)<3)])
# 라벨 인코딩
true_news['target'] = 'true'
fake_news['target'] = 'fake'
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
# 영어가 아닌 것 삭제
true_news['content'] = true_news['content'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
fake_news['content'] = true_news['content'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
cnt_vect = CountVectorizer(max_df=0.95, max_features=3000, stop_words='english')
feat_vect = cnt_vect.fit_transform(true_news['content'])
feat_vect.shape
lda = LatentDirichletAllocation(n_components=2)
lda.fit(feat_vect)
print(lda.components_.shape)
lda.components_
def display_topics(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('Topic #', topic_index)
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes = topic_word_indexes[:no_top_words]
        
        feature_concat = ' '.join([feature_names[i] for i in top_indexes])
        print(feature_concat)
        
feature_names = cnt_vect.get_feature_names()
display_topics(lda, feature_names, 15)
    
plt.figure(figsize = (20,20)) # Text that is not Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(true_news[true_news.subject == 'politicsNews'].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is not Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(true_news[true_news.subject == 'worldnews'].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is not Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(true_news.text))
plt.imshow(wc , interpolation = 'bilinear')
cnt_vect = CountVectorizer(max_df=0.95, max_features=3000, stop_words='english')
feat_vect = cnt_vect.fit_transform(fake_news['content'])
lda = LatentDirichletAllocation(n_components=6)
lda.fit(feat_vect)
print(lda.components_.shape)
lda.components_
feature_names = cnt_vect.get_feature_names()
display_topics(lda, feature_names, 15)
fake_news['subject'].unique()
plt.figure(figsize = (20,20)) # Text that is Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(fake_news.text))
plt.imshow(wc , interpolation = 'bilinear')
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# 텍스트만으로 뉴스를 분류할 예정이기 때문에 불필요한 변수 전부 제외
true_news = true_news[['content','target']]
fake_news = fake_news[['content','target']]
# 결합
final_data = pd.concat([true_news,fake_news])
final_data.head()
print(true_news.shape, fake_news.shape)
final_data.shape
# 학습 50%, 검정 30%, 테스트20%로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(final_data['content'],final_data['target'],train_size=0.5)
X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test, train_size = 0.6)
# target 값이 몰린 부분이 없는지 확인
print(y_train.value_counts())
print('='*30)
print(y_valid.value_counts())
print('='*30)
print(y_test.value_counts())
print(X_train.shape,X_valid.shape,X_test.shape)
# 벡터화
tf_vect = TfidfVectorizer(max_features = 10000, stop_words='english')
tf_vect.fit(X_train)
X_train_tf_vect = tf_vect.transform(X_train)
X_valid_tf_vect = tf_vect.transform(X_valid)
X_test_tf_vect = tf_vect.transform(X_test)
# LGBM 모델 학습
lgbm_clf = LGBMClassifier(n_estimators = 1000)
evals = [(X_valid_tf_vect, y_valid)]
lgbm_clf.fit(X_train_tf_vect, y_train, early_stopping_rounds=100, eval_metric='accuracy',eval_set = evals)
# 최종 예측값
pred = lgbm_clf.predict(X_test_tf_vect)
# 정확도
accuracy_score(pred,y_test)
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english')),
    ('lr_clf', LogisticRegression())
])

params = {'tfidf_vect__max_features':[7000,10000,12000],
          'tfidf_vect__max_df':[0.7,0.9],
         'lr_clf__C':[0.1,1,5,10]}
grid_cv_pipe = GridSearchCV(pipeline, param_grid = params, cv=3, scoring='accuracy',verbose=1)
grid_cv_pipe.fit(X_train,y_train)
print(grid_cv_pipe.best_params_, grid_cv_pipe.best_score_)


pred = grid_cv_pipe.predict(X_test)
print('Logistic Reg 정확도 {0:.3f}'.format(accuracy_score(y_test,pred)))