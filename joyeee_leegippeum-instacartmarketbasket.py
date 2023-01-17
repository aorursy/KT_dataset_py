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
import zipfile
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/aisles.csv.zip","r") as zip_ref:
    zip_ref.extractall("./")
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/orders.csv.zip","r") as zip_ref:    
    zip_ref.extractall("./")
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/departments.csv.zip","r") as zip_ref:
    zip_ref.extractall("./")
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/products.csv.zip","r") as zip_ref:    
    zip_ref.extractall("./")
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/order_products__train.csv.zip","r") as zip_ref:
    zip_ref.extractall("./")
with zipfile.ZipFile("/kaggle/input/instacart-market-basket-analysis/order_products__prior.csv.zip","r") as zip_ref:    
    zip_ref.extractall("./")
    
# zip 압축풀기
# 현재위치인 output_kaggle에 csv 파일 저장
import pandas as pd
aisles = pd.read_csv('aisles.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
departments = pd.read_csv('departments.csv')
order_products__prior = pd.read_csv('order_products__prior.csv')
order_products__train = pd.read_csv('order_products__train.csv')
aisles
departments
products
# products에 aisle과 department가 연결되어 있음.
df_product1 = pd.merge(products, aisles, on="aisle_id")
df_product1
df_product2 = pd.merge(df_product1, departments, on="department_id")
df_product2
# product_id를 key로 사용하여 aisles,departments를 products에 merge.
orders
orders['user_id'].value_counts()
# 주문내역을 user_id로 count 해보면 206209명의 고객 정보가 있다는 것을 알 수 있음.
# 각 고객 한명당 주문건수가 4번에서 부터 100번까지 있음.
# 총 주문건수는 3421083개
orders.eval_set.value_counts()
# 총 206209명의 고객 중 train 할 고객은 131209명이고 test할 고객은 75000명이다.
order_products__prior
# 원본 데이터(모집단)
# fit, predict 모델 검증 끝난후에
order_products__train
# 모델 검증을 위한 샘플 데이터(표본)
df_product2
df_order_product = pd.merge(order_products__train, df_product2, on="product_id")
df_order_product
df0 = pd.merge(orders, df_order_product, on="order_id")
df0
# 모든 데이터 다 merge 한 것.
df.keys()
# order_number: 주문한 횟수
# order_dow: 주문한 요일
# order_hour_of_day: 하루 중 주문한 시각
# columns 정리하기(순서변경, 삭제)

df1 = df0[['user_id', 'order_id', 'order_number', 'order_dow',
       'order_hour_of_day',
       'add_to_cart_order', 'reordered', 'product_id', 'product_name', 'aisle_id', 'aisle',
       'department_id', 'department']]

# days_since_prior_order: 목표가 다음에 주문할 것이 무엇일지 예측하는 것이기 때문에 필요없는 컬럼같음.
df['eval_set'].value_counts()
# df는 orders에서 train 데이터만 모아놓은 것.
# ['eval_set'] 컬럼 삭제해도 됨.
df1
# 전처리 끝
df1.keys()

# 종속변수: 'reordered' 
# 0,1로 나누어지는 범주형 변수

# 독립변수: 'user_id', 'order_id', 'order_number', 'order_dow', 'order_hour_of_day','product_id', 'product_name', 'aisle_id', 'aisle', 'department_id', 'department'
# 연속형과 범주형이 섞여있음.
df2 = df1[['user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'product_name', 'aisle', 'department', 'reordered']]
df2
df2.to_csv('df2.csv')

# jamovi로 확인해보기
# Elbow Method
from sklearn.cluster import KMeans
distortions = []
for i in range(1,5):
    kmeans = KMeans(n_clusters=i, n_init=10, max_iter=300)
    kmeans.fit(df2)
    distortions.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,5), distortions, marker='o')
plt.show()  # 꺽인 부분 찾기

# # Silhouette
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=2, max_iter=300)
# labels = kmeans.fit_predict(squad)
# target = pd.DataFrame(kmeans.labels_, columns=['reordered'])
# 종속변수, 독립변수 지정하기
y = df2['reordered']
x = df2[['user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'product_name', 'aisle', 'department']]
x_dummies = pd.get_dummies(x[['order_dow','order_hour_of_day','product_name','aisle','department']], drop_first=True)
new_x = pd.concat([df2['user_id'],x_dummies],axis=1)

from sklearn.model_selection import train_test_split
x_train0, x_test0, y_train0, y_test0 = train_test_split(new_x, y, test_size=0.5)

# sample size로 데이터 양 줄이기(0.25 수준)
# 모델링 속도 향상을 위한 쪼개기 작업
x_train, x_test, y_train, y_test = train_test_split(x_train0, y, test_size=0.3)

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = pd.read_csv('')
x_test
y_train
y_test
# VotingClassifier() with No_params
# voting(hard, soft)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
clflog = LogisticRegression()
clfrf = RandomForestClassifier()
clfgn = GaussianNB()
clfsvc = SVC()
clfknn = KNeighborsClassifier()

from sklearn.ensemble import VotingClassifier
eclf_h = VotingClassifier(estimators = [('lr',clflog),('rf',clfrf),('gnb',clfgn),('svc',clfsvc),('knn',clfknn)], voting='hard')
eclf_s = VotingClassifier(estimators = [('lr',clflog),('rf',clfrf),('gnb',clfgn),('svc',clfsvc),('knn',clfknn)], voting='soft')

from sklearn.metrics import classification_report
models = [clflog, clfrf, clfgn, clfsvc, clfknn, eclf_h, eclf_s]

for model in models:
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    score = model.score(x_test,y_test)
    print(classification_report(y_test,predictions))
# VotingClassifier(hard)
# GridSearchCV
# best_params_

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
clflog = LogisticRegression()
clfrf = RandomForestClassifier()
clfgn = GaussianNB()
clfsvc = SVC()
clfknn = KNeighborsClassifier()

from sklearn.ensemble import VotingClassifier
eclf_h = VotingClassifier(estimators = [('lr',clflog),('rf',clfrf),('gnb',clfgn),('svc',clfsvc),('knn',clfknn)],voting='hard')
c_params = [0.001,0.01,0.1,1,5,10.50,100,300,500,1000]
params = {
    'lr__solver':['liblinear','lbfgs','saga'], 
    'lr__penalty':['l1','l2','elasticnet'], 
    'lr__C':c_params,
    'rf__criterion':['gini','entropy'],
    'rf__min_samples_leaf':[1,2,3,4,5],
    'rf__n_estimators':[100,150,200],
    # 'gnb__':[], 파라미터 지정할 필요 없는듯.. 그냥 default로
    'svc__C':c_params,
    'svc__gamma':[0.001,0.01,0.1,1,10],
    'svc__kernel':['rbf','sigmoid'],
    'svc__decision_function_shape':['ovo'],
    'knn__n_neighbors':[1,2,3,4,5,6,7,8,9,10],
    'knn__weights':['uniform','distance']
}

# In your example, the cv=5, so the data will be split into train and test folds 5 times. 
# The model will be fitted on train and scored on test. 
# These 5 test scores are averaged to get the score.
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator = eclf_h, param_grid=params, cv=5, n_jobs=-1)
grid = grid.fit(x_train,y_train)
grid.best_params_     # VotingClassifier의 best_params_의 의미는 lr, rf, gnb, svc, knn 다함께 사용 할때의 최적의 파라미터라는 뜻.
# VotingClassifier(hard)의 score with best_params_

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
clflog = LogisticRegression()
clfrf = RandomForestClassifier()
clfgn = GaussianNB()
clfsvc = SVC()
clfknn = KNeighborsClassifier()
# 아래처럼 best_params 넣어주어야 함
# clflog = LogisticRegression(C=5.0, penalty='l2', solver='liblinear')
# clfdt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=5)

from sklearn.ensemble import VotingClassifier
eclf_h = VotingClassifier(estimators = [('lr',clflog),('rf',clfrf),('gnb',clfgn),('svc',clfsvc),('knn',clfknn)], voting='hard')
eclf_h.fit(x_train,y_train)
y_pred = eclf_h.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

print(eclf_h.score(x_test,y_test))
# VotingClassifier(soft)
# 위에꺼 따라하기

eclf_s = VotingClassifier(estimators = [('lr',clflog),('rf',clfrf),('gnb',clfgn),('svc',clfsvc),('knn',clfknn)], voting='soft')