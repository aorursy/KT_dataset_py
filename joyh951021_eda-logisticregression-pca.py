import pandas as pd

import numpy as np
data_dir = '../input/adultcsv/'

data = pd.read_csv(data_dir+'adult.csv')
data.shape
data.head()
# 사용할 feature들 : age, education.num, sex, race, capital.gain, capital.loss, hours.per.week, native.country

# y값 : income



# 필요한 칼럼들만 남기기

columns = ['age', 'education.num', 'sex', 'race', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',

          'income']



data = data[columns].copy()

data.shape
# 결측치 확인

data.isnull().sum()
# age feature 값들 확인 => 그대로 연속적인 값으로 써도 괜찮을 듯!

data['age'].unique()

# education.num 값들 확인 => 그대로 사용해도 괜찮을 듯!

data['education.num'].unique()
# sex 값들 확인

data['sex'].unique()
# Female, male 을 여자면 0, 남자면 1로 설정해주자

data.loc[data['sex'] == 'Female', 'sex'] = 0

data.loc[data['sex'] == 'Male', 'sex'] = 1

data['sex'].value_counts()
# race에 여러가지 값들이 있어서 우선 사용할 feature에서 유보.. 어떤 인종에다가 가중치를 두어야 하나!?

data['race'].value_counts()
data['capital.gain'].unique()
data['capital.loss'].unique()
data['hours.per.week'].unique()
data['native.country'].value_counts()

# 미국이 압도적으로 많으므로 미국이면 0, 다른나라면 1로 바꾸어도 무방할듯!
# 미국이면 0, 미국이 아니면 1로 바꾸어주기

data.loc[data['native.country'] == 'United-States', 'native.country'] = 0

data.loc[data['native.country'] != 0, 'native.country'] = 1

data['native.country'].value_counts()
data['income'].value_counts()
# income $50k(5만달러)가 넘으면 1, 같거나 넘지않으면 0으로 label붙여주기

data.loc[data['income'] == '<=50K', 'income'] = 0

data.loc[data['income'] != 0, 'income'] = 1
data['income'].value_counts()
# 인종 feature은 native.country와 비슷한 성격의 feature이므로 race 칼럼 제거

del data['race']

final_data = data.copy()

final_data.head()
# feature별 칼럼별로 dtypes 확인

final_data.dtypes
final_data['income'] = final_data['income'].astype(int)

final_data.dtypes
# KFold-validation 방법 사용

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
features = ['age','education.num','sex','capital.gain',

            'capital.loss','hours.per.week','native.country']



# KFold-validation에서 Fold개수 5개로 설정후 Shuffle(마구섞기)허용

kf = KFold(n_splits=5, shuffle=True)



# KFold 5번하기 떄문에 한번할 때마다 정확도 담을 리스트 할당

accrs = []

# Fold 횟수 출력

fold_idx = 1



# split train/test data 

for train_idx, test_idx in kf.split(final_data):

    print(f'Fold num : {fold_idx}')

    train_d, test_d = final_data.iloc[train_idx], final_data.iloc[test_idx]

    

    # train 데이터에서 x,y값 할당

    train_x = train_d[features]

    train_y = train_d['income']

    

    # test 데이터에서 x,y값 할당

    test_x = test_d[features]

    test_y = test_d['income']

    

    # 모델 정의 

    model = LogisticRegression() #lbgfs는 최적화알고리즘

    # train 데이터로 학습시키기

    model.fit(train_x, train_y)

    

    # 모델한번 학습하고 test한 후 accuracy 측정

    mean_accr = model.score(test_x, test_y)

    # KFold 할때마다 모델 accuracy 측정해서 accrs리스트에 담기

    accrs.append(mean_accr)

    

    # Fold 횟수 한 번 끝날때마다 1씩 증가

    fold_idx += 1



# 5번 KFold한 정확도 5개값의 평균값 출력

print(np.average(accrs))
# PCA 임포트

from sklearn.decomposition import PCA



features = ['age','education.num','sex','capital.gain',

            'capital.loss','hours.per.week','native.country']



# KFold-validation에서 Fold개수 5개로 설정후 Shuffle(마구섞기)허용

kf = KFold(n_splits=5, shuffle=True)



# KFold 5번하기 떄문에 한번할 때마다 정확도 담을 리스트 할당

accrs = []

# Fold 횟수 출력

fold_idx = 1



# split train/test data 

for train_idx, test_idx in kf.split(final_data):

    print(f'Fold num : {fold_idx}')

    train_d, test_d = final_data.iloc[train_idx], final_data.iloc[test_idx]

    

    # PCA 할당, n_components : 몇개의 feature로 줄일건지

    pca = PCA(n_components=4)

    

    # train 데이터에서 x,y값 할당

        # x = feature에다가 PCA의 fit_transform 적용

    train_x = pca.fit_transform(train_d[features])

    train_y = train_d['income']

    

    # test 데이터에서 x,y값 할당

        # test의 x(feature)에는 PCA의 transform만 적용!

    test_x = pca.transform(test_d[features])

    test_y = test_d['income']

    

    # 모델 정의 

    model = LogisticRegression() #lbgfs는 최적화알고리즘

    # train 데이터로 학습시키기

    model.fit(train_x, train_y)

    

    # 모델한번 학습하고 test한 후 accuracy 측정

    mean_accr = model.score(test_x, test_y)

    # KFold 할때마다 모델 accuracy 측정해서 accrs리스트에 담기

    accrs.append(mean_accr)

    

    # Fold 횟수 한 번 끝날때마다 1씩 증가

    fold_idx += 1



# 5번 KFold한 정확도 5개값의 평균값 출력

print(np.average(accrs))