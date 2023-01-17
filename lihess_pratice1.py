import numpy as np # 다차원 배열 처리를 지원하는 라이브러리
import pandas as pd 
import matplotlib.pyplot as plt # 시각화
import matplotlib
%matplotlib inline

import seaborn as sns # 시각화
import missingno as msno # 누락 데이터 시각화 라이브러리

import xgboost as xgb # Gradient Boosting
import warnings
sns.set(style='white', context='notebook', palette='deep')

# 기본 라이브러리 추가
np.random.seed(1989)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ", train.shape) # 행, 열
print("Test shape : ", test.shape )
train.head()
print(train.info()) # 트레인 셋의 각 행의 속성
print(test.info())
targets = train['target'].values # 타겟 행의 값을 따로 변수로 지정
sns.set(style="darkgrid")
ax = sns.countplot(x = targets)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))
plt.title('Distribution of Target', fontsize=20)
plt.xlabel('Claim', fontsize=20)
plt.ylabel('Frequency [%]', fontsize=20)
ax.set_ylim(top=700000)
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('Oh no')
print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values)) == 0 else print('Oh no')
print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] else print('Oh no')
# null값 확인
train_null = train
train_null = train_null.replace(-1, np.NaN) # -1값을 누락값으로 간주하여 null값으로 대체

msno.matrix(df=train_null.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
test_null = test
test_null = test_null.replace(-1, np.NaN)

msno.matrix(df=test_null.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
# 널값의 데이터를 포함한 행 추출
train_null = train_null.loc[:, train_null.isnull().any()]
test_null = test_null.loc[:, test_null.isnull().any()]

print(train_null.columns)
print(test_null.columns)
print('Columns \t Number of NaN')
for column in train_null.columns:
    print('{}:\t {}'.format(column,len(train_null[column][np.isnan(train_null[column])])))
# divides all features in to 'bin', 'cat' and 'etc' group.

feature_list = list(train.columns) #컬럼들을 리스트화함
def groupFeatures(features): # groupFeatures함수 정의
    features_bin = []
    features_cat = []
    features_etc = []
    for feature in features: # 각 특정에 맞는 리스트에 해당 컬럼을 추가하는 과정
        if 'bin' in feature:
            features_bin.append(feature)
        elif 'cat' in feature:
            features_cat.append(feature)
        elif 'id' in feature or 'target' in feature:
            continue
        else:
            features_etc.append(feature)
    return features_bin, features_cat, features_etc

feature_list_bin, feature_list_cat, feature_list_etc = groupFeatures(feature_list)
# 리턴 값을 받아옴

print("# of binary feature : ", len(feature_list_bin))
print("# of categorical feature : ", len(feature_list_cat))
print("# of other feature : ", len(feature_list_etc))
def TrainTestHistogram(train, test, feature): # TrainTestHistogram 함수 정의. 히스토그램 생성 함수
    fig, axes = plt.subplots(len(feature), 2, figsize=(10, 40))
    fig.tight_layout() # 글자 안 겹치게

    left  = 0  
    right = 0.9   
    bottom = 0.1   
    top = 0.9     
    wspace = 0.3 
    hspace = 0.7 

    plt.subplots_adjust(left=left, bottom=bottom, right=right, 
                        top=top, wspace=wspace, hspace=hspace)
    # 간격 조정
    
    count = 0
    for i, ax in enumerate(axes.ravel()):
        # ravel : 다차원의 배열을 1차원의 배열로 만들어주는 함수
        if i % 2 == 0:
            title = 'Train: ' + feature[count]
            ax.hist(train[feature[count]], bins=30, normed=False)
            ax.set_title(title)
            # "bis=30"30개의 막대로 구분, "normed=False" 확률밀도가 아닌 빈도로 표시
        else:
            title = 'Test: ' + feature[count]
            ax.hist(test[feature[count]], bins=30, normed=False)
            ax.set_title(title)
            count = count + 1
TrainTestHistogram(train, test, feature_list_bin)
TrainTestHistogram(train, test, feature_list_cat)
TrainTestHistogram(train, test, feature_list_etc)
left  = 0  
right = 0.9    
bottom = 0.1
top = 0.9      
wspace = 0.3   
hspace = 0.7

fig, axes = plt.subplots(13, 2, figsize=(10, 40))
plt.subplots_adjust(left=left, bottom=bottom, right=right, 
                    top=top, wspace=wspace, hspace=hspace)

for i, ax in enumerate(axes.ravel()):
    title = 'Train: ' + feature_list_etc[i]
    ax.hist(train[feature_list_etc[i]], bins=20, normed=True)
    ax.set_title(title)
    ax.text(0, 1.2, train[feature_list_etc[i]].head(), horizontalalignment='left',
            verticalalignment='top', style='italic',
       bbox={'facecolor':'red', 'alpha':0.2, 'pad':10}, transform=ax.transAxes)
etc_ordianal_features = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01',
                    'ps_reg_02', 'ps_car_11', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03',
                    'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
                    'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',
                    'ps_calc_14']

etc_continuous_features = ['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']

train_null_columns = train_null.columns
test_null_columns = test_null.columns
for feature in train_null_columns:
    if 'cat' in feature or 'bin' in feature:
        train_null[feature].fillna(train_null[feature].value_counts().idxmax(), inplace=True)
        # 사용빈도가 가장 높은 수로 대체, 실수형 데이터가 안 나오게 할라고
    elif feature in etc_continuous_features:
        train_null[feature].fillna(train_null[feature].median(), inplace=True)
        # 중앙값으로 대체
    elif feature in etc_ordianal_features:
        train_null[feature].fillna(train_null[feature].value_counts().idxmax(), inplace=True)
    else:
        print(feature)
for feature in test_null_columns:
    if 'cat' in feature or 'bin' in feature:
        test_null[feature].fillna(test_null[feature].value_counts().idxmax(), inplace=True)
    elif feature in etc_continuous_features:
        test_null[feature].fillna(test_null[feature].median(), inplace=True)
    elif feature in etc_ordianal_features:
        test_null[feature].fillna(test_null[feature].value_counts().idxmax(), inplace=True)
    else:
        print(feature)
msno.matrix(df=train.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))
# null값이 없어짐~
msno.matrix(df=train.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))
# null값이 없어짐~
def oneHotEncode_dataframe (df, features): 
    # oneHotEncode_dataframe 함수 정의
    for feature in features:
        temp_onehot_encoded = pd.get_dummies(df[feature])
        # df 데이터의 feature 행을 One-hot encoding 한다.
        column_names = ["{}_{}".format(feature, x) for x in temp_onehot_encoded.columns]
        # 컬럼의 이름을 feature_x의 형식으로 지정하는 데,이때 x는 temp_onehot_encoded.columns에 따른다.
        temp_onehot_encoded.columns = column_names
        # 컬럼명을 변경한다.
        df = df.drop(feature, axis=1)
        df = pd.concat([df, temp_onehot_encoded], axis=1) 
        # 기존에 있던 특성 행을 지우고 세분화한 정보를 붙인다.
        # 세분화된 정보를 붙였기떄문에 기본의 특성을 필요없음
    return df
train = oneHotEncode_dataframe(train, feature_list_cat)
test = oneHotEncode_dataframe(test, feature_list_cat)
train.head()
#아래의 데이터를 확인해보면 "ps_car_11_cat_70" 과 같이 인코딩된 행들을 찾아볼 수 있음
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    # 의사결정트리에서 엔트로피 대신 트리를 생성하는 기준으로 사용하는 지니계수에 대한 함수
    # 지니계수 : 전체 중에서 특정 class에 속하는 관측치의 비율을 모두 제외한 값으로 불순도를 나타냄
    assert( len(actual) == len(pred) )
    # 예측값과 실측값의 길이가 같지않다면 AsertionError를 발생시킴
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    # asarray(a, dtype, order) : 입력을 배열 형태로 변환해주는 함수
    # a: 배열로 변환할 수 있는 데이터. 이때의 np.c_는 actual와 pred, actual 길이를 균등하게 나눈 값을 세로로 붙여서 이차원 배열을 만든다.
    # array함수와 유사하나 입력 데이터가 이미 배열 형태라면 새로운 배열을 생성하지 않는 것이 특징.
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    # all[:,2](np.arange(len(actual)))을 기본 정렬 기준으로, all[:,1](pred)을 보조 정렬 기준으로. 이떄 all[:,1]는 내림차순
    totalLosses = all[:,0].sum()
    # 실측값을 모두 더함
    giniSum = all[:,0].cumsum().sum() / totalLosses
    # 식츨값의 누적합의 합을 실측값의 합으로 나눔
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p): # 지니계수 정규화 함수
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain): 
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score
from sklearn.model_selection import StratifiedShuffleSplit # 추가 해야됨
n_split = 3
SSS = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=1989)
# n_split : 반복횟수, random_state : 난 수 생성 시드 설정
# 매개변수를 최적화함
params = {
    'min_child_weight': 10.0,
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
}
X = train.drop(['id', 'target'], axis = 1).values
y = train.target.values
test_id = test.id.values
test = test.drop('id', axis = 1)
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)
# zeros_like : 지정된 배열과 동일한 모양과 유형으로 0 배열 반환.
SSS.get_n_splits(X, y)
print(SSS)
for train_index, test_index in SSS.split(X, y):
    print("TRAIN: ", train_index, "TEST: ", test_index)
for i, (train_index, test_index) in enumerate(SSS.split(X, y)):
    print('------# {} of {} shuffle split------'.format(i + 1, n_split))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    #분할된 데이터를 XGBost 형식으로 변환
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! 
    model = xgb.train(params, d_train, 2000, watchlist, 
                      early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=100)
    # d_train : 학습 데이터의 레이블
    # 2000 : 반복횟수
    # watchlist : 훈련 중에 검증 성능으로 평가되는 목록
    # verbose_eval : 검증 세트의 평가 지횩 주어진 단계마다 출력됨
    # 참고 : https://xgboost.readthedocs.io/en/latest/python/python_api.html
    
    print('------# {} of {} prediction------'.format(i + 1, n_split))
    # Predict on our test data
    p_test = model.predict(d_test)
    sub['target'] = sub['target'] + p_test/n_split
sub.to_csv('stratifiedShuffleSplit_xgboost.csv', index=False)
