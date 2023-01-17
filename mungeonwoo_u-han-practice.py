import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

import seaborn as sns
import missingno as msno

import xgboost as xgb
import warnings
sns.set(style='white', context = 'notebook', palette='deep')
warnings.filterwarnings("ignore")


np.random.seed(1989)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)
train.head()
print(train.info())
print(test.info())
targets = train['target'].values
sns.set(style="darkgrid")
ax = sns.countplot(x=targets)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)),
               (p.get_x()+ 0.3, p.get_height()+10000))
plt.title('Distribution of Target', fontsize = 20)
plt.xlabel('Claim', fontsize =20)
plt.ylabel('Frequency [%]', fontsize= 20)
ax.set_ylim(top = 700000)
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('Oh no')
print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values)) == 0 else print('Oh no')
print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] else print('Oh no')
import missingno as msno

train_null = train
train_null = train_null.replace(-1, np.NAN)

msno.matrix(df= train_null.iloc[:, :], figsize=(20,14), color=(0.8,0.5,0.2))
test_null = test
test_null = test_null.replace(-1, np.NAN)

msno.matrix(df=test_null.iloc[:,:], figsize=(20,14), color=(0.8,0.5,0.2))
train_null = train_null.loc[:, train_null.isnull().any()]
test_null = test_null.loc[:, test_null.isnull().any()]

print(train_null.columns)
print(test_null.columns)
print('Columns \t Number of NaN')
for column in train_null.columns:
    print('{}: \t {}'.format(column, len(train_null[column][np.isnan(train_null[column])])))
#divides all features in to 'bin', 'cat' and 'etc' group.

feature_list = list(train.columns) # train의 컬럼들을 리스트화해서 feature_list에 넣음
def groupFeatures(features): # groupFeatures 함수 정의 파라미터로 features(리스트)를 받음
    features_bin = [] # features_bin 리스트 생성
    features_cat = [] # features_cat 리스트생성
    features_etc = [] # features_etc 리스트 생성
    for feature in features : # 파라미터로 받은 features 리스트를 하나씩 빼서 for문
        if 'bin' in feature: # feature에 'bin' 이라는 단어가 들어가면
            features_bin.append(feature) # features_bin 리스트에 feature를 추가
        elif 'cat' in feature: # 또는 'cat'이라는 단어가 들어가면
            features_cat.append(feature) #features_cat에 추가
        elif 'id' in feature or 'target' in feature: #또는 feature에 'id' 또는 'target'이 들어가면
            continue # 다음꺼 계속
        else: # 그것도 아니면 features_etc에 추가
            features_etc.append(feature)
    return features_bin, features_cat, features_etc

feature_list_bin, feature_list_cat, feature_list_etc = groupFeatures(feature_list)

#feature_list_bin, cat, etc 에 groupFeature 함수에 feature_list를 파라미터로 넣은 return 값들을 넣음

print("# of binary feature : ", len(feature_list_bin)) # 길이들을 출력
print("# of categorical feature : ", len(feature_list_cat))
print("# of other feature : ", len(feature_list_etc))
def TrainTestHistogram(train, test, feature):   # TrainTestHistogram 함수 정의 train, test, feature를 파라미터로 받음
    fig, axes = plt.subplots(len(feature), 2, figsize=(10,40)) # feature길이 행 , 2열로 subplot 생성 크기 10,40 사이즈
    fig.tight_layout() # 그래프랑 글자들끼리 겹치지 않게 딱 들어맞게 만들어줌
    
    left = 0
    right = 0.9
    bottom = 0.1
    top = 0.9
    wspace = 0.3
    
    hspace = 0.7
    
    plt.subplots_adjust(left=left, bottom = bottom, right=right, top = top, wspace=wspace, hspace=hspace)
    # 그래프들의 간격을 조정
    count = 0
    
    for i, ax in enumerate(axes.ravel()): # ravel 함수. numpy에 있는 함수로 여러 리스트로 되어있는 것을 하나로 만들어줌.
        # enumerate는 리스트에 인덱스를 포함하게 만든다. 그래서 i에 인덱스 저장
        # ax에 값 저장.
        if i % 2 == 0 :  # i가 짝수이면.
            title = 'Train : ' + feature[count] # title에 train + feature의 count 번째에 있는 값으로 title 정의
            ax.hist(train[feature[count]], bins =30, normed = False)
            # 히스토그램 그리기    bins = 30은 30개의 막대기로 구분한다는 뜻. 몇개의 막대기로 구분할 것인가.
            #normed = false 는 확률밀도가 아니라 빈도를 표시한다는 뜻.
            ax.set_title(title) # 제목 설정
            
        else: # i가 홀수이면
            title = 'Test : ' + feature[count]
            ax.hist(test[feature[count]], bins = 30, normed = False)
            ax.set_title(title) # 제목설정
            count = count + 1 # 카운트 증가
TrainTestHistogram(train,test,feature_list_bin)
# TrainTestHistogram 함수에 train, test, feature_list_bin 을 넣음
TrainTestHistogram(train, test, feature_list_cat)

# TrainTestHistogram 함수에 train, test, feature_list_cat 을 넣음
TrainTestHistogram(train, test, feature_list_etc)

# TrainTestHistogram 함수에 train, test, feature_list_etc 을 넣음
left = 0
right = 0.9
bottom = 0.1
top = 0.9
wspace = 0.3

hspace = 0.7

fig, axes = plt.subplots(13,2,figsize=(10,40))
# plt.subplots 13행 2열 10 40사이즈 생성
plt.subplots_adjust(left=left, bottom=bottom, right = right, top = top, wspace=wspace, hspace=hspace)

for i, ax in enumerate(axes.ravel()):
    title = 'Train: ' + feature_list_etc[i] # title 변수에 'Train' + etc의 i번째에있는 값 더해서 title 정의
    ax.hist(train[feature_list_etc[i]], bins=20,normed=True)
    # normed = True는 정규분포의 확률밀도 함수로 나타낸다..
    ax.set_title(title) # 제목 생성
    ax.text(0, 1.2, train[feature_list_etc[i]].head(), horizontalalignment = 'left',
           verticalalignment='top', style = 'italic', bbox={'facecolor': 'red', 'alpha':0.2, 'pad' : 10},
           transform = ax.transAxes)
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.text.html 참고..
    # 크기, 내용, text 위치들 , italic 채 , bbox 위에 사각형을 만듦 .
etc_ordianal_features = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01',
                    'ps_reg_02', 'ps_car_11', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03',
                    'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
                    'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',
                    'ps_calc_14']
etc_continuous_features = ['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']

train_null_columns = train_null.columns # train_null의 컬럼들을 저장
test_null_columns = test_null.columns # test_null의 컬럼들을 저장.

for feature in train_null_columns : # train_null_columns에 들어있는 컬럼들을 feature로 받아.
    if 'cat' in feature or 'bin' in feature: # feature에 'cat' 또는 'bin'이 들어있으면
        train_null[feature].fillna(train_null[feature].value_counts().idxmax(), inplace= True)
        # train_null[feature]의 값들의 개수를 각각(예: 1이 3개 2가 2개 3이 4개...등) 센다음에 그 값들의 개수가 가장 큰 걸로 nan값을 채움.
        #inplace = true 를 사용해야 train_null[feature]에 해당 내용이 반영된다.
    elif feature in etc_continuous_features: # 또는 etc_continuous_features에 feature가 있으면
        train_null[feature].fillna(train_null[feature].median(), inplace=True)
        # nan 값을 중앙 값으로 채운다.
    elif feature in etc_ordianal_features: # 또는 etc_rodianal_feature에 feature가 있으면.
        train_null[feature].fillna(train_null[feature].value_counts().idxmax(), inplace=True)
        # 맨위와 마찬가지
    else :
        print(feature)
    
for feature in test_null_columns: # 테스트 마찬가지
    if 'cat' in feature or 'bin' in feature:
        # For categorical and binary features with postfix, substitue null values with the most frequent value to avoid float number.
        test_null[feature].fillna(test_null[feature].value_counts().idxmax(), inplace=True)
    elif feature in etc_continuous_features:
        test_null[feature].fillna(test_null[feature].median(), inplace=True)
    elif feature in etc_ordianal_features:
        # For categorical and binary features which was assumed, substitue null values with the most frequent value to avoid float number.
        test_null[feature].fillna(test_null[feature].value_counts().idxmax(), inplace=True)
    else:
        print(feature)
for feature in train_null_columns: # train_null_columns의 값들을 feature에 넣어
    train[feature] = train_null[feature]
   # train의 feature의 컬럼에   train_null 의 feature를 넣음
# 

for feature in test_null_columns:  
    test[feature] = test_null[feature]
msno.matrix(df=train.iloc[:,:], figsize=(20,14), color=(0.3,0.6,0.3))

# 널 값이 얼마나 들어있는지 볼수 있는 그래프 그림. 행열 처음부터 끝까지 크기, 색깔
msno.matrix(df=test.iloc[:,:], figsize=(20,14), color = (0.2,0.3,0.8))

# 널 값이 얼마나 들어있는지 볼수 있는 그래프 그림. 행열 처음부터 끝까지 크기, 색깔
def oneHotEncode_dataframe(df, features):
    for feature in features:
        temp_onehot_encoded = pd.get_dummies(df[feature])
        column_names = ["{}_{}".format(feature, x) for x in temp_onehot_encoded.columns]
        temp_onehot_encoded.columns = column_names
        df = df.drop(feature, axis= 1)
        df = pd.concat([df, temp_onehot_encoded], axis=1)
    return df
train = oneHotEncode_dataframe(train, feature_list_cat)
test = oneHotEncode_dataframe(test, feature_list_cat)
def gini(actual, pred, compcol = 0, sortcol = 1):
    assert(len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype = np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) /2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a,p) / gini(a,a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score
from sklearn.model_selection import StratifiedShuffleSplit
n_split = 3
SSS = StratifiedShuffleSplit(n_splits=3, test_size = 0.5, random_state = 1989)
params = {
    'min_child_weight' : 10.0,
    'max_depth' : 7,
    'max_delta_step': 1.8,
    'colsample_bytree' : 0.4,
    'subsample' : 0.8,
    'eta' : 0.025,
    'gamma' : 0.65,
    'num_boost_round' : 700
}
X = train.drop(['id', 'target'], axis = 1).values
y = train.target.values
test_id = test.id.values
test = test.drop('id', axis = 1)
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)
SSS.get_n_splits(X,y)
print(SSS)
for train_index, test_index in SSS.split(X,y):
    print("TRAIN: ", train_index, "TEST: ", test_index)
for i, (train_index, test_index) in enumerate(SSS.split(X,y)):
    print('--------# {} of {} shuffle split----------'.format(i + 1, n_split))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    model = xgb.train(params, d_train, 2000, watchlist,
                     early_stopping_rounds=100, feval =gini_xgb, maximize = True, verbose_eval = 100)
    
    print('----- # {} of {} prediction-------'.format(i + 1, n_split))
    
    p_test = model.predict(d_test)
    sub['target'] = sub['target'] + p_test/n_split
# sub.to_csv('stratifiedShuffleSplit_xgboost.csv', index=False)
