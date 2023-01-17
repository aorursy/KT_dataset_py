import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.tail()
train.shape
train.drop_duplicates()
train.shape

#shape가 동일하므로, 중복 없다.
test.shape
# column 한개는 목표인 target이 빠진거에요.
train.info()
# 데이터는 int와 float형만 있습니다.
# 메타 데이터 만들기. (메타 데이터는 데이터를 위한 데이터)
# 이미 주어진 데이터는 더미화 되어있는 것들이 많다. 이는 사용자가 보기엔 불편할 수 있으므로
# 사용자가 보기 편리하게 메타 데이터를 만들어보자.

data = [] # 빈 시리즈를 만든다.
for f in train.columns: # 각 column별로 f.
    # role, level, keep, dtype
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    } # 중괄호로 묶인 것은 dict(dictionary type)형 이다. 키와 밸류 한쌍을 가지는 형태.
    data.append(f_dict) # 만든 딕셔너리를 리스트에 추가해줍니다.

meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
# 반복문을 통해 만들어진 Data를 이용하여 데이터 프레임을 만들고, 이름을 meta라고 하겠습니다.
# meta는 varname을 인덱스로 취합니다. inplace=True면 적용.
type(f_dict) # 혼자 확인해 본 것. f_dict는 딕셔너리 형. { }
meta
# 메타 데이터 활용 예시. 메타의 level column이 'nomial'이고,
# meta의 keep이 True인 것들만 인덱싱 해 봅시다.
meta[(meta.level == 'nominal') & (meta.keep)].index
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role']})
# role과 level로 메타를 그룹화하고 count만 column인 상태인데
# .reset_index()로 바꿔서 display함. (좀 어렵다)
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe() # .describe()를 이용해 해당 v의 특징 출력

# min을 확인했을 때, -1이면 missing values를 가진 것
# 각각의 min~max range다르다. scaling이 필요하다.

v = meta[(meta.level == 'binary') & (meta.keep)].index
train[v].describe()
# 대부분 0 값이 많다.
# target은 0이 1보다 훨씬 많고,
# 높은 정확도와 높은 평가.
# 오버 샘플링, 언더 샘플링. 여러 전략 있다.

desired_apriori=0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)
vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta

# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))
# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)
v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    plt.figure()
    fig, ax = plt.subplots(figsize=(20,10))
    # Calculate the percentage of target=1 per category value
    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(f, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();
def corr_heatmap(v):
    correlations = train[v].corr() # 상관관계 테이블

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linecolor='skyblue',linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)
s = train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
# Feature Engineering
# Creating dummy variables
# 카테고리형(범주형)을 더미형으로 변환하여 능률을 높이자.
# 카테고리 형태의 값은 순서, 크기가 아니다. 중요한 것은
# 어디에 속해있느냐이다. (0or1) 따라서 더미화 한다.

# level이 nomial이면서 meta.keep이 True인 것은 카테고리형 뿐.
# 이들을 더미화합니다.
# drop_first=True를 이용하여 더미에 사용한 항목을 지운다.
v = meta[(meta.level == 'nominal') & (meta.keep)].index
print('Before dummification we have {} variables in train'.format(train.shape[1]))
train = pd.get_dummies(train, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))
# extra interaction variables
# 정수형(interval)은 PolynomialFeatures를 이용하여 능률 상승
# 그들의 조합으로 이루어진 것을 column으로 사용하자.

# PolynomialFeatures? (다항식 피쳐)
# PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
# 지정한 degree 이하의 모든 피쳐의 다항식 조합으로 구성된 새 피쳐 매트릭스 생성
# For example, if an input sample is two dimensional and of the form [a, b],
# the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
# interaction_only가 True면 같은 항을 2회 이상 곱하지 않는다. (2^2등 안쓴다.)
# include_bias가 False면 0승은 사용 안한다.
# 유의점 : degree와 입력 수에 따라 overfitting(과적) 유발 가능.

# 그렇다면 degree가 2이고, 0승을 제외한 조합 결과들이 interactions가 된다.
# 이후 v를 드랍, 즉 원래 데이터를 지운다.
v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis=1, inplace=True)  # Remove the original columns
# Concat the interaction variables to the train data
print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
train = pd.concat([train, interactions], axis=1)
print('After creating interactions we have {} variables in train'.format(train.shape[1]))
poly
# X는 numpy를 이용하여 0~5까지 수를 생성하고, 2개씩 쌍을 지음
X = np.arange(6).reshape(3, 2)
X
# degree를 2 주었다. 각 쌍에서 그 들로 만들 수 있는 값들 나타남
poly = PolynomialFeatures(2)
poly.fit_transform(X)
poly = PolynomialFeatures(degree = 3,interaction_only=True)
poly.fit_transform(X)
poly = PolynomialFeatures(include_bias=False)
poly.fit_transform(X)
# Feature Selection (머신에게 모두 할당할 수 있으나, 할 수 있는 부분들은 직접 해줌으로써 처리속도 향상)
# 분산이 없거나 매우 낮은 feature를 제거하자.
# Variant Thresould(from sklearn)을 이용.
# 0짜리는 전단계에서 지웠다. 1% 미만 지우면 31개 지워진다.
selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id', 'target'], axis=1)) # Fit to train without id and target variables
f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements
# f는 토글(0과1을 전환)해주는 함수.
v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
# .get_support는 임계점을 통과한 값인지 아닌지 Boolean으로 가르쳐준다. 이 값을 뒤집으면 곧
# 임계점을 넘지 못한 값들이고, v에는 그 값들이 들어갑니다.
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))
selector.fit(train.drop(['id', 'target'], axis=1))
selector.get_support()
# Selecting features with a Random Forest and SelectFromModel
# 여기는 처리시간을 줄이는게 목적. 우리 손으로 줄여줄 수 있는 것 빼줌으로써.
# Sklearn's SelectFromModel을 사용하여 보관할 변수 수를 지정할 수 있다. 
# threshold on the level of feature importance를 임의 지정 가능.

X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
# (트리갯수=1000개(오래걸려), 랜덤시드=0, 병렬작업수=1(fit, predict 둘다 가능))
rf.fit(X_train, y_train)
importances = rf.feature_importances_
# Feature importance?
# 학습된 모델은 Feature importance를 가진다. 결과물이라고 봐도 된다.
indices = np.argsort(rf.feature_importances_)[::-1]
# np.argsort로 정렬한다. 오름차순으로 나타나는데, [::-1]을 통해 슬라이싱은 안하고, 전체 표현
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
tt = np.array([42, 38, 12, 25])
abb = np.argsort(tt)[::-1]
abb
# np.argsort로 정렬한다.
# SelectFrom Model을 이용하여 사용할 적절한 분류기,
# feature importance에 대한 임계값 지정가능
# get_support를 이용하여 train data 변수 수 제한 가능
# Random Forest 결과의 일부만 사용합시다. (feature_importance 기준)
sfm = SelectFromModel(rf, threshold='median', prefit = True)
print('Number of features before selection: {}'.format(X_train.shape[1]))
n_features = sfm.transform(X_train).shape[1]
print('Number of features after seleciton: {}'.format(n_features))
selected_vars = list(feat_labels[sfm.get_support()])
train = train[selected_vars + ['target']]
# Feature scaling
# StandardScaler를 이용, trainset에 적용시킨다.
# 여기서는 classifiers들마다 성능 다르다.
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))
