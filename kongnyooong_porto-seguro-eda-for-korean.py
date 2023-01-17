# import library



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from sklearn.impute import SimpleImputer # 결측치 대체를 위한 라이브러리

from sklearn.preprocessing import PolynomialFeatures # 교호작용 변수 생성 (중요한 변수끼리의 곱)

from sklearn.preprocessing import StandardScaler



from sklearn.feature_selection import VarianceThreshold # FeatureSelection에서 분산이 기준치보다 낮은 feature는 탈락

from sklearn.feature_selection import SelectFromModel # Feature Importance를 제공하는 모델의 importance를 활용하여 변수선택



from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier



warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
# porto 대회가 워낙 옛날 대회이다보니 그 사이에 캐글 인터페이스나 데이터 경로들이 변경되었음

# 모든 대회들이 하위 폴더가 하나 혹은 몇개씩 더 생겼으므로 path를 지정해주는 것이 편함



data_path = "../input/porto-seguro-safe-driver-prediction/"



df_train = pd.read_csv(data_path+"train.csv")

df_test = pd.read_csv(data_path+"test.csv")
df_train.head()
df_test.head()
# 데이터의 행과 열을 확인해본다.

# 타겟과 id를 제외한 58개의 feature가 존재한다.

# 특이한 점은 train set이 test set보다 적다.



print(df_train.shape)

print(df_test.shape)
df_train.isnull().sum()
import missingno as msno # 결측치 시각화 라이브러리



msno.matrix(df=df_train.iloc[:,:40], figsize=(14, 10))
df_train.info()
# append를 위해 빈 리스트를 만들어주었음

data = []



for f in df_train.columns:

    # 데이터의 역할을 지정 (독립변수, 종속변수, id (PM))

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

         

    # 데이터의 레벨을 지정 (명목변수, 간격변수, 순서변수등을 레벨이라고 표현한 듯)

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif df_train[f].dtype == float:

        level = 'interval'

    elif df_train[f].dtype == int:

        level = 'ordinal'

        

    # id는 False로 지정해주어 버리기로 하고, 나머지는 True로 가져감

    keep = True

    if f == 'id':

        keep = False

    

    # 데이터의 타입 지정

    dtype = df_train[f].dtype

    

    # DataFrame으로 만들기 위해 리스트에 append하기 전에 딕셔너리 타입으로 만들어주었음

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(f_dict)



# 변수의 이름을 인덱스로 하는 데이터프레임을 만들어줌     

meta = pd.DataFrame(data, columns = ["varname", "role", "level", "keep", "dtype"])

meta.set_index("varname", inplace = True)
meta
# ex1



meta[(meta["level"] == "nominal") & (meta["keep"])].index
# ex2



meta.groupby(["role", "level"])["role"].size()
Interval = meta[(meta["level"] == "interval") & (meta["keep"])].index
# describe를 통해 interval 변수들의 통계량을 확인 



df_train[Interval].describe()
Ordinal = meta[(meta["level"] == "ordinal") & (meta["keep"])].index
# describe를 통해 Ordinal 변수들의 통계량을 확인 



df_train[Ordinal].describe()
Binary = meta[(meta["level"] == 'binary') & (meta["keep"])].index
# describe를 통해 Ordinal 변수들의 통계량을 확인 



df_train[Binary].describe()
f, ax = plt.subplots(figsize = (8,8))



df_train['target'].value_counts().plot.pie(explode = [0, 0.1], autopct = '%1.1f%%', 

                                               shadow = True, colors = ['lightcoral', 'lightskyblue'],

                                              textprops={'fontsize': 18})

plt.title("Target PiePlot", size = 20)



# 불균형이 굉장히 심하다.
# 언더샘플링 비율을 지정해주기 위함 

desired_apriori=0.10



# target 변수의 클래스에 따른 인덱스 지정 

idx_0 = df_train[df_train["target"] == 0].index

idx_1 = df_train[df_train["target"] == 1].index



# 지정해준 인덱스로 클래스의 길이(레코드 수) 지정

nb_0 = len(df_train.loc[idx_0])

nb_1 = len(df_train.loc[idx_1])



# 언더샘플링 수행

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

undersampled_nb_0 = int(undersampling_rate*nb_0)

print('target=0에 대한 언더샘플링 비율: {}'.format(undersampling_rate))

print('언더샘플링 전 target=0 레코드의 개수: {}'.format(nb_0))

print('언더샘플링 후 target=0 레코드의 개수: {}'.format(undersampled_nb_0))



# 언더샘플링 비율이 적용된 개수 만큼 랜덤하게 샘플을 뽑아서 그 인덱스를 저장

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)



# 언더샘플링 인덱스와 클래스 1의 인덱스를 리스트로 저장

idx_list = list(undersampled_idx) + list(idx_1)



# 저장한 인덱스로 train set 인덱싱

df_train = df_train.loc[idx_list].reset_index(drop=True)
# 실습을 위해 불균형 데이터 랜덤으로 생성 



import scipy as sp



n0 = 200; n1 = 20

rv1 = sp.stats.multivariate_normal([-1, 0], [[1, 0], [0, 1]])

rv2 = sp.stats.multivariate_normal([+1, 0], [[1, 0], [0, 1]])

X0 = rv1.rvs(n0, random_state=0)

X1 = rv2.rvs(n1, random_state=0)

X_imb = np.vstack([X0, X1])

y_imb = np.hstack([np.zeros(n0), np.ones(n1)])

X_train = pd.DataFrame(data = X_imb, columns = ["X0", "X1"])

y_train = pd.DataFrame(data = y_imb, columns = ["target"])



#-------------------------------------------------------------------------------------------------------------

# 1) RandomUnderSampler



from imblearn.under_sampling import RandomUnderSampler

Undersampled_train, Undersampled_target = RandomUnderSampler(random_state=0).fit_sample(X_train, y_train)



nb_0 = len(y_train[y_train["target"] == 0.0].index)

undersampled_nb_0 = len(Undersampled_target[Undersampled_target["target"] == 0.0].index)



print('RandomUnderSampler 전 target=0 레코드의 개수: {}'.format(nb_0))

print('RandomUnderSampler 후 target=0 레코드의 개수: {}'.format(undersampled_nb_0))



#-------------------------------------------------------------------------------------------------------------

# 2) TomekLinks



from imblearn.under_sampling import TomekLinks

Undersampled_train, Undersampled_target = TomekLinks().fit_sample(X_train, y_train)
vars_with_missing = []



# 모든 컬럼에 -1이라는 값이 1개 이상 있는 것을 확인하여 출력

# 어느 변수에 몇개의 레코드가 있는지, 비율은 얼마나 되는지 까지 확인하여 깔끔하게 출력된다.



for f in df_train.columns:

    missings = df_train[df_train[f] == -1][f].count()

    if missings > 0:

        vars_with_missing.append(f)

        missings_perc = missings/df_train.shape[0]

        

        print('Variable {}\t has {:>10} records\t ({:.2%})\t with missing values'.format(f, missings, missings_perc))

print()        

print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
# 결측치가 너무 많았던 변수들 제거 

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

df_train.drop(vars_to_drop, inplace=True, axis=1)



# 만들어주었던 메타데이터 업데이트 (버린 변수를 keep = True에서 False로)

meta.loc[(vars_to_drop),'keep'] = False  



# 그 외의 결측치를 평균과 최빈값으로 대체

# SimpleImputer를 사용 (커널에서는 그냥 Imputer를 사용하는데 업데이트 후 이름이 바뀐듯)

mean_imp = SimpleImputer(missing_values=-1, strategy='mean')

mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')

df_train['ps_reg_03'] = mean_imp.fit_transform(df_train[['ps_reg_03']])

df_train['ps_car_12'] = mean_imp.fit_transform(df_train[['ps_car_12']])

df_train['ps_car_14'] = mean_imp.fit_transform(df_train[['ps_car_14']])

df_train['ps_car_11'] = mode_imp.fit_transform(df_train[['ps_car_11']])
# 이 커널에서는 이런식으로 유니크값이 몇개있는지 확인했다.



Nominal = meta[(meta["level"] == 'nominal') & (meta["keep"])].index



for f in Nominal:

    dist_values = df_train[f].value_counts().shape[0]

    print('Variable {} has {} distinct values'.format(f, dist_values))
# 개인적인 생각으로는 이 코드와 같이 nuniuqe()를 사용하면 훨씬 간편하지 않을까 싶다.



Nominal = meta[(meta["level"] == 'nominal') & (meta["keep"])].index



for f in Nominal:

    print('Variable {} has {} distinct values'.format(f, df_train[f].nunique()))
# 아래와 같은 데이터가 있다고 가정해보자



ex_list = [("남자", 1), ("여자", 1), ("여자", 1), ("여자", 0), ("남자", 0)]



ex = pd.DataFrame(data = ex_list, columns = ["성별", "target"])
# 이런 방법으로 인코딩을 수행한다.

# 인코딩할 범주형 변수와 target을 groupby해준 후 평균값을 취해준다.



성별_mean = ex.groupby("성별")["target"].mean()
# 그렇게 되면 아래와 같은 값을 얻을 수 있다.

# 남자의 경우 2개의 데이터에서 target값이 1과0 이므로 0.5가 나오고,

# 여자의 경우 3개의 데이터에서 target값이 1이 2개 0이 1개이므로 0.6667이 나온다.

# 이 값으로 해당 unique값을 인코딩 해준다.



성별_mean
# 커널에서 구현한 mean-encoding 코드

# 코드가 매우 복잡해보이지만 결국 위에서 보았던 예제 방식을 구현한 것이다.

# 오버피팅 방지를 위해 noise를 추가하고, smoothing을 적용하기 때문에 코드가 복잡해보인다.

# smoothing을 통해 평균값이 치우친 상황을 조금이나마 보완해준다. (전체 평균값으로 가깝게)

# smoothing에 대한 자세한 이론은 위의 설명에 있는 출처 링크에서 확인! 





# 오버피팅 방지를 위해 약간의 noise를 추가한다고 한다.

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

    # agg를 사용해서 평균값을 구해줌

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # 오버피팅 방지를 위한 smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    prior = target.mean()

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # train, test에 적용시켜준다.

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

  

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
# 위에서 구현한 함수를 ps_car_11_cat(104개의 유니크 값)에 적용시켜준다.

# feature가 바뀌었으므로 메타데이터를 업데이트 해준다.



train_encoded, test_encoded = target_encode(df_train["ps_car_11_cat"], 

                             df_test["ps_car_11_cat"], 

                             target=df_train.target, 

                             min_samples_leaf=100,

                             smoothing=10,

                             noise_level=0.01)

    

df_train['ps_car_11_cat_te'] = train_encoded

df_train.drop('ps_car_11_cat', axis=1, inplace=True)

meta.loc['ps_car_11_cat','keep'] = False  

df_test['ps_car_11_cat_te'] = test_encoded

df_test.drop('ps_car_11_cat', axis=1, inplace=True)
Nominal = meta[(meta["level"] == 'nominal') & (meta["keep"])].index





# 변수별로 반복문을 돌려서 barplot을 그린다.

for f in Nominal:

    plt.figure()

    fig, ax = plt.subplots(figsize=(20,10))

    ax.grid(axis = "y", linestyle='--')

    # 변수 별 target=1의 비율 계산

    cat_perc = df_train[[f, 'target']].groupby([f],as_index=False).mean()

    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    

    # 위에서 계산해준 비율을 통해 target = 1의 데이터 중 어떤 유니크값의 비율이 높은지 확인할 수 있다.

    sns.barplot(ax=ax, x=f, y='target',palette = "Pastel1", edgecolor='black', linewidth=0.8, data=cat_perc, order=cat_perc[f], )

    plt.ylabel('% target', fontsize=18)

    plt.xlabel(f, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show();
def corr_heatmap(Interval):

    correlations = df_train[Interval].corr()



    # Create color map ranging between two colors

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',

                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})

    plt.show();

    

Interval = meta[(meta["role"] == "target") | (meta["level"] == 'interval') & (meta["keep"])].index

corr_heatmap(Interval)
# plot을 만드는데 꽤나 긴 시간이 걸리기 때문에 10%의 sample만 뽑아서 사용



# s = df_train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=df_train, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_13', data=df_train, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_14', data=df_train, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_15', y='ps_car_13', data=df_train, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
Ordinal = meta[(meta["role"] == "target") | (meta["level"] == 'ordinal') & (meta["keep"])].index

corr_heatmap(Ordinal)
Nominal = meta[(meta["level"] == 'nominal') & (meta["keep"])].index

print('One-Hot Encoding 전 train 데이터 셋 변수 개수: {}'.format(df_train.shape[1]))

df_train = pd.get_dummies(df_train, columns=Nominal, drop_first=True)

df_test = pd.get_dummies(df_test, columns=Nominal, drop_first=True)

print('One-Hot Encoding 후 train 데이터 셋 변수 개수: {}'.format(df_train.shape[1]))



# 52개의 변수가 늘어났다.
Interval = meta[(meta["level"] == 'interval') & (meta["keep"])].index



poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

interactions = pd.DataFrame(data=poly.fit_transform(df_train[Interval]), columns=poly.get_feature_names(Interval))

interactions.drop(Interval, axis=1, inplace=True)  # interactions 데이터프레임에서 기존 변수 삭제



# 새로 만든 변수들을 기존 데이터에 concat 시켜줌

print('교호작용 변수 생성 전 train 데이터 셋 변수 개수: {}'.format(df_train.shape[1]))

df_train = pd.concat([df_train, interactions], axis=1)

df_test = pd.concat([df_test, interactions], axis=1)

print('교호작용 변수 생성 후 train 데이터 셋 변수 개수: {}'.format(df_train.shape[1]))
# 만들어진 feature 확인



df_train.head()