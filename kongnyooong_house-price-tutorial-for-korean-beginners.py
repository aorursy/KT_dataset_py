import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import os 

from collections import Counter



plt.style.use('seaborn')

sns.set(font_scale=1.5)



import missingno as msno



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline





# 기본적 모듈을 임포트 해줍니다.
os.listdir("../input")



# input의 하위폴더를 확인해줍니다. 
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# 트레인, 테스트 데이터를 불러옵니다.
df_train.head()



# 잘 불러졌는지 확인해봅니다.
df_train.shape, df_test.shape



# train 데이터는 1460개의 데이터와 81개의 feature

# test 데이터는 1459개의 데이터와 80개의 feature가 있습니다.
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))



# 편의상 수치형 변수와 명목형 변수를 나눠줍니다.

# 수치형 변수는 38개, 명목형 변수는 43개가 있습니다.
print(df_train[numerical_feats].columns)

print("*"*80)

print(df_train[categorical_feats].columns)



# 변수명을 확인해봅니다. 
def detect_outliers(df, n, features):

    outlier_indices = []

    for col in features:

        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col], 75)

        IQR = Q3 - Q1

        

        outlier_step = 1.5 * IQR

        

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        

    return multiple_outliers

        

Outliers_to_drop = detect_outliers(df_train, 2, ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold'])



# train 데이터의 이상치를 탐색합니다.

# IQR(튜키의 방법)을 이용한 함수를 지정하여 이상치 탐색을 수행합니다.
df_train.loc[Outliers_to_drop]



# 이상치가 발견된 행을 확인합니다.
df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

df_train.shape



# 이상치들을 제거해주고, 결과를 확인합니다.

# 행의 수가 1338로 줄어든것을 확인할 수 있습니다.
for col in df_train.columns:

    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msperc)



# train 데이터 각 column의 결측치가 몇 %인지 확인합니다. 

# df_train[col].isnull().sum() : 해당 열의 결측치가 몇개인지 알 수 있게하는 문장입니다. (TRUE=1(결측치), FALSE=0으로 계산)

# df_train[col].shape[0] : 해당 열의 차원 (열이 지정되어 있으므로 행의 갯수를 보여줍니다.)

# 100 * (df_train[col].isnull().sum() / df_train[col].shape[0] : 위의 설명을 통해 %를 출력해주는 문장임을 알 수 있습니다.
for col in df_test.columns:

    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msperc)



# test 데이터도 확인해줍니다.

# train, test 모두 PoolQc 데이터가 가장 결측치가 많습니다.
missing = df_train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(figsize = (12,6))



# 직관적으로 확인하기 위해 barplot을 그려봅니다.
for col in numerical_feats:

    print('{:15}'.format(col), 

          'Skewness: {:05.2f}'.format(df_train[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(df_train[col].kurt())  

         )

    

# 수치형 변수의 Skewness(비대칭도), Kurtosis(첨도)를 확인합니다.

# 이는 분포가 얼마나 비대칭을 띄는가 알려주는 척도입니다. (비대칭도: a=0이면 정규분포, a<0 이면 오른쪽으로 치우침, a>0이면 왼쪽으로 치우침)

# 비대칭도와 첨도를 띄는 변수가 여럿 보입니다. Target Feature인 "SalePrice" 또한 약간의 정도를 보이는 것으로 보입니다.
corr_data = df_train[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

                      'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]



colormap = plt.cm.PuBu

sns.set(font_scale=1.0)



f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Numeric Features with Sale Price',y=1,size=18)

sns.heatmap(corr_data.corr(),square = True, linewidths = 0.1,

            cmap = colormap, linecolor = "white", vmax=0.8)



# Heat Map은 seaborn 덕분에 직관적으로 이해가 가능하여 변수 간 상관관계에 대하여 쉽게 알 수 있습니다.

# 또한 변수 간 다중 공선성을 감지하는 데 유용합니다.

# 대각선 열을 제외한 박스 중 가장 진한 파란색을 띄는 박스가 보입니다.

# 첫 번째는 'TotalBsmtSF'와 '1stFlrSF'변수의 관계입니다.

# 두 번째는 'Garage'와 관련한 변수를 나타냅니다. 

# 두 경우 모두 변수 사이의 상관 관계가 너무 강하여 다중 공선성(MultiColarisity) 상황이 나타날 수 있습니다. 

# 변수가 거의 동일한 정보를 제공하므로 다중 공선성이 실제로 발생한다는 결론을 내릴 수 있습니다.

# 또한 확인해야할 부분은 'SalePrice'와의 상관 관계입니다. 

# 'GrLivArea', 'TotalBsmtSF'및 'OverallQual'은 큰 관계를 보입니다. 

# 나머지 변수와의 상관 관계를 자세히 알아보기 위해 Zoomed Heat Map을 확인합니다.
k= 11

cols = corr_data.corr().nlargest(k,'SalePrice')['SalePrice'].index

print(cols)

cm = np.corrcoef(df_train[cols].values.T)

f , ax = plt.subplots(figsize = (12,10))

sns.heatmap(cm, vmax=.8, linewidths=0.1,square=True,annot=True,cmap=colormap,

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':14},yticklabels = cols.values)



# 가장 눈에 띄는 GarageCars와 GarageArea, TotalBsmtSF와 1stFlrSF는 서로 밀접하게 연관되어 있음을 알 수 있습니다.

# Target feature와 가장 밀접한 연관이 있는 feature는 'OverallQual', 'GrLivArea'및 'TotalBsmtSF'로 보입니다.

# 먼저 말했던 GarageCars와 GarageArea, TotalBsmtSF와 1stFlrSF, TotRmsAbvGrd와 GrLivArea는 모두 매우 유사한 정보를 포함하고 있으며 다중공선성이 나타난다고 할 수 있습니다. 

# SalePrice와 더 연관되어있는 변수인 GarageCars와 TotalBsmtSF, GrLivArea를 남기고 나머지는 이후에 버리도록 합니다.

# SalePrice와의 연관을 더 알아보기 위해 PairPlot을 그려보도록 합니다.
sns.set()

columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars','FullBath','YearBuilt','YearRemodAdd']

sns.pairplot(df_train[columns],size = 2 ,kind ='scatter',diag_kind='kde')

plt.show()



# 위의 Zoomed Heat Map에서 다중공선성을 보이는 변수 중 SalePrice와 연관이 덜 한 변수를 제외하고 PairPlot을 그립니다.

# 'TotalBsmtSF'와 'GrLiveArea'는 데이터 설명에서 알 수 있듯이 지하실의 면적과 생활공간 면적을 의미합니다.

# 'TotalBsmtSF'와 'GrLiveArea'의 plot을 보면 점들이 직선처럼 그려지고 대부분의 점은 해당 선 아래에 유지됩니다. 

# 이것은 일반적으로 지하실 면적이 지상 생활 면적과 같을 수 있지만, 더 크진 않기 때문에 보여지는 특징이라고 할 수 있습니다.

# 'SalePrice'와 'YearBuilt'의 plot을 보면 우상향 곡선을 보입니다.

# 이것은 전년도 대비 주택 가격 상승의 가속을 의미한다고 할 수 있습니다.
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(16,13))

OverallQual_scatter_plot = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis = 1)

sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

TotalBsmtSF_scatter_plot = pd.concat([df_train['SalePrice'],df_train['TotalBsmtSF']],axis = 1)

sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

GrLivArea_scatter_plot = pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis = 1)

sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

GarageCars_scatter_plot = pd.concat([df_train['SalePrice'],df_train['GarageCars']],axis = 1)

sns.regplot(x='GarageCars',y = 'SalePrice',data = GarageCars_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

FullBath_scatter_plot = pd.concat([df_train['SalePrice'],df_train['FullBath']],axis = 1)

sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

YearBuilt_scatter_plot = pd.concat([df_train['SalePrice'],df_train['YearBuilt']],axis = 1)

sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

YearRemodAdd_scatter_plot = pd.concat([df_train['SalePrice'],df_train['YearRemodAdd']],axis = 1)

YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')



# Target Feature "SalePrice"와 가장 밀접한 연관이 있다고 판단됐던 변수들의 Scatter Plot을 그립니다.

# OverallQual, GarageCars, Fullbath와 같은 변수들은 실제로는 범주형 데이터의 특징을 보인다고 할 수 있습니다. (등급, 갯수 등을 의미하기 때문)
for catg in list(categorical_feats) :

    print(df_train[catg].value_counts())

    print('#'*50)

    

# 범주형 변수들과 각 범주들을 확인합니다.
li_cat_feats = list(categorical_feats)

nr_rows = 15

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=df_train["SalePrice"], data=df_train, ax = axs[r][c])

    

plt.tight_layout()    

plt.show()



# BoxPlot을 그려 Categorical Feature와 SalePrice의 관계를 확인합니다.

# 일부 범주는 다른 범주보다 SalePrice와 관련하여 더 다양하게 보입니다. 

# Neighborhood 변수는 주택 가격 편차가 매우 크므로 영향이 크다고 생각됩니다.

# SaleType 또한 마찬가지입니다.

# 또한 수영장이 있으면 가격이 크게 증가하는 것 같습니다.

# 정리하면 SalePrice에 영향을 많이 끼치는 변수로는 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual', 'CentralAir', 'Electrical', 'KitchenQual', 'SaleType' 등이 있습니다.
num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars',

                   'FullBath','YearBuilt','YearRemodAdd']



num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',

                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath',

                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',

                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']



catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 

                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 

                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 

                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 

                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 

                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 

                  'SaleCondition' ]



# 편의를 위해 SalePrice와 관련이 큰 변수와 아닌 변수를 분리해놓습니다.

# "Id"는 submission 때문에 따로 빼놓겠습니다.
f, ax = plt.subplots(1, 1, figsize = (10,6))

g = sns.distplot(df_train["SalePrice"], color = "b", label="Skewness: {:2f}".format(df_train["SalePrice"].skew()), ax=ax)

g = g.legend(loc = "best")



print("Skewness: %f" % df_train["SalePrice"].skew())

print("Kurtosis: %f" % df_train["SalePrice"].kurt())



# Target Feature인 SalePrice의 비대칭도와 첨도를 확인합니다. 

# 그래프와 수치를 확인하면 정상적으로 분포되지 않는것을 확인할 수 있습니다. 

# 예측의 정확도를 높히기 위해 로그 변환을 수행합니다.
df_train["SalePrice_Log"] = df_train["SalePrice"].map(lambda i:np.log(i) if i>0 else 0)



f, ax = plt.subplots(1, 1, figsize = (10,6))

g = sns.distplot(df_train["SalePrice_Log"], color = "b", label="Skewness: {:2f}".format(df_train["SalePrice_Log"].skew()), ax=ax)

g = g.legend(loc = "best")



print("Skewness: %f" % df_train['SalePrice_Log'].skew())

print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())



df_train.drop('SalePrice', axis= 1, inplace=True)



# kewness, Kurtosis를 없애주기 위해 로그를 취해줍니다.

# Log변환을 수행한 새로운 feature "SalePrice_Log"를 만들고 전 Feature인 "Saleprice"를 버려줍니다.

# 로그를 취해준 그래프와 수치가 바뀐 모습을 볼 수 있습니다. (정규근사화)
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']



for col in cols_fillna:

    df_train[col].fillna('None',inplace=True)

    df_test[col].fillna('None',inplace=True)

    

# 위에서 설명한 바와 같이 '없다'의 의미를 갖는 변수들입니다.

# NaN을 없다는 의미의 None으로 대체해줍니다.
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)



# 결측치의 처리 정도를 확인해 줍니다.
df_train.fillna(df_train.mean(), inplace=True)

df_test.fillna(df_test.mean(), inplace=True)



# 나머지 결측치들은 평균값으로 대체하겠습니다.
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)



# 다시 확인해보면 결측치가 사라진 것을 알 수 있습니다.
df_train.isnull().sum().sum(), df_test.isnull().sum().sum()
id_test = df_test['Id']



to_drop_num  = num_weak_corr

to_drop_catg = catg_weak_corr



cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 



for df in [df_train, df_test]:

    df.drop(cols_to_drop, inplace= True, axis = 1)

    

# SalePrice와의 상관관계가 약한 모든 변수를 삭제합니다.
df_train.head()



# 삭제가 잘 진행되었는지 확인합니다.
catg_list = catg_strong_corr.copy()

catg_list.remove('Neighborhood')



for catg in catg_list :

    sns.violinplot(x=catg, y=df_train["SalePrice_Log"], data=df_train)

    plt.show()

    

# 각 범주들을 개별로 살펴봅니다.
fig, ax = plt.subplots()

fig.set_size_inches(16, 5)

sns.violinplot(x='Neighborhood', y=df_train["SalePrice_Log"], data=df_train, ax=ax)

plt.xticks(rotation=45)

plt.show()



# 범주가 가장 많은 Neighborhood 변수도 살펴봅니다.
for catg in catg_list :

    g = df_train.groupby(catg)["SalePrice_Log"].mean()

    print(g)

    

# 각 범주들에 해당되는 SalePrice_Log 평균을 살펴봅니다.  
# 'MSZoning'

msz_catg2 = ['RM', 'RH']

msz_catg3 = ['RL', 'FV'] 





# Neighborhood

nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']

nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']



# Condition2

cond2_catg2 = ['Norm', 'RRAe']

cond2_catg3 = ['PosA', 'PosN'] 



# SaleType

SlTy_catg1 = ['Oth']

SlTy_catg3 = ['CWD']

SlTy_catg4 = ['New', 'Con']



# 수치형 변환을 위해 Violinplot과 SalePrice_Log 평균을 참고하여 각 변수들의 범주들을 그룹화 합니다.
for df in [df_train, df_test]:

    

    df['MSZ_num'] = 1  

    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    

    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        

    

    df['NbHd_num'] = 1       

    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    

    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    



    df['Cond2_num'] = 1       

    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    

    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    

    

    df['Mas_num'] = 1       

    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 

    

    df['ExtQ_num'] = 1       

    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     

    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     

    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     

   

    df['BsQ_num'] = 1          

    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     

    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     

 

    df['CA_num'] = 0          

    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    



    df['Elc_num'] = 1       

    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 





    df['KiQ_num'] = 1       

    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     

    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     

    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      

    

    df['SlTy_num'] = 2       

    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  

    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  

    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4 

    

# 각 범주별로 수치형 변환을 실행합니다.
new_col_HM = df_train[['SalePrice_Log', 'MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']]



colormap = plt.cm.PuBu

plt.figure(figsize=(10, 8))

plt.title("Correlation of New Features", y = 1.05, size = 15)

sns.heatmap(new_col_HM.corr(), linewidths = 0.1, vmax = 1.0,

           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 12})



# 변환하여 새로 만들어진 numerical feature들 또한 Heat Map을 그려봅니다.

# NbHd_num, ExtQ_num, BsQ_num, KiQ_num를 제외하고는 SalePrice_Log와 큰 상관관계가있는 열은 거의 없습니다.
df_train.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)

df_test.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)



# 기존 범주형 변수와 새로 만들어진 수치형 변수 역시 유의하지 않은 것들은 삭제합니다. 
df_train.head()



# 완벽히 삭제되어 유의하다고 판단되는 수치형 변수만 남았습니다.
df_test.head()
from sklearn.model_selection import train_test_split

from sklearn import metrics



X_train = df_train.drop("SalePrice_Log", axis = 1).values

target_label = df_train["SalePrice_Log"].values

X_test = df_test.values

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.2, random_state = 2000)



# Test하기 전 Validation 과정을 겨처줍니다.

# train데이터의 20%를 validation으로 주고 80%을 train으로 남겨주어 분리해줍니다.
import xgboost

regressor = xgboost.XGBRegressor(colsample_bytree = 0.4603, learning_rate = 0.06, min_child_weight = 1.8,

                                 max_depth= 3, subsample = 0.52, n_estimators = 2000,

                                 random_state= 7, ntrhead = -1)

regressor.fit(X_tr,y_tr)



# XGBoost 모델을 만들어줍니다.
y_hat = regressor.predict(X_tr)



plt.scatter(y_tr, y_hat, alpha = 0.2)

plt.xlabel('Targets (y_tr)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

plt.show()



# 예측 된 y 값 (y_hat)에 대한 Scatter Plot을 그려봅니다.
regressor.score(X_tr,y_tr)
y_hat_test = regressor.predict(X_vld)





plt.scatter(y_vld, y_hat_test, alpha=0.2)

plt.xlabel('Targets (y_vld)',size=18)

plt.ylabel('Predictions (y_hat_test)',size=18)

plt.show()



# validation으로 예측해봅니다.
regressor.score(X_vld,y_vld)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_tr, y = y_tr, cv = 10)



# k-fold validation을 수행합니다.
print(accuracies.mean())

print(accuracies.std())



# 정확도를 확인해봅니다.
use_logvals = 1



pred_xgb = regressor.predict(X_test)



sub_xgb = pd.DataFrame()

sub_xgb['Id'] = id_test

sub_xgb['SalePrice'] = pred_xgb



if use_logvals == 1:

    sub_xgb['SalePrice'] = np.exp(sub_xgb['SalePrice']) 



sub_xgb.to_csv('xgb.csv',index=False)



# use_logvals는 Log를 취해준 Target feature을 exp해주기 위해 사용되는 스위치 역할입니다.

# 제대로 된 예측을 위해 학습 후 Log변환을 풀어줘야하기 때문입니다.

# 이 셀의 코드를 통해 submission까지 완료하게됩니다.