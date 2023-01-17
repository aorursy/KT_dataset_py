#importing necessary models and libraries



#Math tools

from scipy import stats

from scipy.stats import skew,norm  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

import scipy.stats as stats





#Visualizing tools

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.patches import Rectangle

sns.set(style="ticks")



#preprocessing tools

import pandas as pd

import numpy as np



from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_size = train.shape[0]

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")



import warnings

warnings.filterwarnings(action="ignore")
#skewness and kurtosis

print("Skewness: {}".format(train['SalePrice'].skew()))

print("Kurtosis: {}".format(train['SalePrice'].kurt()))

print("--------------------------------------")

print(train['SalePrice'].describe())

# Customizing the layout

def multi_plot(feature):

    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



    ax1 = fig.add_subplot(grid[0, :2])

    ax1.set_title('Distplot')

    sns.distplot(train.loc[:,feature], fit=norm,color="mediumseagreen", ax = ax1)



    ax2 = fig.add_subplot(grid[1, :2])

    ax2.set_title('QQ_plot')

    stats.probplot(train.loc[:,feature], plot = ax2)



    ax3= fig.add_subplot(grid[0, 2:])

    ax3.set_title('Scatter Plot')

    sns.scatterplot(range(train.shape[0]), train[feature].values,color='orangered')



    ax4 = fig.add_subplot(grid[1, 2:])

    ax4.set_title('Box Plot')

    sns.boxplot(train.loc[:,feature], orient='v', ax = ax4 , color='darkorange');
multi_plot('SalePrice')
f,ax = plt.subplots(1,2,figsize=(16,4))

sns.boxplot(train['GrLivArea'],ax=ax[0],color="darkorange")

plt.scatter(train['GrLivArea'],train['SalePrice'],color='#9b59b6')

#outlier detection

plt.axvline(x=4600,color='r')

plt.xlabel('GrLiveArea')

plt.ylabel('SalePrice')

plt.show()

#outlier removal

train.drop(train[train['GrLivArea']>4500].index,axis=0,inplace=True)
f,ax = plt.subplots(1,2,figsize=(16,4))

sns.boxplot(train['GrLivArea'],ax=ax[0],color="springgreen")

plt.scatter(train['GrLivArea'],train['SalePrice'],color='limegreen')

plt.xlabel('GrLiveArea')

plt.ylabel('SalePrice')

plt.show()
f,ax = plt.subplots(1,1,figsize=(16,4))

# Uncomment the below line and see why Swarmplot was suggested instead of Scatterplot,

# when using on discrete or categorical variable.

# sns.scatterplot('OverallQual','SalePrice', data = train)

sns.swarmplot('OverallQual','SalePrice', data = train , palette="Set2")





#outlier detection

ax.add_patch(Rectangle((2.5,200000),1,100000 ,linewidth=5,edgecolor='b',facecolor='none'))

plt.xlabel('OverallQual')

plt.ylabel('SalePrice')

plt.show()
outlier_index = train[(train['OverallQual'] == 4) & (train['SalePrice'] > 200000)].index



# *outlier

# We can see OverallQual increases along with SalePrice and the pattern shows that each qual level covers the previous levels completely 

# but on index 457(highlighted rectangle) looks different. so we will remove it .



# outlier removal

train.drop(outlier_index,axis=0,inplace=True)
# Finding numeric features

numeric_cols = train.select_dtypes(exclude='object').columns

numeric_cols_length = len(numeric_cols)  



fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))

plt.subplots_adjust(right=2)

plt.subplots_adjust(top=2)



# skiped Id and saleprice feature

for i in range(1,numeric_cols_length-1):

    feature = numeric_cols[i]

    plt.subplot(numeric_cols_length, 3, i)

    sns.scatterplot(x=feature, y='SalePrice', data=train,color='crimson')

    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

    plt.ylabel('SalePrice', size=15, labelpad=12.5)

           

plt.show()

corr = train.select_dtypes(include='number').corr()

plt.figure(figsize=(16,6))

corr_saleprice = corr['SalePrice'].sort_values(ascending=False)[1:]

ax = sns.barplot(corr_saleprice.index,corr_saleprice.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
# # Correlation of top 10 feature with saleprice

corWithSalePrice = train.corr().nlargest(10,'SalePrice')['SalePrice'].index

f , ax = plt.subplots(figsize = (18,12))

corr = train[corWithSalePrice].corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=0.8,square=True,annot=True,cmap="YlGnBu")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
sns.lmplot('TotalBsmtSF','1stFlrSF',data=train,height=7, aspect=1.6, palette="Set1" , 

           line_kws={'color': 'salmon'},scatter_kws={'color': 'turquoise'})
plt.figure(figsize=(16,8))

sns.boxplot(x='OverallQual', y='TotalBsmtSF',data=train,palette='Set2')

sns.swarmplot(x='OverallQual', y='TotalBsmtSF',data=train,palette='Set1')

plt.show()
f,ax = plt.subplots(1,2,figsize=(16,4))

sns.stripplot('TotRmsAbvGrd','GrLivArea',data=train,ax=ax[0])

sns.stripplot('GarageCars','GarageArea',data=train,ax = ax[1], palette="Set2")

plt.show()
f,ax = plt.subplots(1,1,figsize=(16,4))

sns.scatterplot(x='LotArea', y='SalePrice',data=train,color='mediumspringgreen')



#outlier detection

ax.add_patch(Rectangle((200000,320000),25000,100000 ,linewidth=5,edgecolor='orangered',facecolor='none'))

plt.show()

outlier_index = train[(train['LotArea'] > 200000) & (train['SalePrice'] > 300000)]

year_diff = outlier_index['YearBuilt'] - outlier_index['YrSold']
f,ax = plt.subplots(1,1,figsize=(16,4))

sns.scatterplot(x='LotArea', y='SalePrice',data=train[(train['YrSold'] - train['YearBuilt']) > 40 ],color='mediumseagreen')

ax.add_patch(Rectangle((150000,200000),70000,190000 ,linewidth=5,edgecolor='orangered',facecolor='none'))
# *outlier

# Surprisingly, We got two more outlier. 

# Filtered data are more than 40 years older buildings.

# Most of the values plotted within 100000 area. So, it is safe to remove these outliers.

# outlier removal

outlier_index = train[(train['LotArea'] > 150000) & (train['SalePrice']>200000) & (train['SalePrice'] < 400000) ].index

train.drop(outlier_index,axis=0,inplace=True)
sns.jointplot("TotalBsmtSF", "SalePrice", data=train,height=8,color='rebeccapurple')
sns.regplot('TotalBsmtSF','SalePrice',data=train,color='dodgerblue')
f,ax = plt.subplots(1,3,figsize=(16,4))

sns.pointplot(x=train["Alley"], y=train["SalePrice"],jitter=True,ax=ax[0],color='orchid');

sns.boxplot(x='Alley', y='SalePrice',data=train,ax=ax[1],palette='Set2')

sns.stripplot(x='Alley', y='SalePrice',data=train,ax=ax[2])
# Pie chart

mszoning = train['MSZoning'].value_counts()

labels = mszoning.index

sizes = mszoning.values

explode = (0.1, 0, 0, 0,0)

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red']

patches, texts = plt.pie(sizes, colors=colors,explode=explode, shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()





# Donut chart

PavedDrive = train['PavedDrive'].value_counts()

labels = PavedDrive.index

sizes = PavedDrive.values

explode = (0.05,0.05,0.05)

fig1, ax1 = plt.subplots()

patches = ax1.pie(sizes, pctdistance=0.8,explode = explode, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
# After removing the some of outliers, you can see that skewness is reduced. Still Saleprice is not normally distributed.

multi_plot('SalePrice')

#skewness and kurtosis

print("Skewness: {}".format(train['SalePrice'].skew()))

print("Kurtosis: {}".format(train['SalePrice'].kurt()))
#Log - transformation

train['SalePrice_log'] = np.log1p(train['SalePrice'])

multi_plot('SalePrice_log')

#skewness and kurtosis

print("Skewness: {}".format(train['SalePrice_log'].skew()))

print("Kurtosis: {}".format(train['SalePrice_log'].kurt()))
# this will remove the overfitted features



def remove_overfit_features(df,weight):

    overfit = []

    for i in df.columns:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df) * 100 > weight:

            overfit.append(i)

    overfit = list(overfit)

    return overfit





overfitted_features = remove_overfit_features(train,99)

train.drop(overfitted_features,inplace=True,axis=1)

test.drop(overfitted_features,inplace=True,axis=1)

train_labels = train['SalePrice_log']

train_features = train.drop(['SalePrice','SalePrice_log'], axis=1)

test_features = test



# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset

all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.drop('Id',inplace=True,axis=1)

all_features.shape
#visualize missing data

missing_value = all_features.isnull().sum().sort_values(ascending=False) / len(all_features) * 100

missing_value = missing_value[missing_value != 0]

missing_value = pd.DataFrame({'Missing value' :missing_value,'Type':missing_value.index.map(lambda x:all_features[x].dtype)})

missing_value.plot(kind='bar',figsize=(16,4))

plt.show()
print("Total No. of missing value {} before Imputation".format(sum(all_features.isnull().sum())))

def fill_missing_values():

 

    fillSaleType = all_features[all_features['SaleCondition'] == 'Normal']['SaleType'].mode()[0]

    all_features['SaleType'].fillna(fillSaleType,inplace=True)



    fillElectrical = all_features[all_features['Neighborhood']=='Timber']['Electrical'].mode()[0]

    all_features['Electrical'].fillna(fillElectrical,inplace=True)



    exterior1_neighbor = all_features[all_features['Exterior1st'].isnull()]['Neighborhood'].values[0]

    fillExterior1 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]

    all_features['Exterior1st'].fillna(fillExterior1,inplace=True)



    exterior2_neighbor = all_features[all_features['Exterior2nd'].isnull()]['Neighborhood'].values[0]

    fillExterior2 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]

    all_features['Exterior2nd'].fillna(fillExterior2,inplace=True)



    bsmtNeigh = all_features[all_features['BsmtFinSF1'].isnull()]['Neighborhood'].values[0]

    fillBsmtFinSf1 = all_features[all_features['Neighborhood'] == bsmtNeigh]['BsmtFinSF1'].mode()[0]

    all_features['BsmtFinSF1'].fillna(fillBsmtFinSf1,inplace=True)



    kitchen_grade = all_features[all_features['KitchenQual'].isnull()]['KitchenAbvGr'].values[0]

    fillKitchenQual = all_features[all_features['KitchenAbvGr'] == kitchen_grade]['KitchenQual'].mode()[0]

    all_features['KitchenQual'].fillna(fillKitchenQual,inplace=True)

        

    all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

       

    all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure',

                'BsmtFinType1', 'BsmtFinType2','PoolQC']:

        all_features[col] = all_features[col].fillna('None')

    

    categorical_cols =  all_features.select_dtypes(include='object').columns

    all_features[categorical_cols] = all_features[categorical_cols].fillna('None')

    

    numeric_cols = all_features.select_dtypes(include='number').columns

    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)

    

    all_features['Shed'] = np.where(all_features['MiscFeature']=='Shed', 1, 0)

    

    #GarageYrBlt -  missing values there for the building which has no Garage, imputing 0 makes huge difference with other buildings,

    #imputing mean doesn't make sense since there is no Garage. So we'll drop it

    all_features.drop(['GarageYrBlt','MiscFeature'],inplace=True,axis=1)

    

    all_features['QualitySF'] = all_features['GrLivArea'] * all_features['OverallQual']



fill_missing_values()



print("Total No. of missing value {} after Imputation".format(sum(all_features.isnull().sum())))
all_features = all_features.drop(['PoolQC',], axis=1)
# converting some numeric features to string

all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['YrSold'] = all_features['YrSold'].astype(str)

all_features['MoSold'] = all_features['MoSold'].astype(str)





# Filter the skewed features

numeric = all_features.select_dtypes(include='number').columns

skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
# Normalize skewed features using boxcox

for i in skew_index:

    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)

all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']



all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']

all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +

                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])

all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +

                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))

all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +

                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +

                              all_features['WoodDeckSF'])
def booleanFeatures(columns):

    for col in columns:

        all_features[col+"_bool"] = all_features[col].apply(lambda x: 1 if x > 0 else 0)

booleanFeatures(['GarageArea','TotalBsmtSF','2ndFlrSF','Fireplaces','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'])  

def logs(columns):

    for col in columns:

        all_features[col+"_log"] = np.log(1.01+all_features[col])  



log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','MiscVal','YearRemodAdd','TotalSF']



logs(log_features)

def squares(columns):

    for col in columns:

        all_features[col+"_sq"] =  all_features[col] * all_features[col]



squared_features = ['GarageCars_log','YearRemodAdd', 'LotFrontage_log', 'TotalBsmtSF_log', '2ndFlrSF_log', 'GrLivArea_log' ]



squares(squared_features)
# There is a natural order in their values for few categories, so converting them to numbers gives more meaning

quality_map = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

quality_cols = ['BsmtQual', 'BsmtCond','ExterQual', 'ExterCond','FireplaceQu','GarageQual', 'GarageCond','KitchenQual','HeatingQC']

for col in quality_cols:

    all_features[col] = all_features[col].replace(quality_map)



all_features['BsmtExposure'] = all_features['BsmtExposure'].replace({"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3})



all_features["PavedDrive"] =all_features["PavedDrive"].replace({"N" : 0, "P" : 1, "Y" : 2})



bsmt_ratings = {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}

bsmt_col = ['BsmtFinType1','BsmtFinType2']

for col in bsmt_col:

    all_features[col] = all_features[col].replace(bsmt_ratings)



    

all_features["OverallScore"]   = all_features["OverallQual"] * all_features["OverallCond"]

all_features["GarageScore"]    = all_features["GarageQual"] * all_features["GarageCond"]

all_features["ExterScore"]     = all_features["ExterQual"] * all_features["ExterCond"]

all_features = pd.get_dummies(all_features).reset_index(drop=True)
X = all_features.iloc[:len(train_labels), :]

X_test = all_features.iloc[len(train_labels):, :]
overfitted_features = remove_overfit_features(X,99)

X = X.drop(overfitted_features, axis=1)

X_test = X_test.drop(overfitted_features, axis=1)