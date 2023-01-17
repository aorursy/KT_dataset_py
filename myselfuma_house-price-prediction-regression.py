
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from IPython.display import HTML
import seaborn as sns
from sklearn.cross_validation import train_test_split
import sklearn.tree
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor

train_df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_df=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#train_df.info()
#Seperating based on data types. We will re-arrange after analysing the data
Data_types=pd.DataFrame(train_df.dtypes,columns=['datatype'])
Categorical_Features=Data_types.loc[Data_types['datatype']=='object']
Continuous_Features=Data_types.loc[Data_types['datatype']!='object']
del Data_types
Continuous_Features['Count']=np.nan #count of the data present
Continuous_Features['Missing_pc']=np.nan #Missing percentage
Continuous_Features['Cardinality']=np.nan 
Continuous_Features['Minimum']=np.nan
Continuous_Features['First_Qrt']=np.nan #First Quartile
Continuous_Features['Mean']=np.nan
Continuous_Features['Median']=np.nan
Continuous_Features['Third_Qrt']=np.nan #Third Quartile
Continuous_Features['Maximum']=np.nan
Continuous_Features['Std_dev']=np.nan #Standard Deviation
for i in Continuous_Features.index:
    stat_df=pd.DataFrame(train_df[i].describe())   
    Continuous_Features.loc[i,'Count']=stat_df.loc['count',i]
    Continuous_Features.loc[i,'Missing_pc']=((1460-(Continuous_Features.loc[i,'Count']))/1460)*100
    Continuous_Features.loc[i,'Cardinality']=train_df[i].unique().shape[0]
    Continuous_Features.loc[i,'Minimum']=stat_df.loc['min',i]
    Continuous_Features.loc[i,'First_Qrt']=stat_df.loc['25%',i]
    Continuous_Features.loc[i,'Mean']=stat_df.loc['mean',i]
    Continuous_Features.loc[i,'Median']=stat_df.loc['50%',i]
    Continuous_Features.loc[i,'Third_Qrt']=stat_df.loc['75%',i]
    Continuous_Features.loc[i,'Maximum']=stat_df.loc['max',i]
    Continuous_Features.loc[i,'Std_dev']=stat_df.loc['std',i]

Continuous_Features.head()

    
Categorical_Features['Count']=np.nan #count of the data present
Categorical_Features['Missing_pc']=np.nan #Missing percentage
Categorical_Features['Cardinality']=np.nan 
Categorical_Features['Mode']=np.nan
Categorical_Features['Mode_Freq']=np.nan #Mode Frequency
Categorical_Features['Mode_pc']=np.nan #Mode Percentage
Categorical_Features['Sec_Mode']=np.nan # Second Mode
Categorical_Features['Sec_Mod_Freq']=np.nan # Second Mode Frequency
Categorical_Features['Sec_Mode_pc']=np.nan #Second Mode Percentage

for i in Categorical_Features.index:
    stat_df=pd.DataFrame(train_df[i].describe())   
    Categorical_Features.loc[i,'Count']=stat_df.loc['count',i]
    Categorical_Features.loc[i,'Missing_pc']=((1460-(stat_df.loc['count',i]))/1460)*100
    Categorical_Features.loc[i,'Cardinality']=stat_df.loc['unique',i]
    Categorical_Features.loc[i,'Mode']=stat_df.loc['top',i]
    Categorical_Features.loc[i,'Mode_Freq']=stat_df.loc['freq',i]
    Categorical_Features.loc[i,'Mode_pc']=(stat_df.loc['freq',i]/1460)*100
    stat_df=pd.DataFrame(train_df.loc[train_df[i]!=stat_df.loc['top',i]][i].describe())
    Categorical_Features.loc[i,'Sec_Mode']=stat_df.loc['top',i]
    Categorical_Features.loc[i,'Sec_Mod_Freq']=stat_df.loc['freq',i]
    Categorical_Features.loc[i,'Sec_Mode_pc']=(stat_df.loc['freq',i]/1460)*100

Categorical_Features.head()
fig,axes=plt.subplots(nrows=19,ncols=2,sharex=False,sharey=False,figsize=(20,100))
j=0
k=0
for i in Continuous_Features.index:
    dens=train_df.groupby([i])[i].agg(['count'])
    s=dens['count'].sum()
    c=np.ceil((dens['count'].count())/10)
    dens['density']= dens['count']/s
    dens.drop(columns={'count'},inplace=True)
    dens.plot(kind='bar',ax=axes[k,j])
    if c>15:
        for index, label in enumerate(axes[k,j].xaxis.get_ticklabels()):
            if index % c !=0:
                label.set_visible(False)          
    j=(j+1)%2
    if j==0:
        k+=1
#fig.suptitle("How the data is spread")
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()
fig, ax = plt.subplots(nrows=12, ncols=3,figsize=(20,100))
ax=ax.flatten()
cols = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
        'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
        'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
        'PoolArea','MiscVal','MoSold','YrSold']
#colors=['#415952', '#f35134', '#243AB5', '#243AB5']
j=0

for i in ax:
    if j==0:
        i.set_ylabel('SalePrice')
    i.scatter(train_df[cols[j]], train_df['SalePrice'],  alpha=0.5)
    i.set_xlabel(cols[j])
    i.set_title('Pearson: %s'%train_df.corr().loc[cols[j]]['SalePrice'].round(2))
    j+=1

#plt.suptitle('CONTINUOUS', fontsize=20)
plt.show()
train_df.corr()[["OverallQual","SalePrice"]]
import seaborn as sns
fig, ax = plt.subplots(nrows=15, ncols=3,figsize=(20,80))
ax=ax.flatten()
cols = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
        'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
        'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
        'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature',
        'SaleType','SaleCondition']
#colors=['#415952', '#f35134', '#243AB5', '#243AB5']
j=0
for i in ax:
    if j==43:
        break
    if j==0:
        i.set_ylabel('SalePrice')
    sns.boxplot(x=cols[j],y="SalePrice", data=train_df,ax=i)
    i.set_xlabel(cols[j])
    #i.set_title('Pearson: %s'%train_df.corr().loc[cols[j]]['SalePrice'].round(2))
    j+=1

plt.show()
HTML('<table><tr><th><b></b></th><th><b>Data Quality Issues</b></th><th><b>Potential Handling Strategies</b></th></tr><tr><th><b>Alley</b></th><th>Missing values >60%</th><th>To be removed</th></tr><tr><th><b>Utilities</b></th><th>cardinality 1 for - NoSeWa</th><th>To be removed</th></tr><tr><th><b>MasVnrType</b></th><th>has missing value</th><th>To be replace by -None</th></tr><tr><th><b>BsmtQual</b></th><th>has missing value</th><th>Because Basement not present</th></tr><tr><th><b>BsmtCond</b></th><th>has missing value</th><th>Because Basement not present</th></tr><tr><th><b>BsmtExposure</b></th><th>has missing value</th><th>Because Basement not present</th></tr><tr><th><b>BsmtFinType1</b></th><th>has missing value</th><th>Because Basement not present</th></tr><tr><th><b>BsmtFinType2</b></th><th>has missing value</th><th>Because Basement not present</th></tr><tr><th><b>FireplaceQu</b></th><th>has Missing values</th><th>Because Fireplace not present</th></tr><tr><th><b>GarageType</b></th><th>has missing values</th><th>Because Garage not present</th></tr><tr><th><b>GarageFinish</b></th><th>has missing values</th><th>Because Garage not present</th></tr><tr><th><b>GarageQual</b></th><th>has missing values</th><th>Because Garage not present</th></tr><tr><th><b>GarageCond</b></th><th>has missing values</th><th>Because Garage not present</th></tr><tr><th><b>PoolQC</b></th><th>Missing values >60%</th><th>Because Pool not present</th></tr><tr><th><b>Fence</b></th><th>Missing values >60%</th><th>Because Fence not present</th></tr><tr><th><b>MiscFeature</b></th><th>Missing values >60%</th><th>Remove the field</th></tr></table><table><tr><th><b>FEATURES SET -2 </b></th><th></th><th></th></tr><tr><th><b>MSZoning</b></th><th>has correlation</th><th></th></tr><tr><th><b>Street</b></th><th></th><th>can be removed</th></tr><tr><th><b>Condition2</b></th><th>If condition2 is RRNn or Feedr the price will be low. If it is PosN the price is high.Very few records present for that.</th><th>Condition2 can be removed</th></tr><tr><th><b>HouseStyle</b></th><th>has correlation</th><th></th></tr><tr><th><b>ExterQual</b></th><th>has correlation</th><th></th></tr><tr><th><b>HeatingQC</b></th><th>has correlation</th><th></th></tr><tr><th><b>CentralAir</b></th><th>has correlation</th><th></th></tr><tr><th><b>KitchenQual</b></th><th>has correlation</th><th></th></tr></table>')
import scipy.stats as stats
R = train_df["OverallQual"].astype(str)
cols = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope',
        'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
        'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
        'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature',
        'SaleType','SaleCondition']


for i in cols:
    C = train_df[i].astype(str)
    tab=pd.crosstab(R,C)
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    print("p-value: OverAllQualtiy vs %s"%i+" = %s"%p)

#take alook into some records
R = train_df["OverallQual"].astype(str)
C = train_df['Neighborhood'].astype(str)
pd.crosstab(R,C)
from IPython.display import Image
Image("../input/hp-analysis/FE.png")
all_feature_df=train_df.copy()
encode={'NA' : 0,'Po' : 1,'Fa' : 2,'TA' : 3,'Gd' : 4,'Ex' : 5,'Gtl' : 0,'Mod' : 1,'Sev' : 2,'Y': 1,'N': 0,'P':0.5,'Fin' : 3,
        'RFn' : 2,'UnF' : 1,'LwQ' : 2,'Rec' : 3,'BLQ' : 4,'ALQ' : 5,'GLQ' : 6,'MnWw' : 1,'GdWo' : 2,'MnPrv' : 3, 'GdPrv' : 4,'Av':3,'Mn':2,'No':1,
       'IR3':1,'IR2':2,'Reg':3,'Low':1,'Lvl':2,'Bnk':3,'HLS':4}

all_feature_df['SalePrice']=all_feature_df['SalePrice']-all_feature_df['MiscVal']
all_feature_df['YYMOSold']=all_feature_df['YrSold'].astype('str')+all_feature_df['MoSold'].astype('str')
all_feature_df['Age']=(all_feature_df['YrSold']-all_feature_df['YearBuilt'])+(all_feature_df['MoSold']/12)

all_feature_df['OtherArea']=all_feature_df['WoodDeckSF']+all_feature_df['OpenPorchSF']+all_feature_df['EnclosedPorch']+all_feature_df['3SsnPorch']\
+all_feature_df['ScreenPorch']+all_feature_df['PoolArea']+all_feature_df['GarageArea']

#all_feature_df['LotAreaRemain']=all_feature_df['LotArea']-all_feature_df['1stFlrSF']-all_feature_df['OtherArea']

all_feature_df['ExterQual']=all_feature_df['ExterQual'].map(encode)
all_feature_df['ExterCond']=all_feature_df['ExterCond'].map(encode)
all_feature_df['HeatingQC']=all_feature_df['HeatingQC'].map(encode)
all_feature_df['BsmtQual']=all_feature_df['BsmtQual'].map(encode)
all_feature_df['CentralAir']=all_feature_df['CentralAir'].map(encode)
#all_feature_df['GarageFinish']=all_feature_df['GarageFinish'].map(encode)
all_feature_df['BsmtFinType1']=all_feature_df['BsmtFinType1'].map(encode)
all_feature_df['BsmtCond']=all_feature_df['BsmtCond'].map(encode)
all_feature_df['BsmtExposure']=all_feature_df['BsmtExposure'].map(encode)
#all_feature_df['PavedDrive']=all_feature_df['PavedDrive'].map(encode)
#all_feature_df['PoolQC']=all_feature_df['PoolQC'].map(encode)
all_feature_df['KitchenQual']=all_feature_df['KitchenQual'].map(encode)


all_feature_df=pd.get_dummies(all_feature_df,columns=['Neighborhood'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['MSZoning'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['SaleCondition'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Condition1'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Condition2'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Functional'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Exterior1st'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Exterior2nd'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Street'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Alley'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['MasVnrType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Heating'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Electrical'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['HouseStyle'])


all_feature_df['Condition1_Artery']=((all_feature_df['Condition1_Artery']+all_feature_df['Condition2_Artery'])%2)+1
all_feature_df['Condition1_Feedr']=((all_feature_df['Condition1_Feedr']+all_feature_df['Condition2_Feedr'])%2)+1
all_feature_df['Condition1_Norm']=((all_feature_df['Condition1_Norm']+all_feature_df['Condition2_Norm'])%2)+1
all_feature_df['Condition1_PosA']=((all_feature_df['Condition1_PosA']+all_feature_df['Condition2_PosA'])%2)+1
all_feature_df['Condition1_PosN']=((all_feature_df['Condition1_PosN']+all_feature_df['Condition2_PosN'])%2)+1
all_feature_df['Condition1_RRAe']=((all_feature_df['Condition1_RRAe']+all_feature_df['Condition2_RRAe'])%2)+1
all_feature_df['Condition1_RRAn']=((all_feature_df['Condition1_RRAn']+all_feature_df['Condition2_RRAn'])%2)+1
all_feature_df['Condition1_RRNn']=((all_feature_df['Condition1_RRNn']+all_feature_df['Condition2_RRNn'])%2)+1

#all_feature_df['Exterior1st_AsbShng']=((all_feature_df['Exterior1st_AsbShng']+all_feature_df['Exterior2nd_AsbShng'])%2)+1
#all_feature_df['Exterior1st_AsphShn']=((all_feature_df['Exterior1st_AsphShn']+all_feature_df['Exterior2nd_AsphShn'])%2)+1
#all_feature_df['Exterior1st_CBlock']=((all_feature_df['Exterior1st_CBlock']+all_feature_df['Exterior2nd_CBlock'])%2)+1
#all_feature_df['Exterior1st_HdBoard']=((all_feature_df['Exterior1st_HdBoard']+all_feature_df['Exterior2nd_HdBoard'])%2)+1
#all_feature_df['Exterior1st_MetalSd']=((all_feature_df['Exterior1st_MetalSd']+all_feature_df['Exterior2nd_MetalSd'])%2)+1
#all_feature_df['Exterior1st_Plywood']=((all_feature_df['Exterior1st_Plywood']+all_feature_df['Exterior2nd_Plywood'])%2)+1
#all_feature_df['Exterior1st_Stucco']=((all_feature_df['Exterior1st_Stucco']+all_feature_df['Exterior2nd_Stucco'])%2)+1
#all_feature_df['Exterior1st_Wd Sdng']=((all_feature_df['Exterior1st_Wd Sdng']+all_feature_df['Exterior2nd_Wd Sdng'])%2)+1


all_feature_df.drop(columns={'Utilities','MiscVal','MiscFeature','YrSold','MoSold','Id','LotFrontage','Condition2_Artery','Condition2_Feedr',
                             'Condition2_Norm','Condition2_PosA','Condition2_PosN','Condition2_RRAe','Condition2_RRAn',
                            'Condition2_RRNn','HouseStyle','YearBuilt','Exterior1st','Exterior2nd',
                             'MasVnrType','MasVnrArea','BsmtUnfSF','Heating','Electrical','GarageYrBlt','PavedDrive',
                            'PoolArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Alley','Street',
                            'MSZoning'},inplace=True)
# 'Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_HdBoard','Exterior2nd_ImStucc',
#'Exterior2nd_MetalSd','Exterior2nd_Plywood',
#'Exterior2nd_Stone','Exterior2nd_Stucco','Exterior2nd_VinylSd','Exterior2nd_Wd Sdng','Exterior1st_AsbShng','Exterior1st_AsphShn',

#Other features which we may conside later
#all_feature_df['LotShape']=all_feature_df['LotShape'].map(encode)
#all_feature_df['LandContour']=all_feature_df['LandContour'].map(encode)
#all_feature_df['LandSlope']=all_feature_df['LandSlope'].map(encode)
#all_feature_df['BsmtFinType2']=all_feature_df['BsmtFinType2'].map(encode)
#all_feature_df['Fence']=all_feature_df['Fence'].map(encode)
#all_feature_df['FireplaceQu']=all_feature_df['FireplaceQu'].map(encode)
#all_feature_df['PoolQC']=all_feature_df['PoolQC'].map(encode)
#all_feature_df['GarageCond']=all_feature_df['GarageCond'].map(encode)
#all_feature_df['GarageQual']=all_feature_df['GarageQual'].map(encode)


#all_feature_df=pd.get_dummies(all_feature_df,columns=['MSSubClass'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['LotConfig'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['BldgType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['RoofStyle'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['RoofMatl'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Foundation'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['SaleType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['GarageType'])


all_feature_df.drop(columns={'MSSubClass','LotShape','LandContour','LotConfig','LandSlope','BldgType',
                             'RoofStyle','RoofMatl','Foundation','ExterCond','BsmtFinType2','BsmtFinSF2','1stFlrSF',
                             '2ndFlrSF','LowQualFinSF','BsmtHalfBath','HalfBath','KitchenAbvGr','Fireplaces',
                            'FireplaceQu','GarageType','GarageArea','GarageFinish','GarageQual','GarageCond','SaleType',
                             'Fence','BsmtFullBath','ExterQual'},inplace=True)#'ExterQual',#'GarageFinish',

all_feature_df.fillna(0,inplace=True)
X=all_feature_df.copy()
Y=all_feature_df['SalePrice']
X.drop(columns={'SalePrice'},inplace=True)
feature_labels=X.columns[0:]
forest=RandomForestRegressor(n_estimators=100,criterion='mse',random_state=1,max_features='sqrt',bootstrap=False) #bootstrap=False,max_features='sqrt' using 
forest.fit(X, Y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feature_labels[indices[f]],importances[indices[f]]))
#Finding best features using sequential feature selection
#commenting out as this approach was not helpful

#from  itertools import combinations

# def cal_score(X_train, X_cv, y_train, y_cv,indices):
#     l=list(indices)
#     forest1=RandomForestRegressor(n_estimators=100,criterion='mse',max_features='sqrt',random_state=1,bootstrap=False) 
#     X_t=X_train.iloc[:,l]
#     X_c=X_cv.iloc[:,l]
#     forest1.fit(X_t, y_train)
#     y_cv_pred=forest1.predict(X_c)
#     msd=np.sqrt(metrics.mean_squared_error(y_cv_pred,y_cv))
#     return msd


# X=all_feature_df.copy()
# Y=all_feature_df['SalePrice']
# X.drop(columns={'SalePrice'},inplace=True)
# X_train, X_cv, y_train, y_cv =train_test_split(X, Y,test_size=0.33,random_state=1)

# min_features=5
# dim=X.shape[1]
# indices= list(range(dim))
# subsets_rp=[indices]
# scores=[cal_score(X_train, X_cv, y_train, y_cv,indices)]
# loop=0
# while dim > min_features:
#     print('dim :%d, loop :%d'%(dim,loop))
#     subsets_temp=[]
#     score_temp=[]
#     for p in combinations(indices, r=dim-1):
#         loop+=1
#         score = cal_score(X_train, X_cv,y_train, y_cv, p)
#         score_temp.append(score)
#         subsets_temp.append(p)
#     best = np.argmin(score_temp)
#     scores.append(score_temp[best])
#     indices=subsets_temp[best]
#     subsets_rp.append(indices)
#     3dim -= 1

# final_best=np.argmin(scores)  
# print(final_best)
# print(loop)


#all_feature_df.iloc[:,[0,2,3]]
#print(scores[final_best])
#X_train.iloc[:5,list(subsets_rp[190])].columns
X=all_feature_df.copy()
Y=all_feature_df['SalePrice']
X.drop(columns={'SalePrice'},inplace=True)
#print(X.columns)
X_train, X_cv, y_train, y_cv =train_test_split(X, Y,test_size=0.33,random_state=1)

forest1=RandomForestRegressor(n_estimators=100,criterion='mse',max_features='sqrt',random_state=1,bootstrap=False) #bootstrap=False,max_features='sqrt' using 
forest1.fit(X_train, y_train)
y_train_pred=forest1.predict(X_train)
y_cv_pred=forest1.predict(X_cv)
print('MSE train: %.3f, test: %.3f' % (np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),np.sqrt(metrics.mean_squared_error(y_cv,y_cv_pred))))
all_feature_df=train_df.copy()
encode={'NA' : 0,'Po' : 1,'Fa' : 2,'TA' : 3,'Gd' : 4,'Ex' : 5,'Gtl' : 0,'Mod' : 1,'Sev' : 2,'Y': 1,'N': 0,'P':0.5,'Fin' : 3,
        'RFn' : 2,'UnF' : 1,'LwQ' : 2,'Rec' : 3,'BLQ' : 4,'ALQ' : 5,'GLQ' : 6,'MnWw' : 1,'GdWo' : 2,'MnPrv' : 3, 'GdPrv' : 4,'Av':3,'Mn':2,'No':1,
       'IR3':1,'IR2':2,'Reg':3,'Low':1,'Lvl':2,'Bnk':3,'HLS':4}

all_feature_df['SalePrice']=all_feature_df['SalePrice']-all_feature_df['MiscVal']
all_feature_df['YYMOSold']=all_feature_df['YrSold'].astype('str')+all_feature_df['MoSold'].astype('str')
all_feature_df['Age']=(all_feature_df['YrSold']-all_feature_df['YearBuilt'])+(all_feature_df['MoSold']/12)

all_feature_df['OtherArea']=all_feature_df['WoodDeckSF']+all_feature_df['OpenPorchSF']+all_feature_df['EnclosedPorch']+all_feature_df['3SsnPorch']\
+all_feature_df['ScreenPorch']+all_feature_df['PoolArea']+all_feature_df['GarageArea']

#all_feature_df['LotAreaRemain']=all_feature_df['LotArea']-all_feature_df['1stFlrSF']-all_feature_df['OtherArea']

all_feature_df['ExterQual']=all_feature_df['ExterQual'].map(encode)
all_feature_df['ExterCond']=all_feature_df['ExterCond'].map(encode)
all_feature_df['HeatingQC']=all_feature_df['HeatingQC'].map(encode)
all_feature_df['BsmtQual']=all_feature_df['BsmtQual'].map(encode)
all_feature_df['CentralAir']=all_feature_df['CentralAir'].map(encode)
all_feature_df['GarageFinish']=all_feature_df['GarageFinish'].map(encode)
all_feature_df['BsmtFinType1']=all_feature_df['BsmtFinType1'].map(encode)
all_feature_df['BsmtCond']=all_feature_df['BsmtCond'].map(encode)
all_feature_df['BsmtExposure']=all_feature_df['BsmtExposure'].map(encode)
#all_feature_df['PavedDrive']=all_feature_df['PavedDrive'].map(encode)
#all_feature_df['PoolQC']=all_feature_df['PoolQC'].map(encode)
all_feature_df['KitchenQual']=all_feature_df['KitchenQual'].map(encode)


all_feature_df=pd.get_dummies(all_feature_df,columns=['Neighborhood'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['MSZoning'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['SaleCondition'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Condition1'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Condition2'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Functional'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Exterior1st'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Exterior2nd'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Street'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Alley'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['MasVnrType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Heating'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Electrical'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['HouseStyle'])


all_feature_df['Condition1_Artery']=((all_feature_df['Condition1_Artery']+all_feature_df['Condition2_Artery'])%2)+1
all_feature_df['Condition1_Feedr']=((all_feature_df['Condition1_Feedr']+all_feature_df['Condition2_Feedr'])%2)+1
all_feature_df['Condition1_Norm']=((all_feature_df['Condition1_Norm']+all_feature_df['Condition2_Norm'])%2)+1
all_feature_df['Condition1_PosA']=((all_feature_df['Condition1_PosA']+all_feature_df['Condition2_PosA'])%2)+1
all_feature_df['Condition1_PosN']=((all_feature_df['Condition1_PosN']+all_feature_df['Condition2_PosN'])%2)+1
all_feature_df['Condition1_RRAe']=((all_feature_df['Condition1_RRAe']+all_feature_df['Condition2_RRAe'])%2)+1
all_feature_df['Condition1_RRAn']=((all_feature_df['Condition1_RRAn']+all_feature_df['Condition2_RRAn'])%2)+1
all_feature_df['Condition1_RRNn']=((all_feature_df['Condition1_RRNn']+all_feature_df['Condition2_RRNn'])%2)+1

all_feature_df['Exterior1st_AsbShng']=((all_feature_df['Exterior1st_AsbShng']+all_feature_df['Exterior2nd_AsbShng'])%2)+1
all_feature_df['Exterior1st_AsphShn']=((all_feature_df['Exterior1st_AsphShn']+all_feature_df['Exterior2nd_AsphShn'])%2)+1
all_feature_df['Exterior1st_CBlock']=((all_feature_df['Exterior1st_CBlock']+all_feature_df['Exterior2nd_CBlock'])%2)+1
all_feature_df['Exterior1st_HdBoard']=((all_feature_df['Exterior1st_HdBoard']+all_feature_df['Exterior2nd_HdBoard'])%2)+1
all_feature_df['Exterior1st_MetalSd']=((all_feature_df['Exterior1st_MetalSd']+all_feature_df['Exterior2nd_MetalSd'])%2)+1
all_feature_df['Exterior1st_Plywood']=((all_feature_df['Exterior1st_Plywood']+all_feature_df['Exterior2nd_Plywood'])%2)+1
all_feature_df['Exterior1st_Stucco']=((all_feature_df['Exterior1st_Stucco']+all_feature_df['Exterior2nd_Stucco'])%2)+1
all_feature_df['Exterior1st_Wd Sdng']=((all_feature_df['Exterior1st_Wd Sdng']+all_feature_df['Exterior2nd_Wd Sdng'])%2)+1


all_feature_df.drop(columns={'Utilities','MiscVal','MiscFeature','YrSold','MoSold','Id','LotFrontage','Condition2_Artery','Condition2_Feedr',
                             'Condition2_Norm','Condition2_PosA','Condition2_PosN','Condition2_RRAe','Condition2_RRAn',
                            'Condition2_RRNn','HouseStyle','YearBuilt',
                             'MasVnrType_None','MasVnrType_BrkCmn','MasVnrType_BrkFace','BsmtUnfSF','Heating','Electrical','GarageYrBlt','PavedDrive',
                            'PoolArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Alley','Street',
                            'MSZoning',
                             'Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_HdBoard','Exterior2nd_ImStucc',
                             'Exterior2nd_MetalSd','Exterior2nd_Plywood','Exterior2nd_Stone','Exterior2nd_Stucco',
                             'Exterior2nd_VinylSd','Exterior2nd_Wd Sdng','Exterior1st_AsbShng','Exterior1st_AsphShn',},inplace=True)#'Exterior1st','Exterior2nd',
# 'Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_HdBoard','Exterior2nd_ImStucc',
#'Exterior2nd_MetalSd','Exterior2nd_Plywood',
#'Exterior2nd_Stone','Exterior2nd_Stucco','Exterior2nd_VinylSd','Exterior2nd_Wd Sdng','Exterior1st_AsbShng','Exterior1st_AsphShn',

#Other features which we may conside later
#all_feature_df['LotShape']=all_feature_df['LotShape'].map(encode)
#all_feature_df['LandContour']=all_feature_df['LandContour'].map(encode)
#all_feature_df['LandSlope']=all_feature_df['LandSlope'].map(encode)
#all_feature_df['BsmtFinType2']=all_feature_df['BsmtFinType2'].map(encode)
#all_feature_df['Fence']=all_feature_df['Fence'].map(encode)
all_feature_df['FireplaceQu']=all_feature_df['FireplaceQu'].map(encode)
#all_feature_df['PoolQC']=all_feature_df['PoolQC'].map(encode)
#all_feature_df['GarageCond']=all_feature_df['GarageCond'].map(encode)
#all_feature_df['GarageQual']=all_feature_df['GarageQual'].map(encode)


#all_feature_df=pd.get_dummies(all_feature_df,columns=['MSSubClass'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['LotConfig'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['BldgType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['RoofStyle'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['RoofMatl'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Foundation'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['SaleType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['GarageType'])


all_feature_df.drop(columns={'MSSubClass','LotShape','LandContour','LotConfig','LandSlope','BldgType',
                             'RoofStyle','RoofMatl','Foundation','ExterCond','BsmtFinType2','BsmtFinSF2','1stFlrSF',
                             '2ndFlrSF','LowQualFinSF','BsmtHalfBath','HalfBath','KitchenAbvGr','Fireplaces','FireplaceQu',
                             'GarageType','GarageQual','GarageCond','SaleType','Fence','BsmtFullBath','ExterQual',
                             'YYMOSold','GarageFinish','GarageArea'},inplace=True)#'ExterQual',#'GarageFinish',

all_feature_df.fillna(0,inplace=True)
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
X=all_feature_df.copy()
Y=X[['SalePrice']]
X.drop(columns={'SalePrice'},inplace=True)
scaler=MinMaxScaler(feature_range=(0,10))
cols=['LotArea',  'BsmtFinSF1', 'TotalBsmtSF','GrLivArea', 'OtherArea','YearRemodAdd','Age','MasVnrArea']
#cols=X.columns
for i in cols:    
    X[i]=scaler.fit_transform(X[[i]])
#print(X.columns)
X_train, X_cv, y_train, y_cv =train_test_split(X, Y,test_size=0.33,random_state=1)


for i in range (1,10):
    knn = neighbors.KNeighborsRegressor(n_neighbors=i, weights='uniform',algorithm='auto') 
    knn.fit(X_train, y_train)
    y_train_pred=knn.predict(X_train)
    y_cv_pred=knn.predict(X_cv)
    print('i loop: %d ; MSE train: %.3f, test: %.3f' % (i,np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
                                                        np.sqrt(metrics.mean_squared_error(y_cv,y_cv_pred))))


all_feature_df=train_df.copy()
encode={'NA' : 0,'Po' : 1,'Fa' : 2,'TA' : 3,'Gd' : 4,'Ex' : 5,'Gtl' : 0,'Mod' : 1,'Sev' : 2,'Y': 1,'N': 0,'P':0.5,'Fin' : 3,
        'RFn' : 2,'UnF' : 1,'LwQ' : 2,'Rec' : 3,'BLQ' : 4,'ALQ' : 5,'GLQ' : 6,'MnWw' : 1,'GdWo' : 2,'MnPrv' : 3, 'GdPrv' : 4,'Av':3,'Mn':2,'No':1,
       'IR3':1,'IR2':2,'Reg':3,'Low':1,'Lvl':2,'Bnk':3,'HLS':4}

all_feature_df['SalePrice']=all_feature_df['SalePrice']-all_feature_df['MiscVal']
all_feature_df['YYMOSold']=all_feature_df['YrSold'].astype('str')+all_feature_df['MoSold'].astype('str')
all_feature_df['Age']=(all_feature_df['YrSold']-all_feature_df['YearBuilt'])+(all_feature_df['MoSold']/12)

all_feature_df['OtherArea']=all_feature_df['WoodDeckSF']+all_feature_df['OpenPorchSF']+all_feature_df['EnclosedPorch']+all_feature_df['3SsnPorch']\
+all_feature_df['ScreenPorch']+all_feature_df['PoolArea']+all_feature_df['GarageArea']

#all_feature_df['LotAreaRemain']=all_feature_df['LotArea']-all_feature_df['1stFlrSF']-all_feature_df['OtherArea']

all_feature_df['ExterQual']=all_feature_df['ExterQual'].map(encode)
all_feature_df['ExterCond']=all_feature_df['ExterCond'].map(encode)
all_feature_df['HeatingQC']=all_feature_df['HeatingQC'].map(encode)
all_feature_df['BsmtQual']=all_feature_df['BsmtQual'].map(encode)
all_feature_df['CentralAir']=all_feature_df['CentralAir'].map(encode)
all_feature_df['GarageFinish']=all_feature_df['GarageFinish'].map(encode)
all_feature_df['BsmtFinType1']=all_feature_df['BsmtFinType1'].map(encode)
all_feature_df['BsmtCond']=all_feature_df['BsmtCond'].map(encode)
all_feature_df['BsmtExposure']=all_feature_df['BsmtExposure'].map(encode)
#all_feature_df['PavedDrive']=all_feature_df['PavedDrive'].map(encode)
#all_feature_df['PoolQC']=all_feature_df['PoolQC'].map(encode)
all_feature_df['KitchenQual']=all_feature_df['KitchenQual'].map(encode)


all_feature_df=pd.get_dummies(all_feature_df,columns=['Neighborhood'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['MSZoning'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['SaleCondition'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Condition1'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Condition2'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['Functional'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Exterior1st'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Exterior2nd'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Street'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Alley'])
all_feature_df=pd.get_dummies(all_feature_df,columns=['MasVnrType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Heating'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Electrical'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['HouseStyle'])


all_feature_df['Condition1_Artery']=((all_feature_df['Condition1_Artery']+all_feature_df['Condition2_Artery'])%2)+1
all_feature_df['Condition1_Feedr']=((all_feature_df['Condition1_Feedr']+all_feature_df['Condition2_Feedr'])%2)+1
all_feature_df['Condition1_Norm']=((all_feature_df['Condition1_Norm']+all_feature_df['Condition2_Norm'])%2)+1
all_feature_df['Condition1_PosA']=((all_feature_df['Condition1_PosA']+all_feature_df['Condition2_PosA'])%2)+1
all_feature_df['Condition1_PosN']=((all_feature_df['Condition1_PosN']+all_feature_df['Condition2_PosN'])%2)+1
all_feature_df['Condition1_RRAe']=((all_feature_df['Condition1_RRAe']+all_feature_df['Condition2_RRAe'])%2)+1
all_feature_df['Condition1_RRAn']=((all_feature_df['Condition1_RRAn']+all_feature_df['Condition2_RRAn'])%2)+1
all_feature_df['Condition1_RRNn']=((all_feature_df['Condition1_RRNn']+all_feature_df['Condition2_RRNn'])%2)+1

#all_feature_df['Exterior1st_AsbShng']=((all_feature_df['Exterior1st_AsbShng']+all_feature_df['Exterior2nd_AsbShng'])%2)+1
#all_feature_df['Exterior1st_AsphShn']=((all_feature_df['Exterior1st_AsphShn']+all_feature_df['Exterior2nd_AsphShn'])%2)+1
#all_feature_df['Exterior1st_CBlock']=((all_feature_df['Exterior1st_CBlock']+all_feature_df['Exterior2nd_CBlock'])%2)+1
#all_feature_df['Exterior1st_HdBoard']=((all_feature_df['Exterior1st_HdBoard']+all_feature_df['Exterior2nd_HdBoard'])%2)+1
#all_feature_df['Exterior1st_MetalSd']=((all_feature_df['Exterior1st_MetalSd']+all_feature_df['Exterior2nd_MetalSd'])%2)+1
#all_feature_df['Exterior1st_Plywood']=((all_feature_df['Exterior1st_Plywood']+all_feature_df['Exterior2nd_Plywood'])%2)+1
#all_feature_df['Exterior1st_Stucco']=((all_feature_df['Exterior1st_Stucco']+all_feature_df['Exterior2nd_Stucco'])%2)+1
#all_feature_df['Exterior1st_Wd Sdng']=((all_feature_df['Exterior1st_Wd Sdng']+all_feature_df['Exterior2nd_Wd Sdng'])%2)+1


all_feature_df.drop(columns={'Utilities','MiscVal','MiscFeature','YrSold','MoSold','Id','LotFrontage','Condition2_Artery','Condition2_Feedr',
                             'Condition2_Norm','Condition2_PosA','Condition2_PosN','Condition2_RRAe','Condition2_RRAn',
                            'Condition2_RRNn','HouseStyle','YearBuilt',
                             'MasVnrType_None','MasVnrType_BrkCmn','MasVnrType_BrkFace','BsmtUnfSF','Heating','Electrical','GarageYrBlt','PavedDrive',
                            'PoolArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Alley','Street',
                            'MSZoning',
                             'Exterior2nd','Exterior1st'
                            },inplace=True)#'Exterior1st','Exterior2nd',
# 'Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_HdBoard','Exterior2nd_ImStucc',
#'Exterior2nd_MetalSd','Exterior2nd_Plywood',
#'Exterior2nd_Stone','Exterior2nd_Stucco','Exterior2nd_VinylSd','Exterior2nd_Wd Sdng','Exterior1st_AsbShng','Exterior1st_AsphShn',

#Other features which we may conside later
#all_feature_df['LotShape']=all_feature_df['LotShape'].map(encode)
#all_feature_df['LandContour']=all_feature_df['LandContour'].map(encode)
#all_feature_df['LandSlope']=all_feature_df['LandSlope'].map(encode)
#all_feature_df['BsmtFinType2']=all_feature_df['BsmtFinType2'].map(encode)
#all_feature_df['Fence']=all_feature_df['Fence'].map(encode)
all_feature_df['FireplaceQu']=all_feature_df['FireplaceQu'].map(encode)
#all_feature_df['PoolQC']=all_feature_df['PoolQC'].map(encode)
#all_feature_df['GarageCond']=all_feature_df['GarageCond'].map(encode)
#all_feature_df['GarageQual']=all_feature_df['GarageQual'].map(encode)


#all_feature_df=pd.get_dummies(all_feature_df,columns=['MSSubClass'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['LotConfig'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['BldgType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['RoofStyle'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['RoofMatl'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['Foundation'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['SaleType'])
#all_feature_df=pd.get_dummies(all_feature_df,columns=['GarageType'])


all_feature_df.drop(columns={'MSSubClass','LotShape','LandContour','LotConfig','LandSlope','BldgType',
                             'RoofStyle','RoofMatl','Foundation','ExterCond','BsmtFinType2','BsmtFinSF2','1stFlrSF',
                             '2ndFlrSF','LowQualFinSF','BsmtHalfBath','HalfBath','KitchenAbvGr','Fireplaces','FireplaceQu',
                             'GarageType','GarageQual','GarageCond','SaleType','Fence','BsmtFullBath','ExterQual',
                             'YYMOSold','GarageFinish','GarageArea'},inplace=True)#'ExterQual',#'GarageFinish',

all_feature_df.fillna(0,inplace=True)
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
X=all_feature_df.copy()
Y=X[['SalePrice']]
X.drop(columns={'SalePrice'},inplace=True)
scaler=MinMaxScaler(feature_range=(0,10))
cols=['LotArea',  'BsmtFinSF1', 'TotalBsmtSF','GrLivArea', 'OtherArea','Age']
#cols=X.columns
for i in cols:    
    X[i]=scaler.fit_transform(X[[i]])
#print(X.columns)
X_train, X_cv, y_train, y_cv =train_test_split(X, Y,test_size=0.33,random_state=1)


lm = linear_model.Lasso(alpha =0.5,max_iter=10000) 
lm.fit(X_train, y_train)
y_train_pred=lm.predict(X_train)
y_cv_pred=lm.predict(X_cv)
print(' MSE train: %.3f, test: %.3f' % (np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
                                                    np.sqrt(metrics.mean_squared_error(y_cv,y_cv_pred))))

coefficients = pd.DataFrame({"Feature":X_train.columns,"Coefficients":np.transpose(lm.coef_)})
coefficients.loc[coefficients["Coefficients"]==0]
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
X=all_feature_df.copy()
Y=X[['SalePrice']]
X.drop(columns={'SalePrice'},inplace=True)
scaler=MinMaxScaler(feature_range=(0,10))
cols=['LotArea',  'BsmtFinSF1', 'TotalBsmtSF','GrLivArea', 'OtherArea','Age']
#cols=X.columns
for i in cols:    
    X[i]=scaler.fit_transform(X[[i]])
#print(X.columns)
X_train, X_cv, y_train, y_cv =train_test_split(X, Y,test_size=0.33,random_state=1)


lm = linear_model.Ridge(alpha =6,max_iter=10000) 
lm.fit(X_train, y_train)
y_train_pred=lm.predict(X_train)
y_cv_pred=lm.predict(X_cv)
print(' MSE train: %.3f, test: %.3f' % (np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
                                                    np.sqrt(metrics.mean_squared_error(y_cv,y_cv_pred))))
from mlxtend.regressor import StackingRegressor
from sklearn.svm import SVR

X=all_feature_df.copy()
Y=X['SalePrice']
X.drop(columns={'SalePrice'},inplace=True)
scaler=MinMaxScaler(feature_range=(0,10))
cols=['LotArea',  'BsmtFinSF1', 'TotalBsmtSF','GrLivArea', 'OtherArea','Age']
#cols=X.columns
for i in cols:    
    X[i]=scaler.fit_transform(X[[i]])
#print(X.columns)
X_train, X_cv, y_train, y_cv =train_test_split(X, Y,test_size=0.33,random_state=1)

forest1=RandomForestRegressor(n_estimators=100,criterion='mse',max_features='sqrt',random_state=1,bootstrap=False) 
lr = linear_model.Ridge(alpha =6,max_iter=10000) 
lm = linear_model.Lasso(alpha =0.5,max_iter=10000) 
knn = neighbors.KNeighborsRegressor(n_neighbors=4, weights='uniform',algorithm='auto')
l = SVR(kernel='linear')

#stregr = StackingRegressor(regressors=[lr, lm, knn,forest1], meta_regressor=l)
#stregr.fit(X_train, y_train)
#y_train_pred=stregr.predict(X_train)
#y_cv_pred=stregr.predict(X_cv)
lr.fit(X_train, y_train)
lm.fit(X_train, y_train)
knn.fit(X_train, y_train)
forest1.fit(X_train, y_train)

y_cv_pred_lr=lr.predict(X_cv)
y_cv_pred_lm=lm.predict(X_cv)
y_cv_pred_knn=knn.predict(X_cv)
y_cv_pred_forest1=forest1.predict(X_cv)



Data=pd.DataFrame(y_cv_pred_lr)
Datalm=pd.DataFrame(y_cv_pred_lm)
Dataknn=pd.DataFrame(y_cv_pred_knn)
Dataforest1=pd.DataFrame(y_cv_pred_forest1)

Data.rename(columns={0:'lr'},inplace=True)
Data['lm']=Datalm[0]
Data['knn']=Dataknn[0]
Data['forest1']=Dataforest1[0]
final_pred=[]

for i in Data.index:
    mean=Data.iloc[i].mean()
    std=Data.iloc[i].std()
    final_v=mean+(std/3)
    final_pred.append(final_v)

y_c=np.array(y_cv.values)
print(' MSE  test: %.3f' % (np.sqrt(metrics.mean_squared_error(y_c,final_pred))))

n=y_c.shape[0]
for i in range(n):
    print("i:%d  ,%f"%(i,y_c[i]-final_pred[i]))
print(' MSE train: %.3f, test: %.3f' % (np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
                                                    np.sqrt(metrics.mean_squared_error(y_cv,y_cv_pred))))