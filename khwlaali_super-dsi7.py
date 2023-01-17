import numpy as np

import seaborn as sns

import pandas as pd

import math

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4)

%config InlineBackend.figure_format = 'retina'

%matplotlib inline

plt.style.use('fivethirtyeight')

sns.set(font_scale=1.5)
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train_data.shape
test_data.shape
train_data.head()
test_data.head()
fig,ax=plt.subplots()

ax.scatter(x=train_data['GrLivArea'],y=train_data['SalePrice'])

plt.xlabel('GrLivArea',fontsize=13)

plt.ylabel('SalePrice',fontsize=13)

plt.show()
#save SalePrice in sales

y_train = train_data['SalePrice']
# Dropping the saleprice from the train data

train_data.drop('SalePrice',axis=1, inplace= True)
# concat the train data with test data to do cleaing

concat_df = pd.concat([train_data,test_data], axis=0, sort=True)
#check the missing value

concat_df.isnull().sum()
# check the dataframe information

concat_df.info()
total =concat_df.isnull().sum()

missing_data = pd.concat([total], axis=1, keys=['Total'])

missing_data.head(40)
missing_data.tail(40)
# Replase the nan value 

def replace_nan(data,coulmn,value):

    for i in coulmn:

        for x in value:

            data[i] =data[i].fillna(x)
column=['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageCond'

       ,'GarageType','GarageFinish','GarageQual','MiscFeature','PoolQC','Fence','MiscFeature','Exterior1st','Exterior2nd','Functional','KitchenQual','SaleType'

       ,'Utilities','MSZoning']

value=['No Alley','None','No_Bsmt','No_Bsmt','No_Bsmt','No_Bsmt','No_Bsmt','SBrkr','No Fireplace','No Garage',

       'No Garage','No Garage','No Garage','None','None','no Fence','None','VinylSd','VinylSd','Typ','TA','WD','AllPub','RL']

replace_nan(concat_df,column,value)
int_columns=['LotFrontage','MasVnrArea','GarageYrBlt','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GarageArea','TotalBsmtSF','GarageCars'

]

the_value=[70.0,103.685262,439.20,52.61, 0.43,0.0,554.29,472.76,1046.11,1.7,concat_df.median()]

replace_nan(concat_df,int_columns,the_value)
total =concat_df.isnull().sum()

missing_data = pd.concat([total], axis=1, keys=['Total'])

missing_data.head(40)
missing_data.tail(40)
concat_df.info()
concat_df['LotFrontage']= concat_df['LotFrontage'].astype(int)
concat_df['MasVnrArea']= concat_df['MasVnrArea'].astype(int)
concat_df['GarageYrBlt']= concat_df['GarageYrBlt'].astype(int)
concat_df['BsmtFinSF1']= concat_df['BsmtFinSF1'].astype(int)
concat_df['BsmtFinSF2']= concat_df['BsmtFinSF2'].astype(int)
concat_df['BsmtFullBath']= concat_df['BsmtFullBath'].astype(int)
concat_df['BsmtHalfBath']= concat_df['BsmtHalfBath'].astype(int)
concat_df['BsmtUnfSF']= concat_df['BsmtUnfSF'].astype(int)
concat_df['GarageArea']= concat_df['GarageArea'].astype(int)

concat_df['TotalBsmtSF']= concat_df['TotalBsmtSF'].astype(int)
concat_df['GarageCars']= concat_df['GarageCars'].astype(int)
concat_df.info()
concat_df.tail()
concat_df.shape
# Creating dummies for object columns only

concat_df = pd.get_dummies(concat_df,drop_first=True)

concat_df.head()
concat_df.shape
concat_df.corr()
# from sklearn.preprocessing import StandardScaler

# Scaler= StandardScaler()

# model_Scaler = Scaler.fit_transform(concat_df)
# Separating dataframe into train set



df_train = concat_df.iloc[0:1460 , : ]



df_train.shape
# Separating dataframe into test set    



df_test =concat_df.iloc[1460: , : ]

df_test.shape
# # Adding SalePrice column to train dataframe

# concat_train = pd.concat([df_train, y_train], axis=1)

# # concat_train.to_csv('concat_train_cleaned.csv')

# concat_train.head()
# fig,ax=plt.subplots()

# ax.scatter(x=concat_train['GrLivArea'],y=concat_train['SalePrice'])

# plt.xlabel('GrLivArea',fontsize=13)

# plt.ylabel('SalePrice',fontsize=13)

# plt.show()
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LassoCV

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler

# # for Clarrifications:

# df_train is training dataframe without SalePrice column

#  dataframe including SalePrice column



from sklearn.model_selection import train_test_split, cross_val_score





# # X = df_train

# y =sales



# # get train-test split

# X_train, X_test, y_train, y_test = train_test_split(

#     X, y, test_size=0.2, random_state=10) # change random_state and size!





# scaler = StandardScaler()

# X_train = pd.DataFrame(scaler.fit_transform(df_train), columns=X.columns)

# X_test = pd.DataFrame(scaler.transform(df_test), columns=X.columns)
# print("Train:",X_train.shape, X_test.shape )

# print("Test:",y_train.shape, y_test.shape )

from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

X_train_ss = ss.fit_transform(df_train)

X_test_ss = ss.transform(df_test)













# ss = StandardScaler()

# X_train = ss.fit_transform(X_train)

# X_test = ss.transform(X_test)

# df_test = ss.transform(df_test)



# X = pd.DataFrame(data.data, columns=data.feature_names)

# y = data.target
lr = LinearRegression()

lr.fit(X_train_ss,y_train)
from sklearn.model_selection import KFold

from sklearn import datasets

kf = KFold(n_splits=5, shuffle=True, random_state=100) # notice shuffle

scores = cross_val_score(lr, X_train_ss,y_train, cv=kf)



print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())
print("R-score is:",lr.score(X_train_ss, y_train))
df_prdict= lr.predict(X_test_ss)

df_prdict

# lr.coef_
lasso_ = Lasso(alpha=.2)

lasso_.fit(X_train_ss, y_train)

print('Testing Score:',lasso_.score(X_train_ss, y_train))

lasso__pred=lasso_.predict(X_test_ss)

print(lasso__pred)

# lasso.coef_

#X_train.head(2)

# y_train.head(2)
lasso_cv = LassoCV()

lasso_cv.fit(X_train_ss, y_train)

print('The Beast alpha :' , lasso_cv.alpha_)

print('The Beast score :' , lasso_cv.score(X_train_ss, y_train))



lassocv__pred=lasso_cv.predict(X_test_ss)

print('thr test predict:',lassocv__pred)
lasso_cv.coef_

coef = pd.Series(lasso_cv.coef_, index = df_train.columns)

coef

selec = coef[coef != 0]

selec
selected_feat = selec.index

selected_feat
print(len(selected_feat))
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()

imp_coef
print(selected_feat)
imp_coef = selec

import matplotlib

matplotlib.rcParams['figure.figsize'] = (30.0, 30.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth = 10)

dt.fit(X_train_ss, y_train)

print('Traing Score:',dt.score(X_train_ss, y_train))

dt_prdict= dt.predict(X_test_ss)



print('Testing predict:',dt_prdict)
from sklearn.linear_model import  RidgeCV

print("Cross-validated training scores:", scores)

print("Mean cross-validated training score:", scores.mean())



r_cv = RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30])





r_cv.fit(X_train_ss, y_train)





kf = KFold(n_splits=5, shuffle=True, random_state=100) # notice shuffle

scores = cross_val_score(r_cv, X_train_ss, y_train, cv=kf)



print('Traing Score:',r_cv.score(X_train_ss, y_train))

r_cv_prdict= r_cv.predict(X_test_ss)



print('Testing predict:',r_cv_prdict)


r_cv2= RidgeCV(alphas=[0.05, 0.3, 1])

r_cv2.fit(X_train_ss, y_train)

print('Traing Score:',r_cv.score(X_train_ss, y_train))

r_cv2_prdict= r_cv2.predict(X_test_ss)



print('Testing predict:',r_cv2_prdict)


from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor()

rf.fit(X_train_ss, y_train)

rf_prdict= rf.predict(X_test_ss)

print('train score : ',rf.score(X_train_ss, y_train))

print('Testing predict:',rf_prdict)



from sklearn.neighbors import KNeighborsRegressor

kn = KNeighborsRegressor()

kn.fit(X_train_ss, y_train)

kn_prdict=kn.predict(X_test_ss)

print('train score : ',kn.score(X_train_ss, y_train))

print('Testing predict:',kn_prdict)

y_test_sub=lr.predict(X_test_ss)

Sub1 = [x for x in range (1461,2920)]

Submission1 = {'Id':Sub1,

               'SalePrice':y_test_sub}

df_submission = pd.DataFrame(Submission1)

df_submission.to_csv('LR_submission.csv',index=False)

df_submission.head()