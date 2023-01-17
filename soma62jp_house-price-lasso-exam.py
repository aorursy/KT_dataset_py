# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy import stats



import warnings

warnings.filterwarnings('ignore')



%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
#EDA

# 求めたいSalePriceの基本統計量を調べる

(mu, sigma) = stats.norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

print( '\n skew = {:.2f}\n'.format(stats.skew(train['SalePrice'])))



# データの素性を調べる

train.info()
def eda_QQ(df):

    # find the unique values from categorical features

    for col in df.select_dtypes(include=['int64','float64']).columns:



        print(col)



        #(mu, sigma) = stats.norm.fit(df[col])

        #print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

        print( '\n skew = {:.2f}\n'.format(stats.skew(df[col])))

        print( '\n NaN data sum = {:d}\n'.format(df[col].isnull().sum()))

        print( '\n NaN data rate = {:.2f}\n'.format(df[col].isnull().sum() / len(col)))



        plt.figure(figsize=(20,15), facecolor='white')

        ax = plt.subplot(3,3,5)

        

        try:

            sns.distplot(df[col])

            

            #Now plot the distribution

            sns.distplot(df[col] , fit=stats.norm);

            #plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

            plt.ylabel('Frequency')

            plt.title('distribution')            

            

        except:

            print('error',col)



        #Get also the QQ-plot

        fig = plt.figure()

        res = stats.probplot(df[col], plot=plt)

        plt.show()

        

def eda_Object(df):

    # find the unique values from categorical features

    for col in df.select_dtypes(include='object').columns:

        print(col)

        print(df[col].unique())

      

        #print(df.groupby([col,resVar]).size())

        #print(pd.crosstab(df[resVar],df[col]).apply(lambda r: r/r.sum() ,axis=1))

        

        plt.figure(figsize=(30,30), facecolor='white')

        ax = plt.subplot(3,3,5)

        sns.countplot(y=df[col],data=df)

        plt.xlabel(col)

        plt.title(col)

        plt.show()
#Now plot the distribution

eda_QQ(train)
#Now plot the distribution

eda_Object(train)
#順序データはLog補正の対象から外す

drop_table=[]

drop_table.append('OverallQual')

drop_table.append('OverallCond')

drop_table.append('BsmtFullBath')

drop_table.append('FullBath')

drop_table.append('HalfBath')

drop_table.append('BedroomAbvGr')

drop_table.append('KitchenAbvGr')

drop_table.append('TotRmsAbvGrd')

drop_table.append('GarageCars')

drop_table.append('MoSold')

drop_table.append('YrSold')
# 欠損値を調べる

np.set_printoptions(threshold=np.inf)        # 全件表示設定

pd.set_option('display.max_rows',10000)      # 1000件表示設定



train.isnull().sum()
# QQプロットが曲がっているので正規分布から外れている

# Log変換して正規分布に偏りを合わせる

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=stats.norm);



# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#数値データ一括log変換

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: stats.skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats.drop(drop_table)

skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# 基本統計量

train.describe()
# カテゴリカルデータ一括ダミー変数に変換(One-hot)

all_data = pd.get_dummies(all_data)

# 欠損値は平均値に変換

all_data = all_data.fillna(all_data.mean())



all_data.head()
# trainデータとtestデータに分ける

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train['SalePrice']



eda_QQ(X_train)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet,Lasso,LassoCV, LassoLarsCV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split,KFold,cross_validate

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc



# 学習データ = 75% テストデータ = 25%　に分割

train_x, test_x, train_y, test_y = train_test_split(X_train, y, test_size=0.25, 

                                                    shuffle = True , random_state = 0)



def rmse_cv(model,train,y):

    rmse= np.sqrt(-cross_val_score(model, train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
# Lasso回帰

#model_lasso = Lasso(alpha = 0.1225).fit(X_train, y)

scaler = StandardScaler()

model_lasso = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), cv=5)



#標準化すること！

scaler.fit(train_x)

model_lasso.fit(scaler.transform(train_x), train_y) 
#CVのプロット

cv_x = np.log(model_lasso.alphas_)

cv_y = np.array([i.mean() for i in model_lasso.mse_path_])

err = np.array([np.std(i) / np.sqrt(10) for i in model_lasso.mse_path_])

lambda_from_cv = cv_x[np.argmin(cv_y)]

min_cv = np.min(cv_y)

err_from_cv = err[np.argmin(cv_y)]



ose_lambda = -10000

standard = min_cv + err_from_cv

for counter,i in enumerate(cv_y):

    if i < standard:

        if cv_x[counter] > ose_lambda:

            ose_lambda = cv_x[counter]
#CVの過程のプロット

plt.figure(figsize=(15,10))

plt.errorbar(cv_x,cv_y,yerr=err,fmt="o",capthick=1,capsize=10,color="orange",ecolor = "orange")

plt.axvline(x = lambda_from_cv,linestyle=("--"))

plt.axvline(x = ose_lambda,linestyle=("--"),color="green")

sns.set_context('notebook')

plt.title('rmse avarage',fontsize=20)

plt.xlabel('L1 Log',fontsize=15)

plt.ylabel('rmse avarage',fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

sns.set_context('notebook')

plt.show()
# rmseを求める

rmse_cv(model_lasso,scaler.transform(train_x),train_y).mean()
from sklearn.metrics import mean_squared_error,r2_score



def model_Eval(testX,testY):

    

    #標準化すること！

    scaler = StandardScaler()

    scaler.fit(testX)

    testX_std = scaler.transform(testX)

        

    model_lasso.fit(testX_std, testY)



    # rmseを求める

    rmse_cv(model_lasso,testX_std,testY).mean()



    # 係数を格納し、選択された（0ではない）係数を算出

    coef = pd.Series(model_lasso.coef_, index = testX.columns)

    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")



    # 上から10番目の特徴量と下から10番目の特徴量を集計

    imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])



    imp_coef.plot(kind = "barh")

    plt.title("Coefficients in the Lasso Model")



    # 予測と結果

    np.set_printoptions(threshold=10)        # 10件表示設定

    pd.set_option('display.max_rows',10)      # 10件表示設定

    preds = pd.DataFrame({"preds":np.expm1(model_lasso.predict(testX_std)), "true":np.expm1(testY)})

    preds



    # 残差プロット

    preds["residuals"] = preds["true"] - preds["preds"]

    preds.plot(x = "preds", y = "residuals",kind = "scatter")



    # モデルのあてはめ

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)



    ax.scatter(preds["true"], preds["preds"],label="Lasso Model Fitting")

    ax.set_xlabel('predicted')

    ax.set_ylabel('true')

    ax.set_aspect('equal')



    # mseとr2を求める

    mse = mean_squared_error(preds["true"], preds["preds"])

    print('mse:',mse)

    r2 = r2_score(preds["true"], preds["preds"])

    print('r2:',r2)
# 訓練データの検証

model_Eval(train_x,train_y) 
# テストデータの検証

# 過学習の確認 

model_Eval(test_x,test_y)
# 予測開始

#np.epem1で予測結果をLog逆変換すること！！

scaler = StandardScaler()

scaler.fit(X_test)

y_pred_lasso = np.expm1(model_lasso.predict(scaler.transform(X_test))) 
pred_df = pd.DataFrame(y_pred_lasso, index=test["Id"], columns=["SalePrice"])

pred_df
pred_df.to_csv('output.csv', header=True, index_label='Id')