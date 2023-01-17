# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



class Axis():

    row = 0

    col = 1





train_path = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

test_path = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"



train_house_prices_origin_data = pd.read_csv(train_path)

test_house_prices_origin_data = pd.read_csv(test_path)
#TODO: 教師データを見て，目的変数と説明変数の分散共分散行列を求めて必要なデータを抜き出してあげるのが大切だよね

# つまりデータはまだ分けないほうが賢明かも

train_house_prices_origin_data.head()
corr_matrix = train_house_prices_origin_data.corr()

corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ['SalePrice', 'YearBuilt', 'GrLivArea',

       'GarageYrBlt', 'YearRemodAdd', 'GarageArea', 'TotalBsmtSF','1stFlrSF']

scatter_matrix(train_house_prices_origin_data[attributes[0:8]], figsize=(12,8))

plt.show()
train_house_prices_origin_data[attributes]
# train_house_prices_origin_data = train_house_prices_origin_data[train_house_prices_origin_data["SalePrice"] <= 214000] # 75%

train_explain_data = train_house_prices_origin_data.drop(columns=["SalePrice", "Id"])

train_object_data = train_house_prices_origin_data["SalePrice"]
train_object_data.hist()
train_object_data.describe()
plt.plot(np.arange(0,len(train_object_data)), train_object_data, "b.")

plt.show()
## データの正規化

def normalization_min_max_tabel_data(df):

    """

    入力：df:DataFrame pandasを使用しているため，DataFrameのまま処理したい

    出力：df_0_to_1:DataFrame 各説明変数を閉区間[0,1]の範囲に正規化した結果

    """

    df_0_to_1 = (df - df.min()) / (df.max() - df.min()) 

    return df_0_to_1
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline



num_pipeline = Pipeline([

    ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),

    ('min_max_scaler', MinMaxScaler()),

])



#TODO: データが数値(Numerical)属性のみを抽出

attributes = ['YearBuilt', 'GrLivArea',

       'GarageYrBlt', 'YearRemodAdd', 'GarageArea', 'TotalBsmtSF','1stFlrSF']

train_explain_numerical_data = train_explain_data[attributes]

#TODO: 欠損値は平均で埋める



train_explain_not_null_numerical_data = pd.DataFrame(num_pipeline.fit_transform(train_explain_numerical_data), columns=train_explain_numerical_data.columns)

(train_explain_not_null_numerical_data < 0).sum()
# TODO: Not Nullな数値データを使って分散共分散行列を作る

COVxx = np.cov(m=train_explain_not_null_numerical_data.T, bias=False)

COVxy = np.cov(m=train_explain_not_null_numerical_data.T, y=train_object_data, bias=False)

train_explain_numerical_COVxx = pd.DataFrame(COVxx, train_explain_not_null_numerical_data.columns, train_explain_not_null_numerical_data.columns)

index_train_object = pd.Index([train_object_data.name])

train_explain_numerical_COVxy = pd.DataFrame(COVxy, columns=train_explain_not_null_numerical_data.columns.append(index_train_object), index=train_explain_not_null_numerical_data.columns.append(index_train_object)).SalePrice
train_explain_numerical_COVxx
train_explain_numerical_COVxy, train_explain_numerical_COVxy.shape
#TODO: 分散共分散行列の固有値，固有ベクトルを計算する

import numpy.linalg as LA

train_explain_numerical_eigenvalue, train_explain_numerical_eigenvector = LA.eig(train_explain_numerical_COVxx)
train_explain_numerical_COVxx_eigenvalue = pd.DataFrame(train_explain_numerical_eigenvalue, index=train_explain_numerical_COVxx.columns, columns=["EigenValue"])

print(f"COVxx_eigenvalue shape:{train_explain_numerical_COVxx_eigenvalue.shape}")

train_explain_numerical_COVxx_eigenvalue
# 固有値1以上のデータを取得する．（固有値が小さすぎるとランクが落ちる可能性がある）

train_explain_numerical_COVxx_eigenvalue_over_zero = train_explain_numerical_COVxx_eigenvalue[train_explain_numerical_COVxx_eigenvalue>0.02]

# train_explain_numerical_COVxx_eigenvalue_over_zero = train_explain_numerical_COVxx_eigenvalue[train_explain_numerical_COVxx_eigenvalue<1]

train_explain_numerical_COVxx_eigenvalue_over_zero = train_explain_numerical_COVxx_eigenvalue_over_zero.dropna()

print(f"COVxx eigenvalue over 0 shape:{train_explain_numerical_COVxx_eigenvalue_over_zero.shape}")

train_explain_numerical_COVxx_eigenvalue_over_zero
## Dataの準備

norm_linear_reg_train_explain_data_A = train_explain_not_null_numerical_data[train_explain_numerical_COVxx_eigenvalue_over_zero.index]
norm_linear_reg_train_explain_data_A.isnull().sum()
from sklearn.model_selection import KFold

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.metrics import mean_squared_error

# -----------------------------------

# バリデーション

# -----------------------------------



def kfold_rmse_validation(linear_reg_model, X, y, poly_degree):

    """

    目的：線形モデルにクロスバリデーションを使用して評価

    入力： linear_reg_model:線形モデル, X: 説明変数(独立変数), y:目的変数(従属変数)

    出力:  なし

    """

    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)

    # 各foldのスコアを保存するリスト

    score_rmse_list = []



    # クロスバリデーションを行う

    # 学習データを8つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す

    kf = KFold(n_splits=8, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(X):

        x_tr, x_va = X.iloc[tr_idx], X.iloc[va_idx]

        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        

        poly_x_tr = poly_features.fit_transform(x_tr.to_numpy().reshape(len(x_tr), -1))

        poly_x_va = poly_features.fit_transform(x_va.to_numpy().reshape(len(x_va), -1))

        

        linear_reg_model.fit(poly_x_tr, y_tr)



        pred_va = linear_reg_model.predict(poly_x_va)

        rmse = np.sqrt(mean_squared_error(y_va, pred_va)) 



        score_rmse_list.append(rmse)

    

    def display_score_rmse(score_rmse_list):

        print(f"{'='*30} VAL RMSE result {'='*30}")

        print(f"VAL Scores:{score_rmse_list}")

        print(f"VAL Mean:{np.mean(score_rmse_list)}")

        print(f"VAL Std:{np.std(score_rmse_list)}")

    

    display_score_rmse(score_rmse_list)
from sklearn import linear_model

import sklearn.model_selection





model_A = linear_model.LinearRegression()

X_A = norm_linear_reg_train_explain_data_A

y_A = train_object_data

poly_degree = 1

kfold_rmse_validation(model_A, X_A, y_A, poly_degree)
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split





def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))

        val_errors.append(mean_squared_error(y_val, y_val_predict))

#         print(f"Intercept: {model['lin_reg'].intercept_}", f"param θ: {model['lin_reg'].coef_}")

        

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")

    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

    plt.legend(loc="upper right", fontsize=14)   # not shown in the book

    plt.xlabel("Training set size", fontsize=14) # not shown

    plt.ylabel("RMSE", fontsize=14)              # not shown





polynomial_regression = Pipeline([

        ("poly_features", PolynomialFeatures(degree=poly_degree, include_bias=False)),

        ("lin_reg", linear_model.LinearRegression()),

    ])



plot_learning_curves(polynomial_regression, X_A.to_numpy().reshape(len(X_A), -1), y_A.to_numpy().reshape(len(y_A), -1))

# plt.axis([0, 80, 0, 3])           # not shown

# plt.save_fig("learning_curves_plot")  # not shown

plt.show()                        # not shown
polynomial_regression = Pipeline([

        ("poly_features", PolynomialFeatures(degree=poly_degree, include_bias=False)),

        ("lin_reg", linear_model.LinearRegression()),

    ])





X_A = norm_linear_reg_train_explain_data_A

y_A = train_object_data

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_A.to_numpy().reshape(len(X_A), -1), y_A.to_numpy().reshape(len(y_A), -1), random_state=42)

polynomial_regression.fit(X_train, y_train)

predict = polynomial_regression.predict(X_val)

print(f"Intercept: {polynomial_regression['lin_reg'].intercept_}", f"param θ: {polynomial_regression['lin_reg'].coef_}")

coef_sum = polynomial_regression['lin_reg'].coef_.sum()

# print(coef_sum)

print( f"param contribution proportion θ: {polynomial_regression['lin_reg'].coef_ / coef_sum}")

    

fig, ax = plt.subplots(1,3, sharey='row')

ax[0].plot(np.arange(len(X_train)), y_train, "b.")



ax[1].plot(predict, y_val, "r.")

ax[1].plot(np.arange(0, 750000),np.arange(0, 750000), "b-")

# ax[1].plot(np.arange(len(X_val)), predict, "r-", linewidth=2) 



s = 10

lim = 300

ax[2].plot(np.arange(len(X_val))[s:s+lim], y_val[s:s+lim], "+")

ax[2].plot(np.arange(len(X_val))[s:s+lim], predict[s:s+lim], "r-", linewidth=2) 



plt.tight_layout()

plt.show()
train_object_data.describe()
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)

poly_A = poly_features.fit_transform(norm_linear_reg_train_explain_data_A)

norm_linear_reg_train_explain_data_A.iloc[0], poly_A[0]
import math

explain_n = 5

d = 5



math.factorial(explain_n + d) / (math.factorial(explain_n) * math.factorial(d))
# テスト用

test_explain_data = test_house_prices_origin_data.drop(columns=["Id"])



# 数値属性でNOTNULLのデータのみ抽出

#TODO: データが数値(Numerical)属性のみを抽出

attributes = ['YearBuilt', 'GrLivArea',

       'GarageYrBlt', 'YearRemodAdd', 'GarageArea', 'TotalBsmtSF','1stFlrSF']

test_explain_numerical_data = test_explain_data[attributes]

#TODO: 欠損値は平均で埋める

test_explain_not_null_numerical_data = pd.DataFrame(num_pipeline.fit_transform(test_explain_numerical_data), columns=test_explain_numerical_data.columns)



## Dataの準備

linear_reg_test_explain_data_A = test_explain_not_null_numerical_data[train_explain_numerical_COVxx_eigenvalue_over_zero.index]



# 学習・評価

X_A = norm_linear_reg_train_explain_data_A

y_A = train_object_data



test_X_A = linear_reg_test_explain_data_A



polynomial_regression.fit(X_A, y_A)



predict_test = polynomial_regression.predict(test_X_A)



submission = pd.DataFrame({"Id":test_house_prices_origin_data["Id"], "SalePrice":predict_test})

submission.to_csv('submission_third.csv', index=False)
# 対角化　計算機上は0にならないでもとても小さい値となっている

diagonalization = LA.inv(train_explain_numerical_eigenvector) @ train_explain_numerical_COVxx @ train_explain_numerical_eigenvector

diagonalization

# train_explain_numerical_cov_inv = pd.DataFrame(diagonalization, train_explain_numerical_COVxx.columns, train_explain_numerical_COVxx.index, dtype=np.float64)
# 順番は正しいか求めたい

# 順序は大丈夫そう

x1 = [2,2,4]# 独立

x2 = [2,4,2]# 独立

x3 = [1,1,2]# 従属

y = [0,2,4]



df = pd.DataFrame([x1,x2,x3], index=["x1","x2","x3"])

Cxx = pd.DataFrame(np.cov(m=df, bias=True), columns=["x1","x2","x3"], index=["x1","x2","x3"])

_eigenvalue, _eigenvector = LA.eig(Cxx)
_eigenvalue
from sklearn import linear_model

import sklearn.model_selection



model_A = linear_model.LinearRegression()

X = norm_linear_reg_train_explain_data_A

y = train_object_data



X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X,y)

model_A.fit(X_train, y_train)

# model_A.score(X_train, y_train), model_A.score(X_val, y_val)



from sklearn.metrics import mean_squared_error

pred_model_A = model_A.predict(X_val)

rmse_A = np.sqrt(mean_squared_error(y_val, pred_model_A))

rmse_A

model_A.predict_proba(X_val)
polynomial_regression = Pipeline([

        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),

        ("lin_reg", linear_model.LinearRegression()),

    ])



X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_A.to_numpy().reshape(len(X_A), 1), y_A.to_numpy().reshape(len(y_A), 1))

polynomial_regression.fit(X_train, y_train)

predict = polynomial_regression.predict(X_val)

# plt.axis([0, 80, 0, 3])           # not shown

# plt.save_fig("learning_curves_plot")  # not shown

# plt.plot(np.arange(len(X_train)), y_train, "b.")

plt.plot(np.arange(len(X_val)), predict, "r-")

plt.plot(np.arange(len(X_val)), y_val, "+")



plt.show()                        # not shown



print(polynomial_regression["lin_reg"].intercept_, polynomial_regression["lin_reg"].coef_)
import numpy as np

import numpy.random as rnd



np.random.seed(42)



m = 100

X = 6 * np.random.rand(m, 1) - 3

y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)





plt.plot(X, y, "b.")

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.axis([-3, 3, 0, 10])

save_fig("quadratic_data_plot")

plt.show()


from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_features.fit_transform(X)

X[0], X_poly[0]
lin_reg = linear_model.LinearRegression()

lin_reg.fit(X_poly, y)

lin_reg.intercept_, lin_reg.coef_


X_new=np.linspace(-3, 3, 100).reshape(100, 1)

X_new_poly = poly_features.transform(X_new)

y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.")

plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")

plt.xlabel("$x_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.legend(loc="upper left", fontsize=14)

plt.axis([-3, 3, 0, 10])

# save_fig("quadratic_predictions_plot")

plt.show()
from sklearn import linear_model

import sklearn.model_selection





model_A = linear_model.LinearRegression()

X_A = norm_linear_reg_train_explain_data_A

# X_A = X_A.to_numpy().reshape(len(X_A), 1)

y_A = train_object_data



# model_B = linear_model.LinearRegression()

# X_B = norm_linear_reg_train_explain_data_B

# X_B = X_B.to_numpy().reshape(len(X_B), 1)

# y_B = train_object_data



poly_degree = 1

kfold_rmse_validation(model_A, X_A, y_A, poly_degree)

# kfold_rmse_validation(model_B, X_B, y_B, poly_degree)