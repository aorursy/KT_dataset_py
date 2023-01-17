# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/tglf-and-ml/TAML_v1.csv")



dataset.shape
x = dataset.iloc[:, 0:22]

y = dataset.iloc[:, 22:28]
inv_x = x.join((1/x).add_prefix('inv_'))

x = inv_x

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inv_x, y, test_size = 0.15, random_state = 2)



y_cols = list(y.columns)

x_cols = list(x.columns)
import seaborn as sns

import matplotlib.pyplot as plt

corrmat = dataset.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# for y in y_cols:

#     y_train[y] = y_train[y] / (max(abs(y_train[y])) - min(abs(y_train[y])))

#     y_test[y] = y_test[y] / (max(abs(y_train[y])) - min(abs(y_train[y])))

# for x in x_cols:

#     x_train[x] = x_train[x] / (max(abs(x_train[x])) - min(abs(x_train[x])))

#     x_test[x] = x_test[x] / (max(abs(x_train[x])) - min(abs(x_train[x])))
from xgboost import XGBRegressor





for i in y_cols:

    xgb = XGBRegressor()

    xgb.fit(x_train, y_train[i])



    y_pred = xgb.predict(x_test)



    from sklearn.metrics import r2_score

    print(i,r2_score(y_test[i],y_pred))

    

    cost = sum((y_pred - y_test[i])**2) / sum((y_test[i] - np.mean(y_pred))**2)

    print("Cost",cost)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression





def poly_train(x_train, x_test, y_train, y_test, degree):

    poly = PolynomialFeatures(degree = degree)

    x_poly = poly.fit_transform(x_train)

    poly.fit(x_poly, y_train)

#     for i in y_cols:

    model = LinearRegression()

    model.fit(x_poly, y_train)



    y_pred = model.predict(poly.fit_transform(x_test))



    from sklearn.metrics import r2_score

    r_score = r2_score(y_pred,y_test)

    

    if r_score > 0:

        cost = 1-sum((y_pred - y_test)**2) / sum((y_test - np.mean(y_pred))**2)

        print("CorrCoef: ",cost)



        print("R-score: ",r_score)





        coefs = list(model.coef_)[1:]

        intercept = str(model.intercept_)

        features = poly.get_feature_names(x_train.columns)[1:]



        equation = ""

        for i in range(len(coefs)):

            equation += str(coefs[i])+"*"+features[i]+" + "

        equation += intercept

        print("Equation: ",equation)

        

        import matplotlib.pyplot as plt

        corr = LinearRegression()

        corr.fit(y_test.values.reshape(-1, 1), y_pred)

        plt.scatter(y_test, y_pred)

        plt.plot(y_test, corr.predict(y_test.values.reshape(-1, 1)), linewidth=5, color="red")

#         plt.plot(x_test, y_pred)

        plt.xlabel("y actual")

        plt.ylabel("y predicted")



        plt.show()

        
# for y in y_cols:

#     gp.fit(x_train, y_train[y])

#     gp_features_test = gp.transform(x_test)

#     gp_features_train = gp.transform(x_train)

    

#     transformed_x_test = np.hstack((x_test, gp_features_test))

#     transformed_x_train = np.hstack((x_train, gp_features_train))

#     poly_train(transformed_x_train, transformed_x_test, y_train[y], y_test[y], 2)
# gp.fit(x_train.drop(["ZEFF","AS_3"], axis=1), y_train["aa0"])

# gp_features_test = gp.transform(x_test.drop(["ZEFF","AS_3"], axis=1))

# gp_features_train = gp.transform(x_train.drop(["ZEFF","AS_3"], axis=1))

# transformed_x_test = np.hstack((x_test.drop(["ZEFF","AS_3"], axis=1), gp_features_test))

# transformed_x_train = np.hstack((x_train.drop(["ZEFF","AS_3"], axis=1), gp_features_train))

# poly_train(transformed_x_train, transformed_x_test, y_train["aa0"], y_test["aa0"], 2)
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.feature_selection import SelectKBest

import matplotlib.pyplot as plt



def getFeatures(column):

    #Extra Trees

    feature_tree = ExtraTreesRegressor()

    feature_tree.fit(x,y[column])



    feat_importances_tree = pd.Series(feature_tree.feature_importances_, index=x.columns)

    tree_features = feat_importances_tree.nlargest(44)

#     tree_features.plot(kind='barh')

#     plt.show()

    

    #KBest

    KBest = SelectKBest()

    feature_kbest = KBest.fit(x,y[column])

    feat_importances_kbest = pd.Series(feature_kbest.scores_, index=x.columns)

    kbest_features = feat_importances_kbest.nlargest(44)

#     kbest_features.plot(kind='barh')

#     plt.show()

    

    return kbest_features.keys(), tree_features.keys()
def generateEquations(complexity, column):

    features = 44-complexity

    kbest_feats, tree_feats = getFeatures(column)



    kbest_cols = []

    for i in kbest_feats[:-features]:

        kbest_cols.append(i)

        for degree in [1, 2]:

            poly_train(x_train[kbest_cols], x_test[kbest_cols], y_train[column], y_test[column], degree)

            

    tree_cols = []

    for j in tree_feats[:-features]:

        tree_cols.append(j)

        for degree in [1, 2]:

            poly_train(x_train[tree_cols], x_test[tree_cols], y_train[column], y_test[column], degree)
#aa0

generateEquations(5, "aa0")
#aa1

generateEquations(5, "aa1")
#bb0

generateEquations(10, "bb0")
#bb1

generateEquations(5, "bb1")
#cc0

generateEquations(5, "cc0")
#cc1

generateEquations(10, "cc1")
from numpy import arctan2 as atan2

from numpy import arctan as atan

from numpy import sin



aa01 = 0.000586878628672103/(x_test["BETAE"]*x_test["DEBYE"])



aa02 = 13.31109513097/(0.0356951755899343 + 18456.1611227435*x_test["BETAE"]*x_test["DEBYE"])



aa03 = 0.000897132049204281/(x_test["BETAE"]*x_test["DEBYE"]) + 0.0027584039153336*x_test["RLTS_1"]/x_test["P_PRIME_LOC"]



aa11 = 2.43573308629787 + (339.647805490297*x_test["DEBYE"] + 5.51132039007536*x_test["XNUE_LOG10"] - 2.30301539071201)/atan2(0.358195564905137, 1.49455454070443 + 1.3082115684864*x_test["XNUE_LOG10"])



aa12 = 2.43573308629787 + (340.019640794413*x_test["DEBYE"] + 5.51132039007536*x_test["XNUE_LOG10"] - 2.31242970377851)/atan2(0.554635214772797, 2.33742681637598 + 2.04599345048468*x_test["XNUE_LOG10"])



bb01 = sin(0.0451620655873398 + 5.74067564169686*x_test["BETAE"] + 0.394461825492145*x_test["S_KAPPA_LOC"] - 71.2410664957191*x_test["BETAE"]*x_test["S_KAPPA_LOC"])



bb02 = 0.0373669092380423 + 9.27590894685452*x_test["BETAE"] + 3.51301068069448*x_test["DEBYE"] + 0.234524015928965*x_test["S_KAPPA_LOC"] + 0.0667304749508477*x_test["XNUE_LOG10"] - 42.3557868820215*x_test["BETAE"]*x_test["S_KAPPA_LOC"]



bb11 = 48.1094819239912*x_test["DEBYE"]*atan2(0.165065, -x_test["XNUE_LOG10"])



bb12 = 51*x_test["DEBYE"]*atan2(x_test["DEBYE"], -0.191068891493344*x_test["XNUE_LOG10"])



cc02 = 2.24673040703783 + 0.456058785589525*atan(0.983824839288549*x_test["RLTS_1"] - 1.4705312523317)



cc03 = 2.55400033104862 + 0.111633236987856*x_test["Q_LOC"] - 0.191620091936767*0.168311834470472**(0.337000808894145*x_test["RLTS_1"] - 0.9935606507027)



cc04 = 3.16973758870942 + 0.122823053057227*x_test["Q_LOC"] - 0.492099058125246*x_test["TAUS_2"] - 0.268802753511135*0.236650833153819**(0.337000808894145*x_test["RLTS_1"] - 0.9935606507027)



cc11 = 2.45323490772952 + atan(0.352303406911234*x_test["AS_2"]*x_test["Q_LOC"]*x_test["RLTS_1"]) - x_test["AS_2"]*x_test["TAUS_2"]



cc12 = 2.74131903910125 + atan(0.252265295673248*x_test["Q_LOC"]*x_test["RLTS_1"]) - x_test["TAUS_2"]



cc13 = 2.19772511259577 + atan(0.296300432325849*x_test["Q_LOC"]*x_test["RLTS_1"]) - 0.576397030142722*x_test["TAUS_2"]
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = aa01

y = y_test["aa0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = aa02

y = y_test["aa0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = aa03

y = y_test["aa0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = aa11

y = y_test["aa1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = aa12

y = y_test["aa1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = bb01

y = y_test["bb0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = bb02

y = y_test["bb0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = bb11

y = y_test["bb1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = bb12

y = y_test["bb1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = cc02

y = y_test["cc0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = cc03

y = y_test["cc0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = cc04

y = y_test["cc0"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = cc11

y = y_test["cc1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = cc12

y = y_test["cc1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



y_pred = cc13

y = y_test["cc1"]

corr = LinearRegression()

corr.fit(y.values.reshape(-1, 1), y_pred)

plt.scatter(y, y_pred)

plt.plot(y, corr.predict(y.values.reshape(-1, 1)), linewidth=5, color="red")

plt.xlabel("y actual")

plt.ylabel("y predicted")



plt.show()