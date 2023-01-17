# According to Kaggle, my model submission result is 0.127 and 13463.00913 (RMSE) 
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


def data():
    train_ = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
    test_ = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
    dataframe = pd.concat([train_, test_], ignore_index=True)
    return dataframe, train_, test_
df, train, test = data()


def info_(dataframe):
    print(dataframe.shape)

for i in [df, train, test]:
    info_(i)


# NİHAİ DF ( CATBOOST EN ÖNEMLİ DEĞİŞKENLER, ORADA 70 TANE VARDI AMA ONE HOT YAPTIĞIMIZ İÇİN TÜREYEN DEĞİŞKENLERDİ)
list = ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'TotalBsmtSF', 'BsmtFinSF1', 'LotArea', '2ndFlrSF',
        'YearRemodAdd', 'FullBath', 'GarageArea', 'YearBuilt', 'OverallCond', 'Fireplaces', 'LotFrontage',
        'BsmtFullBath', 'HalfBath', 'OpenPorchSF', 'WoodDeckSF', 'GarageYrBlt', 'Id', 'BedroomAbvGr', 'BsmtUnfSF',
        'ScreenPorch', 'MoSold', 'MasVnrArea', 'KitchenAbvGr', 'YrSold', 'TotRmsAbvGrd', 'MSSubClass', 'EnclosedPorch',
        "ExterCond", "KitchenQual", "Neighborhood", "CentralAir", "MSZoning", "SaleCondition", "Condition1",
        "ExterQual", "LandContour", "Functional", "Exterior2nd", "BsmtFinType1", "Exterior1st", "BsmtQual",
        "GarageType", "LotConfig", "HouseStyle", "SaleType", "HeatingQC", "BsmtExposure", "GarageFinish", "MasVnrType",
        "RoofStyle", "GarageCond", "RoofMatl", 'SalePrice']

df = df[list]
print(df.shape)


# BU ÇALIŞMADA DEĞİŞKENLERİ 4 LİSTE HALİNDE İNCELEDİM: NOM-ORD-NUMLIST-NUMLIST2.
# BU LİSTELERE GÖRE FARKLI ÖN İŞLEMELER YAPTIM

obj_list = [col for col in df.columns if df[col].dtypes == "O"]

ord_list = []
for i in obj_list:
    if df[i].str.contains('Gd', regex=True).any() == True:
        ord_list.append(i)

nom_list = []
for i in obj_list:
    if i not in ord_list:
        nom_list.append(i)

num_list = [] # bildiğimiz numerikler
num_list2 = [] # numerik görünümlü kategorikler
for i in df.columns:
    if df[i].dtypes != "O":
        if df[i].nunique() < 15:
            num_list2.append(i)
        elif df[i].nunique() >= 15:
            num_list.append(i)

# TÜM DEĞİŞKENLERİ ALDIM MI, KONTROLÜ
len(num_list2) + len(num_list) + len(nom_list) + len(ord_list)

# ORDINAL
# ORDİNAL DEĞİŞKENLERDE KAGGLE DA OLAN EX-GD-TA.. SIRALAMASINA GÖRE LABEL ENCODER YAPTIM.
for i in df[ord_list].columns:
    for k in range(len(df)):
        if i not in "BsmtExposure":
            if df.loc[k, i] == "Ex":
                df.loc[k, i] = 6
            elif df.loc[k, i] == "Gd":
                df.loc[k, i] = 5
            elif df.loc[k, i] == "TA":
                df.loc[k, i] = 4
            elif df.loc[k, i] == "Fa":
                df.loc[k, i] = 3
            elif df.loc[k, i] == "Po":
                df.loc[k, i] = 2
            else:
                df.loc[k, i] = 1

# BU DEĞİŞKENDEKİ KAGGLE SIRALAMASI FARKLI İDİ: GD-AV-MN GİBİ. BUNU AYRICA DÜZENLEDİM
df.loc[df["BsmtExposure"] == "Gd", "BsmtExposure"] = 5
df.loc[df["BsmtExposure"] == "Av", "BsmtExposure"] = 4
df.loc[df["BsmtExposure"] == "Mn", "BsmtExposure"] = 3
df.loc[df["BsmtExposure"] == "No", "BsmtExposure"] = 2
df["BsmtExposure"] = df["BsmtExposure"].fillna(1)

# NOMINAL

# NA LERE XNA ATAMA
# ( BUNU YAPMAMIN NEDENİ: BAZI İŞLEMLER VERİ SETİNDE NA OLUNCA GERÇEKLEŞMİYOR, BU NEDENLE BİR İSİM KOYDUM)

for i in df[nom_list]:
    df[i].replace(to_replace=np.nan, value='XNA', regex=True, inplace=True)

# RARE ATAMA
# ALT SINIFI 0.05'TEN KÜÇÜK OLANLARA RARE ATADIM.
for i in df[nom_list]:
    for k in df[i].unique():
        if df.loc[df[i] == k, i].shape[0] / len(df) < 0.05:
            df.loc[df[i] == k, i] = "Rare"


# ONE HOT SONUCUNDA SHAPE: (2919,94). 57 SATIRDAN 94 SÜTUN ÇIKTI.
def one_hot_encoder(dataframe, categorical_cols):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=True)
    return dataframe


df2 = one_hot_encoder(df, nom_list)

# NUM LİST2

# MEDIAN ATAMA
# BİR DEĞİŞKENDE 10'DAN AZ EKSİK DEĞER VARSA MEDİAN ATADIM.
for i in df2[num_list2]:
    if df2[i].isnull().sum() < 10:
        df2[i] = df2[i].fillna(df2[i].median())


# ALT SINIF ANALİZİ YAPTIM. SONRASINDA TEK TEK DEĞİŞKENLERİ İNCELEDİM.

def class_analyze(dataframe, target, liste):
    for var in liste:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()})
              .sort_values(by="TARGET_MEDIAN", ascending=False), end="\n\n\n")


class_analyze(df2, "SalePrice", num_list2)

# İNCELEDİĞİM DEĞİŞKENLER AŞAĞIDAKİ GİBİ. SALEPRİCE DEĞERLERİNE GÖRE SIRALAMA YAPARAK LABEL YAPTIM.
# OverallQual ok
# GarageCars
df2.loc[df["GarageCars"] == 3, "GarageCars"] = 5
df2.loc[df["GarageCars"] == 2, "GarageCars"] = 3
df2.loc[df["GarageCars"] == 1, "GarageCars"] = 2
df2.loc[df["GarageCars"] == 0, "GarageCars"] = 1
# FullBath ok
# OverallCond
df2.loc[df["OverallCond"] == 5, "OverallCond"] = 9
df2.loc[df["OverallCond"] == 9, "OverallCond"] = 8
df2.loc[df["OverallCond"] == 8, "OverallCond"] = 5
# Fireplaces ok
# BsmtFullBath 
df2.loc[df["BsmtFullBath"] == 2, "BsmtFullBath"] = 3
df2.loc[df["BsmtFullBath"] == 3, "BsmtFullBath"] = 2
# HalfBath  
df2.loc[df["HalfBath"] == 1, "HalfBath"] = 2
df2.loc[df["HalfBath"] == 2, "HalfBath"] = 1
# BedroomAbvGr
df2.loc[df["BedroomAbvGr"] == 0, "BedroomAbvGr"] = 8
df2.loc[df["BedroomAbvGr"] == 8, "BedroomAbvGr"] = 7
df2.loc[df["BedroomAbvGr"] == 4, "BedroomAbvGr"] = 6
df2.loc[df["BedroomAbvGr"] == 3, "BedroomAbvGr"] = 5
df2.loc[df["BedroomAbvGr"] == 5, "BedroomAbvGr"] = 4
df2.loc[df["BedroomAbvGr"] == 1, "BedroomAbvGr"] = 3
df2.loc[df["BedroomAbvGr"] == 6, "BedroomAbvGr"] = 2
df2.loc[df["BedroomAbvGr"] == 2, "BedroomAbvGr"] = 1
# MoSold
df2.loc[df["MoSold"] == 9, "MoSold"] = 12
df2.loc[df["MoSold"] == 12, "MoSold"] = 11
df2.loc[df["MoSold"] == 8, "MoSold"] = 10
df2.loc[df["MoSold"] == 2, "MoSold"] = 9
df2.loc[df["MoSold"] == 11, "MoSold"] = 8
df2.loc[df["MoSold"] == 3, "MoSold"] = 7
df2.loc[df["MoSold"] == 7, "MoSold"] = 6
df2.loc[df["MoSold"] == 6, "MoSold"] = 5
df2.loc[df["MoSold"] == 10, "MoSold"] = 4
df2.loc[df["MoSold"] == 5, "MoSold"] = 3
df2.loc[df["MoSold"] == 1, "MoSold"] = 2
df2.loc[df["MoSold"] == 4, "MoSold"] = 1
# KitchenAbvGr
df2.loc[df["KitchenAbvGr"] == 1, "KitchenAbvGr"] = 3
df2.loc[df["KitchenAbvGr"] == 2, "KitchenAbvGr"] = 2
df2.loc[df["KitchenAbvGr"] == 0, "KitchenAbvGr"] = 1
df2.loc[df["KitchenAbvGr"] == 3, "KitchenAbvGr"] = 0
# TotRmsAbvGrd
df2.loc[df["TotRmsAbvGrd"] == 11, "TotRmsAbvGrd"] = 14
df2.loc[df["TotRmsAbvGrd"] == 10, "TotRmsAbvGrd"] = 13
df2.loc[df["TotRmsAbvGrd"] == 9, "TotRmsAbvGrd"] = 12
df2.loc[df["TotRmsAbvGrd"] == 8, "TotRmsAbvGrd"] = 11
df2.loc[df["TotRmsAbvGrd"] == 12, "TotRmsAbvGrd"] = 10
df2.loc[df["TotRmsAbvGrd"] == 14, "TotRmsAbvGrd"] = 9
df2.loc[df["TotRmsAbvGrd"] == 7, "TotRmsAbvGrd"] = 8
df2.loc[df["TotRmsAbvGrd"] == 6, "TotRmsAbvGrd"] = 7
df2.loc[df["TotRmsAbvGrd"] == 5, "TotRmsAbvGrd"] = 6
df2.loc[df["TotRmsAbvGrd"] == 4, "TotRmsAbvGrd"] = 5
df2.loc[df["TotRmsAbvGrd"] == 3, "TotRmsAbvGrd"] = 4
df2.loc[df["TotRmsAbvGrd"] == 2, "TotRmsAbvGrd"] = 3
df2.loc[df["TotRmsAbvGrd"] == 13, "TotRmsAbvGrd"] = 2
df2.loc[df["TotRmsAbvGrd"] == 15, "TotRmsAbvGrd"] = 1


# NUM LİST
df2.loc[df2["GrLivArea"] > 4000, "GrLivArea"] = 3800
df2.loc[df2["1stFlrSF"] > 3500, "1stFlrSF"] = 3500
df2.loc[df2["TotalBsmtSF"] > 4000, "TotalBsmtSF"] = 3200
df2.loc[df2["BsmtFinSF1"] > 3000, "BsmtFinSF1"] = 1270  # 0.95 quantile baskıladım.
df2.loc[df2["2ndFlrSF"] > 1750, "2ndFlrSF"] = 1750
df2.loc[df2["GarageArea"] > 1330, "GarageArea"] = 1330
df2.loc[df2["LotFrontage"] > 200, "LotFrontage"] = 200
df2.loc[df2["WoodDeckSF"] > 736, "WoodDeckSF"] = 736
df2.loc[df["GarageArea"] == 0, "GarageYrBlt"] = 0  # Garage Area'ları yok ama yapıldığı yıl 1979, bunun yerine 0 verdim.
df2.loc[df2["BsmtUnfSF"] >= 2122, "BsmtUnfSF"] = 2122
df2.loc[df2["MasVnrArea"] > 1230, "MasVnrArea"] = 1230


# VERİ SETİNDEKİ DİĞER NA'LAR

# EKSİK DEĞER SAYISI 30'DAN AZ OLANLARA MEDIAN ATA
for i in df2.columns:
    if df2[i].isnull().sum() < 30:
        df2[i] = df2[i].fillna(df2[i].median())


# NA OLAN SALEPRICE DIŞINDA SADECE LOTFRONTAGE KALDI.

# NA LERE XNA ATAMA
df2["LotFrontage"].replace(to_replace=np.nan, value=-99, regex=True, inplace=True)

# LotFrontage' da NA olanların SalePrice ortalaması 172k
df2.loc[df2["LotFrontage"] == -99, "SalePrice"].median()

# Genelde SalePrice ortalaması 163k.
df2["SalePrice"].median()

# İKİ DEĞER BİRBİRİNE ÇOK YAKIN. BU NEDENLE MEDIAN ATAYABILIRIM.
df2.loc[df2["LotFrontage"] == -99, "LotFrontage"] = 65

#df2.loc[(df2["SalePrice"]<=175000)&(df2["SalePrice"]>=165000), "LotFrontage"].median()



# DİĞER DEĞİŞKENLER

df3 = df2.copy()
df3.loc[df3["OverallQual"] < 2, "OverallQual"] = 2
df3.loc[df3["GarageCars"] > 4, "GarageCars"] = 4
df3.loc[(df3["OverallQual"] == 2) | (df3["OverallQual"] == 3), "OverallQual"] = 3
# SalePrice 163k, genelde 163k salepricelıların garaj built değeri 2005
df3.loc[df3["GarageYrBlt"] == 2207, "GarageYrBlt"] = 2005
df3.loc[df3["Fireplaces"] == 4, "Fireplaces"] = 0
df3.loc[df3["Fireplaces"] == 3, "Fireplaces"] = 2
# NEW_1
df3["new_area"] = df3["GrLivArea"] + df3["GarageArea"]
# NEW_2  #1K'YA YAKIN DÜŞTÜ
df3["new_home"] = df3["YearBuilt"]
df3.loc[df3["new_home"] == df3["YearRemodAdd"], "new_home"] = 0
df3.loc[df3["new_home"] != df3["YearRemodAdd"], "new_home"] = 1
# NEW_3  #Banyo sayısı toplamları
df3["new_bath"] = df3["FullBath"] + (df3["HalfBath"] * 0.5)


# CATBOOST MODEL : 20.18K
X = df3.loc[:1459, :].drop(["SalePrice", "Id"], axis=1)
y = df3.loc[:1459, "SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

catboost_model = CatBoostRegressor()
catboost_model = catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_pred, y_test)))


# TO CSV , KAGGLE'A YÜKLEMEK İÇİN
df107 = pd.DataFrame({"Id": df3.loc[1460:, "Id"],
                      "SalePrice": catboost_model.predict(df3.loc[1460:, :].drop(["SalePrice", "Id"], axis=1))})

df107.to_csv('deneme7.csv', index=False)



# Robust Scaler
df4=df3.copy()
from sklearn.preprocessing import RobustScaler
robust_list=[col for col in df4.columns if col not in "Id" and col not in "SalePrice"]

col=robust_list
x_transformed=pd.DataFrame(RobustScaler().fit(df4.loc[:,robust_list]).transform(df4.loc[:,robust_list]), columns=col)
x_transformed.head()

non_robust_list=[col for col in df4.columns if col not in robust_list]
df5=pd.concat([df4.loc[:,non_robust_list], x_transformed], axis=1)
df5.head(2)


# CATBOOST MODEL : 20.07K
X = df5.loc[:1459, :].drop(["SalePrice", "Id"], axis=1)
y = df5.loc[:1459, "SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

catboost_model = CatBoostRegressor()
catboost_model = catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_pred, y_test)))


# TO CSV , KAGGLE'A YÜKLEMEK İÇİN

df108 = pd.DataFrame({"Id": df5.loc[1460:, "Id"],
                      "SalePrice": catboost_model.predict(df5.loc[1460:, :].drop(["SalePrice", "Id"], axis=1))})

df108.to_csv('deneme8.csv', index=False)


