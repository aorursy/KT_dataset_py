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
# Procedemos a importar las librerias que usaremos a lo largo del análisis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

style.use("seaborn")
pd.options.display.float_format = '{:.6f}'.format
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col=0)
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col=0)
df.head()
# Observemos la distribución de los datos, mediante un histograma.
plt.figure(figsize=(9,8))
sns.distplot(df['SalePrice'], bins = 100, hist_kws={'alpha': 0.4,"color": "plum"})
# calculamos algunas estadísticas básica de todos los datos en una columna, para ello usamos:
df.describe()
print("Total de valores nulos  en TRAIN", end="\n\n")
for col in df.columns:
    w = df[col].isna().sum()
    if w > 0:
        print(col.ljust(40, " ") + str(w))
test.describe()
print("Total de valores nulos  en TEST", end="\n\n")
for col in test.columns:
    z = test[col].isna().sum()
    if z > 0:
        print(col.ljust(40, " ") + str(z))
#total_data["SalePrice"] = df["SalePrice"]
df_num = df.select_dtypes(include = ['float64','int64'])
df_num.hist(figsize=(30,20), bins = 50, xlabelsize = 10, ylabelsize=9);
df_num_corr = df_num.corr()["SalePrice"][:-1]
golden_features_list = df_num_corr[abs(df_num_corr)> 0.5].sort_values(ascending=False)
print("Existen {} valores fuertemente correlacionados con la variable objetivo SalePrice:\n{}".format(len(golden_features_list),
golden_features_list))
cols_to_use = golden_features_list.index.values.reshape(2, 5)
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(len(cols_to_use)):
    for j in range(len(cols_to_use[i])):
        axs[i][j].scatter(df[cols_to_use[i][j]], df["SalePrice"])
        axs[i][j].set_xlabel(cols_to_use[i][j])

plt.tight_layout()
plt.show()
corr = df_num.corr()
plt.figure(figsize=(20,8))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], annot=True)
categoric = np.array(["MSSubClass","LotShape","LandContour",
"BldgType","HouseStyle","OverallQual","OverallCond",
"RoofStyle","RoofMatl","Exterior1st","Exterior2nd","ExterQual","ExterCond","MasVnrType",
"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","Electrical",
"KitchenQual","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","MiscFeature","LotConfig",
"MiscVal","MoSold","YrSold"]).reshape(11, 3)

fig, axs = plt.subplots(11, 3, figsize=(40, 70))
for i in range(len(categoric)):
    for j in range(len(categoric[i])):
        sns.boxplot(x=df[categoric[i][j]], y=df["SalePrice"], ax=axs[i][j])
plt.show()
total_data = pd.concat([df.drop("SalePrice", axis=1), test])
# Revisamos la cantidad de nulos en las columnas.
print("Total de valores nulos por columna", end="\n\n")
for col in total_data.columns:
    num_nan = total_data[col].isna().sum()
    if num_nan > 0:
        print(col.ljust(40, " ") + str(num_nan))
import missingno

# Extraigo los nombres de la columnas que presentan valores nulos.
null_cols = (total_data.isna()).sum()[total_data.isna().sum() > 0].index

# Matriz gráfica donde los espacios en blanco representan valores nulos
missingno.matrix(total_data[null_cols])
# Los valores nulos en MasVnrType significan que no tienen "Masonry veneer" y esos ya están representado por el valor "None"
# En la segunda serie se tiene que valores donde MasVnrType es "none" hay valores de MasVnrArea mayor que cero, se procederá a reemplazarlos por cero.
total_data.loc[:, "MasVnrArea"] = total_data["MasVnrArea"].fillna(0)
total_data.loc[:, "MasVnrType"] = total_data["MasVnrType"].fillna("None")
total_data.loc[2611, "MasVnrArea"] = 0
# se ha identificado un valor contradictorio donde tiene un valor del tipo None y un campo completo de Area por ende sera = 0.
# Se rellenará los valores nulos de las siguientes columnas con la moda por su
# alta frecuencia de éstas.
total_data["MSZoning"].fillna(total_data["MSZoning"].mode()[0], inplace=True)
total_data["Electrical"].fillna(total_data["Electrical"].mode()[0], inplace=True)
total_data["Utilities"].fillna(total_data["Utilities"].mode()[0], inplace=True)
total_data["Functional"].fillna(total_data["Functional"].mode()[0], inplace=True)
total_data["SaleType"].fillna(total_data["SaleType"].mode()[0], inplace=True)

# Los valores nulos se debe a la falta de cada estructura.
total_data["Alley"].fillna("None", inplace=True)
total_data["FireplaceQu"].fillna("None", inplace=True)
total_data["PoolQC"].fillna("None", inplace=True)
total_data["Fence"].fillna("None", inplace=True)
total_data["MiscFeature"].fillna("None", inplace=True)

# Las siguiente variables se llenarán con Other porque no se especifica el tipo
# y no se indica un valor específico para un nulo en el diccionario de términos.
total_data["Exterior1st"].fillna("Other", inplace=True)
total_data["Exterior2nd"].fillna("Other", inplace=True)

# Se rellena el nulo encontrado en KitchenQual con el score de Tipical/Average
# para no afectar mucho en su resultado final de calidad del hogar (solo es un nulo).
total_data["KitchenQual"].fillna("TA", inplace=True)
# Las siguiente columnas se llenarán con None debido a que estos hogares no
# presentan basement (sótano)
about_bsmt = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
for bsmt in about_bsmt:
    total_data[bsmt].fillna("None", inplace=True)

# Las siguiente columnas se rellenarán con cero ya que al no tener sótano la medida
# que correspondería es cero.
about_bsmt_feet = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
for bsmt_feet in about_bsmt_feet:
    total_data[bsmt_feet].fillna(0, inplace=True)

# Las siguiente columnas se llenarán con None debido a que estos hogares no
# presentan garaje.
about_garage = ["GarageFinish", "GarageType", "GarageQual", "GarageCond"]
for garage in about_garage:
    total_data[garage].fillna("None", inplace=True)

# Las siguiente columnas se llenarán con cero debido a que estos hogares no
# presentan garaje, las medidas que representan son cero.
about_garage_measure = ["GarageCars", "GarageArea", "GarageYrBlt"]
for garage_measure in about_garage_measure:
    total_data[garage_measure].fillna(0, inplace=True)
total_data["LotFrontage"].fillna(total_data["LotFrontage"].mean(), inplace=True)
missingno.matrix(total_data[null_cols])
total_data["SalePrice"] = df["SalePrice"]
#Nuestra primera transformación será realizada a partir de la variable "YearRemodAdd", asi generar las variables de año, mes y día.
date_remod = pd.to_datetime(total_data["YearRemodAdd"].apply(lambda val: str(val)))
date_sold = pd.to_datetime(pd.DataFrame({"year": total_data["YrSold"],
                                         "month": total_data["MoSold"],
                                         "day": [1 for _ in range(total_data.shape[0])]}))

# Se crea la variable que indica la cantidad de días que se demoró en venderse el hogar desde que fue remodelada.
total_data["DaysToSold"] = (date_sold - date_remod).dt.days
corr = total_data["DaysToSold"].corr(df["SalePrice"])
total_data.drop(["YrSold"], axis=1, inplace=True)
print("Correlación:", corr, end='\n\n')
sns.scatterplot(x=total_data["DaysToSold"], y=total_data["SalePrice"])
# Creación de la variable TotalRmNoBed que indica el total de cuartos en la casa incluido los espacios de baño.
total_rooms = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd"]
total_data["TotalRmNoBed"] = total_data[total_rooms].sum(axis=1)
corr = total_data["TotalRmNoBed"].corr(total_data["SalePrice"])
print("Correlación:", corr, end='\n\n')
total_data.drop(total_rooms, axis=1, inplace=True)
sns.scatterplot(x=total_data["TotalRmNoBed"], y=total_data["SalePrice"])
# Muestra mejor correlación que el uso independiente de cada variable.
total_data["TotalFloor"] = total_data["1stFlrSF"] + total_data["2ndFlrSF"] + total_data["TotalBsmtSF"]
corr = total_data["TotalFloor"].corr(df["SalePrice"])
# Y procedemos  a borrar del modelo las variables independientes, que sumadas me dan una mejor correlación.
total_data.drop(["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"], axis=1, inplace=True)
print("Correlación:", corr, end='\n\n')
sns.scatterplot(x=total_data["TotalFloor"], y=total_data["SalePrice"])
# Analizando las variables en relacion a un Deck y Porch.
# Unificamos las variables relacionadas, que hacen referencia a los tipos de Porch y su entrada WoodDeckSF, para ver juntas como se correlación con el precio
porchs = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
total_data["TotalWoodPorch"] = total_data[porchs].sum(axis=1)
corr = total_data["TotalWoodPorch"].corr(df["SalePrice"])
total_data.drop(porchs, axis=1, inplace=True)
print("Correlación:", corr, end='\n\n')
sns.scatterplot(x=total_data["TotalWoodPorch"], y=df["SalePrice"])
total_data["AreaQual"] = total_data["LotArea"].apply(np.log) * total_data["OverallQual"]
corr = total_data["AreaQual"].corr(total_data["SalePrice"])
print("Correlación:", corr, end='\n\n')

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].set_title("Total área del hogar vs Precio del hogar")
sns.scatterplot(x=total_data["LotArea"], y=total_data["SalePrice"], ax=ax[0])

ax[1].set_title("Log-normalization de LotArea")
sns.distplot(total_data.loc[:1460, "LotArea"].apply(np.log), ax=ax[1])

sns.scatterplot(x=total_data["AreaQual"], y=total_data["SalePrice"], ax=ax[2])
total_data.drop(["LotArea", "OverallQual"], axis=1, inplace=True)
plt.show()
# Creamos la variabble X_MSZoning
# Donde : NR = NO Residencial / R = Residencial  
def transform_MSZoning(MSZoning):
    if MSZoning == "A" or MSZoning == "C"  or MSZoning == "I":
        return "NRes"
    elif MSZoning == "RH" or MSZoning == "RL" or MSZoning == "RP" or MSZoning == "RM" or MSZoning == "FV":
        return "Res"

total_data["MSZoning"] = total_data["MSZoning"].apply(transform_MSZoning)

 # Se Transforma la variabble LotShape
 # Donde : IReg = Irregular / Reg = Regular 
def transform_LotShape(LotShape):
    if LotShape == "IR1" or LotShape == "IR2" or LotShape == "IR3":
        return "IReg"
    elif LotShape == "Reg" :
        return "Reg"
total_data["LotShape"] = total_data["LotShape"].apply(transform_LotShape)
total_data["LotShape"].unique()
def transform_LandSlope(LandSlope):
     if LandSlope == "Gtl":
        return 1
     elif LandSlope == "Mod":
        return 2
     elif LandSlope == "Sev":
        return 3

total_data["LandSlope"] = total_data["LandSlope"].apply(transform_LandSlope)
total_data["LandSlope"].unique()
def transform_Utilities(Utilities):
     if Utilities == "AllPub" :
        return 4
     elif Utilities == "NoSewr" :
        return 3
     elif Utilities == "NoSeWa" :
        return 2
     elif Utilities == "ELO" :
       return 1
total_data["Utilities"] = total_data["Utilities"].apply(transform_Utilities)
total_data["Utilities"].unique()
total_data['NumConditions'] = np.where(total_data["Condition1"] == total_data["Condition2"], 1, 2)
total_data.drop(["Condition1", "Condition2"], axis=1, inplace=True)
total_data["NumConditions"].unique()
total_data['Remodelacion'] = np.where(total_data["YearBuilt"] == total_data["YearRemodAdd"], "NRemod", "Remod")
total_data.drop(["YearBuilt", "YearRemodAdd"], axis=1, inplace=True)
total_data["Remodelacion"].unique()
total_data['NumExteriorType'] = np.where(total_data["Exterior1st"] == total_data["Exterior2nd"], 1, 2)
total_data.drop(["Exterior1st", "Exterior2nd"], axis=1, inplace=True)
total_data["NumExteriorType"].unique()
def transform_BsmtExposure(BsmtExposure):
    if BsmtExposure == "None" or "No":
        return 0
    elif BsmtExposure == "Gd" :
        return 3
    elif BsmtExposure == "Av" :
        return 2
    elif BsmtExposure == "Mn" :
        return 1
total_data["BsmtExposure"] = total_data["BsmtExposure"].apply(transform_BsmtExposure)

def transform_Electrical(Electrical):
    if Electrical == "SBrkr":
        return 4
    elif Electrical == "FuseA" :
        return 3
    elif Electrical == "FuseF" :
        return 2
    elif Electrical == "FuseP" :
        return 1
    elif Electrical == "Mix" :
        return 0

total_data["Electrical"] = total_data["Electrical"].apply(transform_Electrical)
total_data["Electrical"].unique()
def transform_Estaciones(MoSold):
    if MoSold in [12 , 1 , 2]:
        return "Invierno"
    elif MoSold in [3 , 4 , 5]:
        return "Primavera"
    elif MoSold in [6 , 7 , 8]:
        return "Verano"
    elif MoSold in [9 , 10 , 11]:
        return "Otoho"

total_data["Estaciones"] = total_data["MoSold"].apply(transform_Estaciones)
total_data.drop(["MoSold"], axis=1, inplace=True)
total_data["Estaciones"].unique()
def transform_LandContour(LandContour):
    if LandContour == "Lvl":
        return 1
    elif LandContour == "Bnk" :
        return 2
    elif LandContour == "HLS" :
        return 3
    elif LandContour == "Low" :
        return 4
     
total_data["LandContour"] = total_data["LandContour"].apply(transform_LandContour)
total_data["LandContour"].unique()
total_data["HasPool"] = np.where(total_data["PoolArea"] > 0, 1, 0)
total_data.drop(["PoolArea"], axis=1, inplace=True)
total_data["HasPool"].unique()
total_data["HasFence"] = np.where(total_data["Fence"] == "None", 0, 1)
total_data.drop(["Fence"], axis=1, inplace=True)
total_data["HasFence"].unique()
def transform_score(score):
    if score == "Ex":
        return 5
    elif score == "Gd":
        return 4
    elif score == "TA":
        return 3
    elif score == "Fa":
        return 2
    elif score == "Po":
        return 1
    else:
        return 0
  
qualities = ["ExterQual","BsmtQual", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual"]   
qual_df = total_data[qualities].applymap(transform_score)
qual_df["SalePrice"] = df["SalePrice"]
qual_df.head()

qual_melt = pd.melt(qual_df.dropna(), value_vars=qualities, id_vars=["SalePrice"])
qual_melt.head()
# Se realiza un grafico de boxplot para evaluar cada una de las calificaciones y/o consideraciones que se tiene para cada variable de calidad.
plt.figure(figsize=(20, 8))
sns.boxplot(data=qual_melt, x="variable", y="SalePrice", hue="value")
# Asignación del puntaje según calidad de estructuras.
total_data["Score"] = qual_df[qualities].sum(axis=1)
plt.figure(figsize=(20, 6))
sns.boxplot(data=total_data, y="SalePrice", x="Score")
total_data.drop(qualities, axis=1, inplace=True)

# Se obtienen las columnas que se van a escalar (reducir el rango).
# El escalamiento serán a todas las variables continuas, no se realizará
# el escalamiento a las variables Categoricas ni las jerárquicas.
"""cols_to_norm = []
for col in total_data.columns:
    unique = total_data[col].unique().shape[0]  #Primero se identifica la cantidad de valores del tipo variables continuos unicos los cuales a su vez no supera los 100 items.
    if unique > 100 and col != "SalePrice":
        cols_to_norm.append(col)
cols_to_norm"""

# En los gráficos de scatterplot que se realizaron antes se observan que hay dos
# hogares cuyas características resultan diferir bastante con respecto a los hogares
# con un precio similar, entonces se prodecerá a eliminarlos.
"""index_outliers = total_data.loc[(total_data["TotalFloor"] > 7000) & (total_data["SalePrice"].notna()), :].index.values
print(index_outliers)
total_data.drop(index=index_outliers, inplace=True)"""
# Obtención de variables categóricas.
dummies_cols = total_data.select_dtypes(include=object).columns
dummies = pd.get_dummies(total_data[dummies_cols]) # One Hot Encoding

# Obtención de variables numéricas.
int_cols = total_data.select_dtypes(include = ['float64','int64']).columns
int_cols = int_cols[int_cols != "SalePrice"]

# Separación entre train.csv y test.csv
data_model = pd.concat([total_data.loc[:1460, int_cols].dropna(), dummies.loc[:1460, :].dropna(), total_data["SalePrice"].dropna()], axis=1)
data_model_test = pd.concat([total_data.loc[1461:, int_cols], dummies.loc[1461:, :]], axis=1)

data_model

from sklearn.preprocessing import StandardScaler

X = data_model.iloc[:, :-1] # Variables explicativas
y = data_model["SalePrice"].apply(np.log1p) # Target

# Se realiza el escalamiento tanto para la data test y train de las variables (continuas)
# escojidas anteriormente.
"""scaler = StandardScaler()
scaler.fit(X[cols_to_norm])
X[cols_to_norm] = scaler.transform(X[cols_to_norm])

scaler.fit(data_model_test[cols_to_norm])
data_model_test[cols_to_norm] = scaler.transform(data_model_test[cols_to_norm])"""
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, make_scorer

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, abs(y_pred)))

# Transforma la función rmsle en un score de sklearn. Este score se usarán para
# medir el rendimiento de cada modelo.
score = make_scorer(rmsle, greater_is_better=False)
# El problema con el modelo de regresión lineal es que con la data en entrenamiento
# da valores negativos, por lo tanto a la hora de compararlo con el valor verdadero
# la brecha resulta muy grande.
params = {'fit_intercept': [True, False]}
lreg = LinearRegression()
gcv_lreg = GridSearchCV(lreg, param_grid=params, scoring=score, cv=5)
gcv_lreg.fit(X, y)
#-(gcv_lreg.best_score_)
gcv_lreg.best_params_
params = {"fit_intercept": [True, False], "normalize": [True, False], "alpha": [val/10 for val in range(1, 10)]}
lasso = Lasso(tol=10**(-2))
gcv_lasso = GridSearchCV(lasso, param_grid=params, scoring=score, cv=5)
gcv_lasso.fit(X, y)
#-(gcv_lasso.best_score_)
gcv_lasso.best_params_
params = {"fit_intercept": [True, False], "normalize": [True, False], "alpha": [val/10 for val in range(1, 10)]}
ridge = Ridge()
gcv_ridge = GridSearchCV(ridge, param_grid=params, scoring=score, cv=5)
gcv_ridge.fit(X, y)
#-(gcv_ridge.best_score_)
gcv_ridge.best_params_
params = {"criterion": ["mse", "mae"], "max_depth": range(2, 10, 2)}
regr = RandomForestRegressor()
gcv_rfreg = GridSearchCV(regr, param_grid=params, scoring=score, cv=5)
gcv_rfreg.fit(X, y)
#-(gcv_rfreg.best_score_)
gcv_rfreg.best_params_
import xgboost as xgb
params = {"learning_rate": [val/100 for val in range(1, 10, 2)],
          "max_depth": range(2, 5),
          "subsample": [0.2, 0.5, 0.7]}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
gcv_xgb = GridSearchCV(xgb_model, param_grid=params, scoring=score, cv=5)
gcv_xgb.fit(X, y)
#-(gcv_xgb.best_score_)
gcv_xgb.best_params_
from lightgbm import LGBMRegressor

params = {"learning_rate": [val/100 for val in range(1, 10, 2)],
          "num_leaves": [7, 15, 22, 31]}

lgbm = LGBMRegressor()
gcv_lgbm = GridSearchCV(lgbm, param_grid=params, scoring=score, cv=5)
gcv_lgbm.fit(X, y)
#gcv_lgbm.best_params_
params = {"learning_rate": [val/100 for val in range(1, 10, 2)],
          "n_estimators": [300, 600, 900]}

gbreg = GradientBoostingRegressor(random_state=42, subsample=0.7)
gcv_gbreg = GridSearchCV(gbreg, param_grid=params, scoring=score, cv=5)
gcv_gbreg.fit(X, y)
#-(gcv_gbreg.best_score_)
gcv_gbreg.best_params_
models = [
    ("lgbm", gcv_lgbm.best_estimator_),
    ("gbreg", gcv_gbreg.best_estimator_)
]

stacker = StackingRegressor(estimators=models, final_estimator=LinearRegression(normalize=True))
gcv_stack = GridSearchCV(stacker, param_grid=[{}], cv=5, scoring=score)
gcv_stack.fit(X, y)
#-(gcv_stack.best_score_)

models = [
          ("LinearRegression", gcv_lreg),
          ("Lasso", gcv_lasso),
          ("Ridge", gcv_ridge),
          ("XGBRegressor", gcv_xgb),
          ("RandomForestRegressor", gcv_rfreg),
          ("LGBMRegressor", gcv_lgbm),
          ("GradientBoostingRegressor", gcv_gbreg),
          ("StackingRegressor", gcv_stack)
]

def get_model_results(model, name):
    """
    Se usará el cross validation para ver la robustez del model por medio de su
    media y varianza, además de observar su mejor score.
    """
    df_res = pd.DataFrame(model.cv_results_)
    main_res_model = df_res.loc[df_res["rank_test_score"] == 1, ["mean_test_score", "std_test_score"]]
    split_cols = ["split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score", "split4_test_score"]
    main_res_model["mean_test_score"] = main_res_model["mean_test_score"].apply(lambda val: val * -1)
    main_res_model["best_score"] = df_res.loc[df_res["rank_test_score"] == 1, split_cols].values.max() * -1
    main_res_model["worst_score"] = df_res.loc[df_res["rank_test_score"] == 1, split_cols].values.min() * -1

    # Obtiene el tiempo promedio que se demora en entrenar el modelo.
    main_res_model["mean_fit_time"] = df_res.loc[df_res["rank_test_score"] == 1, "mean_fit_time"]

    main_res_model.index = [name]

    return main_res_model
model_res = []
for model in models:
    model_res.append(get_model_results(model[1], model[0]))

model_res = pd.concat(model_res).sort_values(by=["mean_test_score"])
model_res
import xgboost as xgb
import lightgbm as lgbm

gbreg_feat_import = pd.DataFrame(gcv_gbreg.best_estimator_.feature_importances_, index=X.columns).sort_values(by=0, ascending=False).iloc[:20, :]

fig, ax = plt.subplots(1, 3, figsize=(20, 8))

xgb.plot_importance(gcv_xgb.best_estimator_, max_num_features=20, ax=ax[0], title="XGBoost", height=0.6)
lgbm.plot_importance(gcv_lgbm.best_estimator_, max_num_features=20, ax=ax[1], title="Lightgbm", height=0.6)

sns.barplot(x=gbreg_feat_import[0], y=gbreg_feat_import.index, ax=ax[2], color="blue")
ax[2].set_title("Gradien Boosting Regressor")
ax[2].set_xlabel("Feature Importance")
ax[2].set_ylabel("Feature")

plt.tight_layout()
plt.show()
preds = gcv_stack.predict(data_model_test)
preds = np.expm1(preds)
preds
respuestas_dict = {'Id': test.index, 'SalePrice': preds}
respuestas = pd.DataFrame(respuestas_dict)
respuestas.shape
respuestas.to_csv("predicciones.csv", index=False)
