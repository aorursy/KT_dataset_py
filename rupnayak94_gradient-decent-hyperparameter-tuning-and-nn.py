import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,RidgeCV,Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
import tensorflow as tf
tf.random.set_seed(4)
np.random.seed(4)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
from keras.utils.np_utils import to_categorical
from tensorflow.keras import models, layers, optimizers, callbacks, regularizers
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pre_train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
pre_test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
pre_train.describe()
pre_train.info()
pre_test["SalePrice"]=0.0
ids=pre_test["Id"]
pre_test.drop(["Id"], axis=1, inplace=True)
pre_train.drop(["Id"], axis=1, inplace=True)
corr=pre_train.corr()
corr["SalePrice"].sort_values(ascending=False)
highcorvars1=["SalePrice", "OverallQual", "GrLivArea", "GarageArea","TotalBsmtSF"]
highcorvars2=["SalePrice", "1stFlrSF", "FullBath","TotRmsAbvGrd", "YearBuilt"]
scatter_matrix(pre_train[highcorvars1], figsize=(14,8))

scatter_matrix(pre_train[highcorvars2], figsize=(14,8))
numeric_features=[x for x in pre_train.columns if pre_train[x].dtype!="object"]
cat_features=[x for x in pre_train.columns if pre_train[x].dtype=="object"]
mediamimputer=SimpleImputer(strategy="median").fit(pre_train[numeric_features])
modeimputer=SimpleImputer(strategy="most_frequent").fit(pre_train[cat_features])

Train_imputed_num=mediamimputer.transform(pre_train[numeric_features])
Train_imputed_cat=modeimputer.transform(pre_train[cat_features])
Train_imputed=pd.concat([pd.DataFrame(Train_imputed_num, columns=numeric_features), 
                         pd.DataFrame(Train_imputed_cat, columns=cat_features)], axis=1)

Test_imputed_num=mediamimputer.transform(pre_test[numeric_features])
Test_imputed_cat=modeimputer.transform(pre_test[cat_features])
Test_imputed=pd.concat([pd.DataFrame(Test_imputed_num, columns=numeric_features), 
                         pd.DataFrame(Test_imputed_cat, columns=cat_features)], axis=1)
Train_imputed["Source"]="Train"
Test_imputed["Source"]="Test"

fulldata=pd.concat([Train_imputed, Test_imputed], axis=0)
fulldata2=pd.get_dummies(fulldata, drop_first=True)

TrainOnly=fulldata2[fulldata2["Source_Train"]==1].drop(["Source_Train"], axis=1).copy() 
TestOnly=fulldata2[fulldata2["Source_Train"]==0].drop(["Source_Train"], axis=1).copy()  

#dividing into train and test(valid) data
features=[x for x in TrainOnly.columns if x!="SalePrice"]
featureY=TrainOnly["SalePrice"]
TrainX, ValidX, TrainY, ValidY=train_test_split(TrainOnly[features], featureY, train_size=0.8, random_state=1)

TestX=TestOnly[features].copy()
TestY=TestOnly["SalePrice"].copy()
print("X shape:",TrainX.shape, "Y shape:", TrainY.shape)
print("X shape:",ValidX.shape, "Y shape:", ValidY.shape)
print("X shape:",TestX.shape, "Y shape:", TestY.shape)
STC=StandardScaler().fit(TrainX)  
TrainX_std=STC.transform(TrainX)
ValidX_std=STC.transform(ValidX)
TestX_std=STC.transform((TestX))
#LR
model_1_LR=LinearRegression(normalize=False).fit(TrainX_std, TrainY)
ValidPred=model_1_LR.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, ValidPred))
model_2_RFR=RandomForestRegressor(random_state=4).fit(TrainX_std, TrainY)
ValidPred=model_2_RFR.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, ValidPred))
#Gridsearch to fine tune for RandomForest
params={"n_estimators":range(1000,1400,50), "max_depth":range(80,100,5), 
        "min_samples_split": range(1,9,3),
        "max_leaf_nodes": range(200,400,50)}
GridRF=GridSearchCV(estimator=RandomForestRegressor(random_state=4),n_jobs=-1,
                    param_grid=params,cv=3, 
                    scoring="neg_mean_squared_error", verbose=3).fit(TrainX_std, TrainY) 
GridRF.best_params_
RFR=GridRF.best_estimator_
model_6_RFR=RFR.fit(TrainX_std, TrainY)
Valid_pred=model_6_RFR.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, Valid_pred))
#RidgeCV to find the best Alpha
model_7_RCV=RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 10.0,100.0,1],
                    store_cv_values=True, fit_intercept=1).fit(TrainX_std, TrainY)
RCV_alphas=model_7_RCV.alpha_
RCV_alphas
#Ridge Regression
model_8_Rdg=Ridge(random_state=4, alpha=100).fit(TrainX_std, TrainY)
Valid_pred=model_8_Rdg.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, Valid_pred))
#Gridsearch to fine tune for Gradient Boost
params={"n_estimators":range(100,2000,100),
        "learning_rate":[0.1,0.01,0.001], "loss":["ls", "lad", "huber", "quantile"]}
GridGD=GridSearchCV(estimator=GradientBoostingRegressor(random_state=4), 
                    param_grid=params,cv=5, n_jobs=-1,
                    scoring="neg_mean_squared_error", verbose=10).fit(TrainX_std, TrainY) 
GridGD.best_params_
GB=GridGD.best_estimator_
model_10_GB=GB.fit(TrainX_std, TrainY)
Valid_pred=model_10_GB.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, Valid_pred))
estimator_list=[("Lr", LinearRegression(normalize=False)),
            ("RFR",RandomForestRegressor(random_state=4,
                                  n_estimators=1350,max_depth=70, max_features="auto",
                                  max_leaf_nodes=250, min_samples_split=4)),
           ("RG", Ridge(random_state=4, alpha=100.0))]
meta=GB=GradientBoostingRegressor(loss="huber", n_estimators=500, random_state=4,
                                      learning_rate=0.1)
model_9_SR=StackingRegressor(estimators=estimator_list, 
                             final_estimator=meta, cv=3, passthrough=True).fit(TrainX_std, TrainY)
Valid_pred=model_9_SR.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, Valid_pred))
model_4_NN=models.Sequential()
model_4_NN.add(layers.Dense(3000, activation="relu", input_shape=(TrainX_std.shape[1],)))
model_4_NN.add(layers.Dense(3000, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e2, l2=1e3),bias_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model_4_NN.add(layers.Dropout(0.4))
model_4_NN.add(layers.Dense(2000, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02),bias_regularizer=regularizers.l2(1e6), activity_regularizer=regularizers.l1(0.01)))
model_4_NN.add(layers.Dropout(0.2))
model_4_NN.add(layers.Dense(1500, activation="relu"))
model_4_NN.add(layers.Dense(1500, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.2),bias_regularizer=regularizers.l2(1e5), activity_regularizer=regularizers.l1(0.01)))
model_4_NN.add(layers.Dropout(0.2))
model_4_NN.add(layers.Dense(1500, activation="relu"))
model_4_NN.add(layers.Dropout(0.2))
model_4_NN.add(layers.Dense(1000, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02),bias_regularizer=regularizers.l2(1e5), activity_regularizer=regularizers.l1(0.01)))
model_4_NN.add(layers.Dense(800, activation="relu"))
model_4_NN.add(layers.Dense(500, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02),bias_regularizer=regularizers.l2(1e5), activity_regularizer=regularizers.l1(0.01)))
model_4_NN.add(layers.Dropout(0.2))
model_4_NN.add(layers.Dense(400, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02),bias_regularizer=regularizers.l2(1e5), activity_regularizer=regularizers.l1(0.01)))
model_4_NN.add(layers.Dense(200, activation="relu"))
model_4_NN.add(layers.Dropout(0.2))
model_4_NN.add(layers.Dense(100, activation="relu"))
model_4_NN.add(layers.Dense(1))
Earlystp=callbacks.EarlyStopping(monitor="loss", mode="min", patience=10)
Savemod=callbacks.ModelCheckpoint(filepath="model1.h5", monitor="val_loss", save_best_only=True)
LRP=callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)
model_4_NN.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="mae",metrics=["mse"])   
model_4_NN.summary()
hist=model_4_NN.fit(TrainX_std, TrainY, epochs=200, batch_size=32, validation_data=(ValidX_std, ValidY),
            callbacks=[Earlystp,LRP, Savemod])
history=hist.history
train_loss=history["loss"]
valid_loss=history["val_loss"]

train_mae=history["mse"]
valid_mae=history["val_mse"]
n=0
epoches=range(n, len(train_loss))

#loss

plt.plot(epoches, train_loss[n:], "bo", label="Train loss")
plt.plot(epoches, valid_loss[n:], "r", label="Valid loss")
plt.legend(loc="best")
plt.title("Loss")
plt.show()
plt.plot(epoches, train_mae[n:], "bo", label="Train mse")
plt.plot(epoches, valid_mae[n:], "r", label="Valid mse")
plt.legend(loc="best")
plt.title("mae")
plt.show()

model_4_NN_L=models.load_model("model1.h5")
ValidPred=model_4_NN_L.predict(ValidX_std)
np.sqrt(mean_squared_error(ValidY, ValidPred))
Test_pred_GD=model_9_SR.predict(TestX_std).ravel()
Test_pred_NN=model_4_NN_L.predict(TestX_std).ravel()
Final_Test_pred=0.5*Test_pred_GD+0.5*Test_pred_NN
submission=pd.DataFrame({"Id":ids, "SalePrice":Final_Test_pred})
submission.to_csv("house_price_NN2.csv", index=False)