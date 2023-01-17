# import library
import pandas as pd
import numpy as np
# import dataset
df = pd.read_csv("/kaggle/input/datminbcompetition/train.csv")
test = pd.read_csv("/kaggle/input/datminbcompetition/test.csv")
# figure out the data type
df.info()
# detect missing values
df.isnull().any()
from scipy.stats import mode
df['fac_1']=df['fac_1'].fillna(df['fac_1'].mode()[0])
df['fac_2']=df['fac_2'].fillna(df['fac_2'].mode()[0])
df['fac_3']=df['fac_3'].fillna(df['fac_3'].mode()[0])
df['fac_4']=df['fac_4'].fillna(df['fac_4'].mode()[0])
df['fac_5']=df['fac_5'].fillna(df['fac_5'].mode()[0])
df['fac_6']=df['fac_6'].fillna(df['fac_6'].mode()[0])
df['fac_7']=df['fac_7'].fillna(df['fac_7'].mode()[0])
df['fac_8']=df['fac_8'].fillna(df['fac_8'].mode()[0])
df['poi_1']=df['poi_1'].fillna(df['poi_1'].mean())
df['poi_2']=df['poi_2'].fillna(df['poi_2'].mean())
df['poi_3']=df['poi_3'].fillna(df['poi_3'].mean())
df['size']=df['size'].fillna(df['size'].mean())
df['price_monthly']=df['price_monthly'].fillna(df['price_monthly'].mean())
df['room_count']=df['room_count'].fillna(df['room_count'].mean())
df.isnull().any()
# library for detecting outliers
import matplotlib.pyplot as plt
import seaborn as sns
# select features for detecting outliers
num_features = ['poi_1','poi_2','poi_3','size','price_monthly','room_count','total_call']
n = 1
plt.figure(figsize=(10,20))
for feature in num_features:
    plt.subplot(2,4,n)
    sns.boxplot(df[feature], palette="Pastel1")
    n+=1
    plt.tight_layout()
# replace string to multiclass numeric
df['gender'].replace({'campur':2,'putra':0,'putri':1}, inplace=True)
# library for splitting training-testing
from sklearn.model_selection import train_test_split
# detecting missing values in data testing
test.isnull().any()
test['fac_1']=test['fac_1'].fillna(test['fac_1'].mode()[0])
test['fac_2']=test['fac_2'].fillna(test['fac_2'].mode()[0])
test['fac_3']=test['fac_3'].fillna(test['fac_3'].mode()[0])
test['fac_4']=test['fac_4'].fillna(test['fac_4'].mode()[0])
test['fac_5']=test['fac_5'].fillna(test['fac_5'].mode()[0])
test['fac_6']=test['fac_6'].fillna(test['fac_6'].mode()[0])
test['fac_7']=test['fac_7'].fillna(test['fac_7'].mode()[0])
test['fac_8']=test['fac_8'].fillna(test['fac_8'].mode()[0])
test['poi_1']=test['poi_1'].fillna(test['poi_1'].mean())
test['poi_2']=test['poi_2'].fillna(test['poi_2'].mean())
test['poi_3']=test['poi_3'].fillna(test['poi_3'].mean())
test['size']=test['size'].fillna(test['size'].mean())
test['price_monthly']=test['price_monthly'].fillna(test['price_monthly'].mean())
test['room_count']=test['room_count'].fillna(test['room_count'].mean())
test.isnull().any()
# make training-testing for data training 
X_train, X_testcek, y_train, y_testcek = train_test_split(df[['fac_1','fac_2','fac_3',
                                                                'fac_4','fac_5','fac_6',
                                                                'fac_7','fac_8','poi_1',
                                                                'poi_2','poi_3','size',
                                                                'price_monthly','room_count','total_call']], 
                                                          df['gender'], test_size=0.25)
# define dataframe which will be predicted
X_test = test[['fac_1','fac_2','fac_3','fac_4','fac_5','fac_6','fac_7','fac_8',
               'poi_1','poi_2','poi_3','size','price_monthly','room_count','total_call']] 
# library for classification
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# generate xgboost classifier
xgb = XGBClassifier(learning_rate =0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=2,
                    gamma=0,
                    subsample=0.5,
                    colsample_bytree=0.6,
                    scale_pos_weight=1)
model = xgb.fit(X_train, y_train)
fits = xgb.predict(X_train)
predscek = xgb.predict(X_testcek)
acc_xgbfits = (fits == y_train).sum().astype(float) / len(fits)*100
acc_xgbcek = (predscek == y_testcek).sum().astype(float) / len(predscek)*100
print("XGBoost's prediction accuracy for training data is: %3.2f" % (acc_xgbfits))
print("XGBoost's prediction accuracy for testing data in sample is: %3.2f" % (acc_xgbcek))
preds = xgb.predict(X_test)
# move submission to dataframe
submission = pd.DataFrame({"id":test['id'],
                          "gender":preds})
# replace numeric multiclass to string
submission['gender'].replace({2:'campur',0:'putra',1:'putri'}, inplace=True)
# print submission
submission
# export submission to csv
submission.to_csv('submission21apr1.csv', index=False)