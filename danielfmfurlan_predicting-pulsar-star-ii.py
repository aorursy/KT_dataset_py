# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = "/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv"

stars = pd.read_csv(path)

results= ["base_TEST_all_columns0.9897150578428906",

          "base_TEST_first_4_columns0.9862164089395752",

          "base_TEST_last_4_columns0.9355167561834904",

          "base_TEST_1st_and_7th_columns0.9742193387833641",

          "base_TEST_1st_and_3th_columns0.971777916403",

          "base_TEST_first_2_columns0.9605646569128446"]

#RES = pd.read_csv("/kaggle/input/predicting-pulsar-stars-i/results.csv")

#RES.head()

print("size of our data : ", len(stars))

stars.head()
print("our data columns :\n", stars.columns)
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder



def encod(cat_feat,df):

    encoder = LabelEncoder()

    

    for each in cat_feat:

        fea = each + "_cat"

        encoded = encoder.fit_transform(df[each])

        df[fea] = encoded
stars2 = stars.copy()

cat = stars2.columns[4:8]

stars2[cat] = stars2[cat].astype(int)



stars2.head()

encod(cat,stars2)

stars2.head()
def split_data(df):

    fr = 0.1

    vsize = int(len(df)*fr)



    train = df[:-2*vsize]

    valid = df[-2*vsize:-vsize]

    test = df[:-vsize]



    for each in [train,valid,test]:

        print(f"Percentage of target values : {stars.target_class.mean():.4f}")

    return train,valid,test
train,valid,test = split_data(stars2)

train.head()
import lightgbm as lgb

from sklearn import metrics



def trainModel(feat_cols,enc_cols,train,valid,test,name):

    

    if type(feat_cols) is not list:

        all_ft = feat_cols.join(enc_cols)

    else:

        all_ft = feat_cols + enc_cols

        

    dtrain = lgb.Dataset(data=train[all_ft], label=train["target_class"])

    dvalid = lgb.Dataset(data=valid[all_ft], label=valid["target_class"])

    #dtest = lgb.Dataset(data=test[_cols], label=test["target_class"])



    param = {"num_leaves" : 64, "objectives":"binary"}

    param["metric"] = "auc"



    num_round = 500

    bst = lgb.train(param,dtrain,num_round,valid_sets=[dvalid],early_stopping_rounds = 10)

    

    valid_pred = bst.predict(valid[all_ft])

    valid_score = name + "VALID_" + str(metrics.roc_auc_score(valid["target_class"],valid_pred))

    print(f"our VALIDATION score is: {valid_score}")

    if len(test.columns) != len(train.columns):

        test_pred = bst.predict(test[feat_cols])

        test_score = name + "TEST_" +  str(metrics.roc_auc_score(test["target_class"], test_pred))

    else:

        test_pred = bst.predict(test[all_ft])

        test_score = name + "TEST_" +  str(metrics.roc_auc_score(test["target_class"], test_pred))



    print(f"our TEST score is: {test_score}")

    results.append(valid_score)

    results.append(test_score)

    return bst,valid_score,test_score
feat_cols = stars2.columns[4:7]

enc_cols = stars2.columns[9:12]

res = trainModel(feat_cols,enc_cols,train,valid,test,"labelEnc_")
stars_cat = stars.copy()

cat_ft = [stars_cat.columns[0],stars_cat.columns[7]]

print("our cat feat: ", cat_ft)

stars_cat[cat_ft] = stars_cat[cat_ft].astype(int)

stars_cat.head()
print("unique values from 1st column : ", stars_cat[cat_ft[0]].unique().sum())
#plt.figure(figsize=(15,15))

#sns.swarmplot(x=stars_cat["target_class"],y=stars_cat[cat_ft[0]])
import category_encoders as ce

def tar_enc(df,cat):

    cat_ft = cat

    train,valid,test = split_data(df)

    

    targ_enc = ce.TargetEncoder(cols=cat_ft)

    targ_enc.fit(train[cat_ft],train["target_class"])

    

    train_enc = train.join(targ_enc.transform(train[cat_ft]).add_suffix("_tgEnc"))

    valid_enc = valid.join(targ_enc.transform(valid[cat_ft]).add_suffix("_tgEnc"))

    test_enc = test.join(targ_enc.transform(test[cat_ft]).add_suffix("_tgEnc"))

    

    return train_enc,valid_enc,test_enc

train_tg,valid_tg,test_tg = tar_enc(stars_cat,cat_ft)



enc_col = [name for name in train_tg.columns[-len(cat_ft):]]

print("enc ", enc_col)

res = trainModel(cat_ft,enc_col,train_tg,valid_tg,test_tg,"tgEnc_")
print(train_tg.columns[-(len(cat_ft)):])
results

#results.to_csv(r"results.csv")
import csv

csvfile = "/kaggle/working/results.csv"



with open(csvfile, "w") as output:

    writer = csv.writer(output, lineterminator='\n')

    for val in results:

        writer.writerow([val])  
def catBoost_enc(df,cat):

    cat_ft = cat

    train,valid,test = split_data(df)

    

    catB_enc = ce.CatBoostEncoder(cols=cat_ft)

    catB_enc.fit(train[cat_ft],train["target_class"])

    

    train_enc = train.join(catB_enc.transform(train[cat_ft]).add_suffix("_catBEnc"))

    valid_enc = valid.join(catB_enc.transform(valid[cat_ft]).add_suffix("_catBEnc"))

    test_enc = test.join(catB_enc.transform(test[cat_ft]).add_suffix("_catBEnc"))

    

    return train_enc,valid_enc,test_enc
train_catB,valid_catB,test_catB = catBoost_enc(stars_cat,cat_ft)

enc_col = [name for name in train_catB.columns[-len(cat_ft):]]

res = trainModel(cat_ft,enc_col,train_catB,valid_catB,test_catB,"catBEnc_")
train_catB.head()
results
stars_all = stars.copy()

stars_all = stars_all.astype(int)

stars_all.tail()
cat_feat = stars_all.columns.drop("target_class")

cat_feat
results
stars_all2 = stars_all.copy()

stars_all2.drop([" Excess kurtosis of the integrated profile"," Skewness of the integrated profile"],axis=1, inplace=True)

stars_all2.head()
cat_feat = stars_all2.columns.drop("target_class")

cat_feat
train_allTE2,valid_allTE2,test_allTE2 = tar_enc(stars_all2,cat_feat)

enc_col = [name for name in train_allTE2.columns[-len(cat_feat):]]

print("enc_col ; ", enc_col)

res = trainModel(cat_feat,enc_col,train_allTE2,valid_allTE2,test_allTE2,"allTG2_")
results
stars_all_Cat = stars_all2.copy()

stars_all_Cat.head()
train_allCat,valid_allCat,test_allCat = catBoost_enc(stars_all_Cat,cat_feat)

enc_col = [name for name in train_allCat.columns[-len(cat_feat):]]

print("enc_col ; ", enc_col)

res = trainModel(cat_feat,enc_col,train_allCat,valid_allCat,test_allCat,"allCat_")
results
train_allTE2.tail()
train_allCat.tail()