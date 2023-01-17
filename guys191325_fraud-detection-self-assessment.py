!pip install git+https://github.com/goolig/dsClass.git
from sklearn.metrics import roc_curve, auc # model performance metrics

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from dsClass.path_helper import *

#print(os.listdir('../input'))
data_path = get_file_path('frd_sample.csv')

df = pd.read_csv(data_path, delimiter=',')

#df = pd.read_csv('../input/frd_sample.csv', delimiter=',')

df.dataframeName = 'frd_sample.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
df.info()
# step1 - encode target variable



d = {"F":1, "G":0}

df["IS_FRAUD"].replace(d, inplace=True)



df.head()
# now lets see how bayes works on a simple example of USER_COUNTRY column



df_exp = df[["EVENT_ID", "USER_COUNTRY","IS_FRAUD"]]



df_exp.head()
df_exp_train = df_exp.query("EVENT_ID <= 2200")

df_exp_train.tail()
df_exp_test  = df_exp.query("EVENT_ID  > 2200")

df_exp_test.tail()
tot_frd = df_exp_train.loc[(df_exp_train.IS_FRAUD == 1)].shape[0] # total fraud

tot_gen = df_exp_train.shape[0] - tot_frd



print([tot_frd,tot_gen])
countries = df["USER_COUNTRY"].unique()

print(countries)
# for each country we calculate likelihood of a transaction being fraudulent



p_f_ctry = {}



for country in countries:

    df_ctry = df_exp_train.loc[(df_exp_train.USER_COUNTRY == country)]

    df_ctry_frd = df_ctry.loc[(df_ctry.IS_FRAUD == 1)]

    

    ctry_frd = df_ctry_frd.shape[0]

    ctry_gen = df_ctry.shape[0] - ctry_frd

    

    if ctry_gen == 0:

        p_f_ctry[country] = 0

    else:

        p_f_ctry[country] = 1000*np.log ( (ctry_frd/tot_frd) / (ctry_gen/tot_gen) )

    

print(p_f_ctry)
y_score = df_exp_test["USER_COUNTRY"]

y_score.replace(p_f_ctry, inplace=True)



y_score.head()
y_test = df_exp_test["IS_FRAUD"]



y_test.head()

# Compute ROC curve and ROC area

    

fpr, tpr, _ = roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)



plt.figure()

lw = 1

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
# add code here
# Let's generate more features:



# 1. USER_COUNTRY/PAYEE_COUNTRY



df_temp = df.copy()



df_temp["USER_CTRY_PAYEE_CTRY"] = df_temp["USER_COUNTRY"].str.cat(df_temp["PAYEE_COUNTRY"], sep=';')



df_temp.head()
# 2. USER_HITS



users = df_temp["USER_ID"].unique()



evt_val_dict = {}

for user in users:

    df_loop = df_temp.loc[(df_temp.USER_ID == user)]

    

    hits = 1

    for index, row in df_loop.iterrows():

        evt_val_dict[row['EVENT_ID']] = hits

        hits = hits + 1

    

df_temp["USER_HITS"] = df_temp["EVENT_ID"].apply(evt_val_dict.get)

df_temp.tail()
print(df_temp.query("USER_ID == 'user21'"))
df_temp["USER_PAYEE"] = df_temp["USER_ID"].str.cat(df_temp["PAYEE_ID"], sep=';')



df_temp.head()



# 3. USER_PAYEE_HITS



user_payee_combinations = df_temp["USER_PAYEE"].unique()



evt_val_dict = {}

for user_payee_comb in user_payee_combinations:

    df_loop = df_temp.loc[(df_temp.USER_PAYEE == user_payee_comb)]

    

    hits = 1

    for index, row in df_loop.iterrows():

        evt_val_dict[row['EVENT_ID']] = hits

        hits = hits + 1

    

df_temp["USER_PAYEE_HITS"] = df_temp["EVENT_ID"].apply(evt_val_dict.get)

df_temp.tail()
print(df_temp.query("USER_PAYEE== 'user647;payee2'"))
features = ["CHANNEL","ISP","USER_COUNTRY","PAYEE_COUNTRY","USER_CTRY_PAYEE_CTRY","USER_HITS","USER_PAYEE_HITS","IS_FRAUD","EVENT_ID"]



df2 = df_temp[features]

df2.head()
df_exp_train = df2.query("EVENT_ID <= 2200")

df_exp_test  = df2.query("EVENT_ID  > 2200")



for feature in features:

    

    if feature == "IS_FRAUD" or feature == "EVENT_ID":

        continue

    

    

    unique_values = df2[feature].unique()

    d = {}

    

    for val in unique_values:

        df_feat = df_exp_train.loc[(df_exp_train[feature] == val)]

        df_feat_frd = df_feat.loc[(df_feat.IS_FRAUD == 1)]



        feat_frd = df_feat_frd.shape[0]

        feat_gen = df_feat.shape[0] - feat_frd

        

        if feat_gen == 0:

            d[val] = 0

        else:

            d[val] = 1000*np.log ( (feat_frd/tot_frd) / (feat_gen/tot_gen) )

    

    df_exp_test[feature].replace(d, inplace=True)

    df_exp_train[feature].replace(d, inplace=True)



df_exp_test.tail()
df_exp_train = df2.query("EVENT_ID <= 2200")

df_exp_test  = df2.query("EVENT_ID  > 2200")



for feature in features:

    

    if feature == "IS_FRAUD" or feature == "EVENT_ID":

        continue

    

    

    unique_values = df2[feature].unique()

    d = {}

    

    for val in unique_values:

        df_feat = df_exp_train.loc[(df_exp_train[feature] == val)]

        df_feat_frd = df_feat.loc[(df_feat.IS_FRAUD == 1)]



        feat_frd = df_feat_frd.shape[0]

        feat_gen = df_feat.shape[0] - feat_frd

        

        feat_frd = feat_frd + 0.00001 # suggested fix

        

        if feat_gen == 0:

            d[val] = 0

        else:

            d[val] = 1000*np.log ( (feat_frd/tot_frd) / (feat_gen/tot_gen) )

    

    df_exp_test [feature].replace(d, inplace=True)

    df_exp_train[feature].replace(d, inplace=True)



df_exp_test.tail()
df_exp_train.info()
# now we aggregate the contribution of different features into a single fraud likelihood



df_exp_train["SCORE"] = 0.0

df_exp_test ["SCORE"] = 0.0



for feature in features:

    

    if feature == "IS_FRAUD" or feature == "EVENT_ID":

        continue

    

    df_exp_train["SCORE"] = df_exp_train["SCORE"] + df_exp_train[feature]

    df_exp_test ["SCORE"] = df_exp_test ["SCORE"] + df_exp_test [feature]

    

df_exp_test.head()
y = df_exp_train["IS_FRAUD"]

y_score = df_exp_train["SCORE"]



# Compute ROC curve and ROC area

    

fpr, tpr, _ = roc_curve(y, y_score)

roc_auc = auc(fpr, tpr)



plt.figure()

lw = 1

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
# Calculate simple "winner takes it all" feature importance over training data



feat_importance = {}



df = df_exp_train



for index, row in df.iterrows():

    max_abs = 0.0

    max_feat = ""

    

    for feature in features:

        if feature == "IS_FRAUD" or feature == "EVENT_ID":

            continue

        if np.absolute(row[feature]) > max_abs:

            max_abs = np.abs(row[feature])

            max_feat = feature

    

    if max_feat in feat_importance:

        feat_importance[max_feat] = feat_importance[max_feat] + 1/df.shape[0]

    else:

        feat_importance[max_feat] = 1/df.shape[0]

        

print(feat_importance)



import matplotlib.pyplot as plt



plt.bar(list(feat_importance.keys()), feat_importance.values(), color='b')

plt.xticks(rotation=90)

plt.show()