# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import category_encoders as ce

import matplotlib as plt



from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split



import lightgbm as lab

from lightgbm import LGBMClassifier



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



X_train_f = pd.read_csv('../input/train.csv')

X_test_f = pd.read_csv('../input/test.csv')

X_test_f['loan_condition']=0 #dummyで0埋め
##addr_stateに緯度経度をjoin

latlong = pd.read_csv('../input/statelatlong.csv')

latlong= latlong.rename(columns={'State':'addr_state'})

latlong.head()

X_train_f=pd.merge(X_train_f, latlong, on='addr_state', how='left')

X_test_f=pd.merge(X_test_f, latlong, on='addr_state', how='left')



#addr_stateにGDP情報をjoin

GDP = pd.read_csv('../input/US_GDP_by_State.csv')

GDP= GDP.rename(columns={'State':'City'})

X_train_f=pd.merge(X_train_f, GDP.query('year == 2015'), on='City', how='left')

X_test_f=pd.merge(X_test_f, GDP.query('year == 2015'), on='City', how='left')



##Cityは州と重複、yearも今はいらないので捨てておく

X_train_f=X_train_f.drop(columns=['City','year'])

X_test_f=X_test_f.drop(columns=['City','year'])



# print(X_train.shape, X_test.shape)

# print(X_train.head(), X_test.head())
# #１、０へ。

initial = {'f':0, 'w':1}

X_train_f.initial_list_status = X_train_f.initial_list_status.replace(initial)

X_test_f.initial_list_status = X_test_f.initial_list_status.replace(initial)



#----------Grade--------

# print(X_train.grade.value_counts(), X_test.grade.value_counts())
#ドメイン知識でグレードを順位化 ---- rev29 Target encordingの的にした際に０埋めの人が不利になるように数字を変更 rev45 split対応

grades = {'A':6, 'B':5, 'C':4, 'D':3, 'E':2, 'F':1, 'G':0}

X_train_f.grade = X_train_f.grade.replace(grades)

X_test_f.grade=X_test_f.grade.replace(grades)



#----------sub_grade--------

# print(X_train.sub_grade.value_counts(), X_test.sub_grade.value_counts())
#ドメイン知識でサブグレードを順位化 (Target encordingの的にした時に０埋めした人が不利になるように。影響の無い事確認した rev32) rev45 split対応

sub_grades = {'A1':34, 'A2':33,'A3':32, 'A4':31, 'A5':30, 'B1':29, 'B2':28, 'B3':27, 'B4':26, 'B5':25, 'C1':24, 'C2':23,'C3':22,'C4':21,'C5':20, 'D1':19,'D2':18,

             'D3':17,'D4':16,'D5':15, 'E1':14,'E2':13,'E3':12,'E4':11,'E5':10,'F1':9,'F2':8,'F3':7,'F4':6,'F5':5,'G1':4,'G2':3,'G3':2,'G4':1,'G5':0}

X_train_f.sub_grade = X_train_f.sub_grade.replace(sub_grades)

X_test_f.sub_grade=X_test_f.sub_grade.replace(sub_grades)

# print(X_train.sub_grade.value_counts(), X_test.sub_grade.value_counts())
#----------emp_title--------

# print(X_train.emp_title.value_counts(), X_test.emp_title.value_counts())

# print(X_train.shape,X_test.shape)
X_train_f["emp_title"]=X_train_f["emp_title"].str.lower()

X_test_f["emp_title"]=X_test_f["emp_title"].str.lower()

# print(X_train_f.emp_title.value_counts())
#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)(rev45 split対応)





X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("emp_title")["emp_title"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("emp_title")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "emp_title")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "emp_title")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "emp_title")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "emp_title")

X_train_f["tag_emp_titles"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_emp_titles"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])





#初見のタイトルを中央値で埋めてみる (rev32)

X_train_f.tag_emp_titles = X_train_f.tag_emp_titles.fillna(3)

X_test_f.tag_emp_titles = X_test_f.tag_emp_titles.fillna(3)



#countを強化

X_count=pd.concat([X_train_f, X_test_f])

count_mean=X_count.groupby('emp_title').ID.count()

X_train_f['emp_title_count']= X_train_f['emp_title'].map(count_mean)

X_test_f['emp_title_count']=X_test_f['emp_title'].map(count_mean)



##ordinal encoder　ちょい修正 (rev34)

oe = ce.OrdinalEncoder(cols=['emp_title'])

X_train_f = oe.fit_transform(X_train_f)

X_test_f = oe.transform(X_test_f)

#----------emp_length--------

# print(X_train_f.shape,X_test_f.shape,X_train_w.shape,X_test_w.shape)
#ドメイン知識でemp_lengthはとりあえず順序化

emp_length = {'10+ years':10,'2 years':2, '< 1 year':0, '3 years':3, '1 year':1, '5 years':5, '4 years':4, '7 years':7, '8 years':8, '6 years':6,'9 years':9}

X_train_f.emp_length = X_train_f.emp_length.replace(emp_length)

X_test_f.emp_length = X_test_f.emp_length.replace(emp_length)





#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)(ver45)

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("emp_length")["emp_length"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("emp_length")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "emp_length")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "emp_length")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "emp_length")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "emp_length")

X_train_f["tag_emp_length"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_emp_length"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])





#初見を中央値で埋めてみる (rev32)(rev45)

X_train_f.tag_emp_length = X_train_f.tag_emp_length.fillna(3)

X_test_f.tag_emp_length = X_test_f.tag_emp_length.fillna(3)



X_train_f.head()
# print(X_train.shape, X_test.shape)

# print(X_train.loan_condition)
#----------home_ownership--------

# print(X_train.home_ownership.value_counts(), X_test.home_ownership.value_counts())
#Home_ownershipの回答は"MORTAGE","RENT","OWN","OTHER"の四択なのでNONEとANYはOTHERに放り込む ver45 split

ownerships={'NONE':'OTHER', 'ANY':'OTHER'}

X_train_f.home_ownership = X_train_f.home_ownership.replace(ownerships)

X_test_f.home_ownership = X_test_f.home_ownership.replace(ownerships)



#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化) ver45 split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("home_ownership")["home_ownership"].count().reset_index(name='home_owner_counts')

grouped_grade = X_target_f.groupby("home_ownership")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "home_ownership")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "home_ownership")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "home_ownership")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "home_ownership")

X_train_f["tag_home_owner"] = X_train_f["grade_counts"]/X_train_f["home_owner_counts"]

X_test_f["tag_home_owner"] = X_test_f["grade_counts"]/X_test_f["home_owner_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])







#初見を中央値で埋めてみる (rev32)ver45

X_train_f.tag_home_owner = X_train_f.tag_home_owner.fillna(3)

X_test_f.tag_home_owner = X_test_f.tag_home_owner.fillna(3)





##ordinal encoder　ちょい修正 (rev34)ver45



oe = ce.OrdinalEncoder(cols=['home_ownership'])

X_train_f = oe.fit_transform(X_train_f)

X_test_f = oe.transform(X_test_f)

#---------purpose--------

# print(X_train_w.purpose.value_counts(), X_test_w.purpose.value_counts())
#----purpose-----”small_business”は危ないとDemoの度に言っているので、small_business_flagを立ててみる



X_train_f['small_business_flag']= ((X_train_f['purpose']=='small_business') & (X_train_f['purpose'].shift(-1) != 'small_business').astype(int))

X_test_f['small_business_flag']= ((X_test_f['purpose']=='small_business') & (X_test_f['purpose'].shift(-1) != 'small_business').astype(int))



#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)(ver42_sub_gradeに変えてみる)ver45 gradeに戻しつつsplit

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("purpose")["purpose"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("purpose")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "purpose")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "purpose")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "purpose")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "purpose")

X_train_f["tag_purpose"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_purpose"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])



#初見を中央値で埋めてみる (rev32)(rev42)

X_train_f.tag_purpose = X_train_f.tag_purpose.fillna(3)

X_test_f.tag_purpose = X_test_f.tag_purpose.fillna(3)





#そのあとOrdinal encoderとCounting encoder (rev21)

#countを強化(rev37)rev42



X_count=pd.concat([X_train_f, X_test_f])

count_mean=X_count.groupby('purpose').ID.count()



X_train_f['purpose_count']= X_train_f['purpose'].map(count_mean)

X_test_f['purpose_count']=X_test_f['purpose'].map(count_mean)



##ordinal encoder　ちょい修正 (rev34)

oe = ce.OrdinalEncoder(cols=['purpose'])

X_train_f = oe.fit_transform(X_train_f)

X_test_f = oe.transform(X_test_f)

#---------title--------
# #-”Dept_consolidation”はいくつも書き方があるので、dept_flagを立てる。(rev21) rev45効果ないからもういいや。

# X_train['dept_flag']= (((X_train['title']=='Debt consolidation') | (X_train['title']=='Credit card refinancing'))&((X_train['title'].shift(-1)!='Debt consolidation').astype(int) | (X_train['title'].shift(-1)!='Credit card refinancing').astype(int)))

# X_test['dept_flag']= (((X_test['title']=='Debt consolidation') | (X_test['title']=='Credit card refinancing'))&((X_test['title'].shift(-1)!='Debt consolidation').astype(int) | (X_test['title'].shift(-1)!='Credit card refinancing').astype(int)))





#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)rev45 split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("title")["title"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("title")["grade"].sum().reset_index(name='grade_counts')



X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "title")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "title")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "title")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "title")

X_train_f["tag_title"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_title"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])







#初見を中央値で埋めてみる (rev32)

X_train_f.tag_title = X_train_f.tag_title.fillna(3)

X_test_f.tag_title = X_test_f.tag_title.fillna(3)



#そのあとOrdinal encoderとCounting encoder 

#countを強化(rev37)

X_count_f=pd.concat([X_train_f, X_test_f])

count_mean=X_count_f.groupby('title').ID.count()

X_train_f['title_count']= X_train_f['title'].map(count_mean)

X_test_f['title_count']=X_test_f['title'].map(count_mean)







#Rev34



oe = ce.OrdinalEncoder(cols=['title'])

X_train_f = oe.fit_transform(X_train_f)

X_test_f = oe.transform(X_test_f)





#zip_code----------

# print(X_train.zip_code.value_counts(), X_test.zip_code.value_counts())
#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)rev45 split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("zip_code")["zip_code"].count().reset_index(name='zip_counts')

grouped_grade = X_target_f.groupby("zip_code")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "zip_code")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "zip_code")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "zip_code")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "zip_code")

X_train_f["tag_zip_code"] = X_train_f["grade_counts"]/X_train_f["zip_counts"]

X_test_f["tag_zip_code"] = X_test_f["grade_counts"]/X_test_f["zip_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])







#初見を中央値で埋めてみる (rev32)

X_train_f.tag_zip_code = X_train_f.tag_zip_code.fillna(3)

X_test_f.tag_zip_code = X_test_f.tag_zip_code.fillna(3)



#末尾だけ捨てて数字として使う

X_train_f.zip_code=X_train_f.zip_code.str.strip('x')

X_test_f.zip_code=X_test_f.zip_code.str.strip('x')



##ordinal encoder　追加 (rev34)

oe = ce.OrdinalEncoder(cols=['zip_code'])

X_train_tf = oe.fit_transform(X_train_f)

X_test_tf = oe.transform(X_test_f)

X_train_f["ord_zip"] = X_train_tf["zip_code"]

X_test_f["ord_zip"] = X_test_tf["zip_code"]



# X_train.head()

# print(X_train.zip_code.value_counts(), X_test.zip_code.value_counts())
#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)ver45  split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("Latitude")["Latitude"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("Latitude")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "Latitude")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "Latitude")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "Latitude")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "Latitude")

X_train_f["tag_Latitude"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_Latitude"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])





#初見を中央値で埋めてみる (rev32)

X_train_f.tag_Latitude = X_train_f.tag_Latitude.fillna(3)

X_test_f.tag_Latitude = X_test_f.tag_Latitude.fillna(3)

##----比較的意味のあったReal State Growth %  をターゲットエンコーディング(ver 29)

#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)rev45 split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("Real State Growth %")["Real State Growth %"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("Real State Growth %")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "Real State Growth %")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "Real State Growth %")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "Real State Growth %")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "Real State Growth %")

X_train_f["tag_Growth"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_Growth"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])





#初見を中央値で埋めてみる (rev32)

X_train_f.tag_Growth = X_train_f.tag_Growth.fillna(3)

X_test_f.tag_Growth = X_test_f.tag_Growth.fillna(3)

#------addr_state-------

# print(X_train.addr_state.value_counts(), X_test.addr_state.value_counts())
#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)rev45 split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("addr_state")["addr_state"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("addr_state")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "addr_state")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "addr_state")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "addr_state")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "addr_state")

X_train_f["tag_addr_state"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_addr_state"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])



#初見を中央値で埋めてみる (rev32)

X_train_f.tag_addr_state = X_train_f.tag_addr_state.fillna(3)

X_test_f.tag_addr_state = X_test_f.tag_addr_state.fillna(3)





#カウントエンコーディング&Ordinary Encoder countを強化(rev37)

X_count=pd.concat([X_train_f, X_test_f])

count_mean=X_count.groupby('addr_state').ID.count()

X_train_f['addr_state_count']= X_train_f['addr_state'].map(count_mean)

X_test_f['addr_state_count']=X_test_f['addr_state'].map(count_mean)

X_train_f.head()







oe = ce.OrdinalEncoder(cols=['addr_state'],handle_unknown='ignore')

X_train_f = oe.fit_transform(X_train_f)

X_test_f = oe.transform(X_test_f)



# print(X_train.addr_state.value_counts(), X_test.addr_state.value_counts())
#----Initial_list-----

# print(X_train.initial_list_status.value_counts(),X_test.initial_list_status.value_counts())
#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)

X_target=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target.groupby("initial_list_status")["initial_list_status"].count().reset_index(name='cat_counts')

grouped_grade = X_target.groupby("initial_list_status")["grade"].sum().reset_index(name='grade_counts')



X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "initial_list_status")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "initial_list_status")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "initial_list_status")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "initial_list_status")

X_train_f["tag_initial_list_status"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_initial_list_status"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])



#初見を中央値で埋めてみる (rev32)

X_train_f.tag_initial_list_status = X_train_f.tag_initial_list_status.fillna(3)

X_test_f.tag_initial_list_status = X_test_f.tag_initial_list_status.fillna(3)







#-------application_type---------
#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化) ver45 split

X_target=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target.groupby("application_type")["application_type"].count().reset_index(name='cat_counts')

grouped_grade = X_target.groupby("application_type")["grade"].sum().reset_index(name='grade_counts')



X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "application_type")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "application_type")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "application_type")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "application_type")

X_train_f["tag_application_type"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_application_type"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])





#初見を中央値で埋めてみる (rev32)

X_train_f.tag_application_type = X_train_f.tag_application_type.fillna(3)

X_test_f.tag_application_type = X_test_f.tag_application_type.fillna(3)



#１、０へ。

appli = {'Individual':0, 'Joint App':1}

X_train_f.application_type = X_train_f.application_type.replace(appli)

X_test_f.application_type = X_test_f.application_type.replace(appli)

#---------'issue_d','earliest_cr_line'

#使えなそうなので削除

X_train_f=X_train_f.drop(columns=['issue_d','earliest_cr_line'])

X_test_f=X_test_f.drop(columns=['issue_d','earliest_cr_line'])

#------annual_incをログ化----　（rev16)



X_train_f.annual_inc = X_train_f.annual_inc.fillna(0)

X_test_f.annual_inc = X_test_f.annual_inc.fillna(0)

X_train_f.annual_inc = X_train_f.annual_inc.replace(0,1)

X_test_f.annual_inc = X_test_f.annual_inc.replace(0,1)

X_train_f["log_annual_inc"]= np.log(X_train_f["annual_inc"])

X_test_f["log_annual_inc"]= np.log(X_test_f["annual_inc"])


#------loan_amntをログ化----　（rev16)

X_train_f.loan_amnt = X_train_f.loan_amnt.fillna(1)

X_test_f.loan_amnt = X_test_f.loan_amnt.fillna(1)

X_train_f.loan_amnt = X_train_f.loan_amnt.replace(0,1)

X_test_f.loan_amnt = X_test_f.loan_amnt.replace(0,1)

X_train_f["log_loan_amnt"]= np.log(X_train_f["loan_amnt"])

X_test_f["log_loan_amnt"]= np.log(X_test_f["loan_amnt"])
#さらなる改善を目指す。Box-cox変換　(rev22)

rc_log = stats.boxcox(X_train_f['loan_amnt'])

rc_logt = stats.boxcox(X_test_f['loan_amnt'])

rc_data,rc_para=rc_log

rc_datat,rc_parat=rc_logt

X_train_f['box_loan_amnt'] = rc_data

X_test_f['box_loan_amnt'] = rc_datat



rc_log = stats.boxcox(X_train_f["inq_last_6mths"]+0.01)

rc_logt = stats.boxcox(X_test_f["inq_last_6mths"]+0.01)

rc_data,rc_para=rc_log

rc_datat,rc_parat=rc_logt

X_train_f["box_inq_last_6mths"] = rc_data

X_test_f["box_inq_last_6mths"] = rc_datat



#------tot_cur_balをBox変換----　（rev23)rev45 split



X_train_f.tot_cur_bal = X_train_f.tot_cur_bal.fillna(0)

X_test_f.tot_cur_bal = X_test_f.tot_cur_bal.fillna(0)

X_train_f.tot_cur_bal = X_train_f.tot_cur_bal.replace(0,0.01)

X_test_f.tot_cur_bal = X_test_f.tot_cur_bal.replace(0,0.01)

#Box-cox変換　(rev23)

rc_log2 = stats.boxcox(X_train_f["tot_cur_bal"])

rc_logt2 = stats.boxcox(X_test_f["tot_cur_bal"])

rc_data2,rc_para2=rc_log2

rc_datat2,rc_parat2=rc_logt2

X_train_f["boxed_tot_cur_bal"] = rc_data2

X_test_f["boxed_tot_cur_bal"] = rc_datat2

#------lnq_last_6mthをBox変換----　（rev22)



X_train_f.inq_last_6mths = X_train_f.inq_last_6mths.fillna(0)

X_test_f.inq_last_6mths = X_test_f.inq_last_6mths.fillna(0)



#Box変換----　（rev23)

X_train_f.dti = X_train_f.dti.fillna(0)

X_test_f.dti = X_test_f.dti.fillna(0)

X_train_f.dti = X_train_f.dti.replace(0,0.01)

X_test_f.dti = X_test_f.dti.replace(0,0.01)

X_train_f.dti = X_train_f.dti.replace(-1.0,0.01)

X_test_f.dti = X_test_f.dti.replace(-1.0,0.01)

#log

X_train_f["log_dti"] = np.log(X_train_f["dti"])

X_test_f["log_dti"] = np.log(X_test_f["dti"])
#ドメイン知識により、年収とローンの引き算とか割り算なんて効きそう？

X_train_f["inc_minus_loan"]=X_train_f["annual_inc"]-X_train_f["loan_amnt"]

X_test_f["inc_minus_loan"]=X_test_f["annual_inc"]-X_test_f["loan_amnt"]

X_train_f["inc_dev_loan"]=X_train_f["annual_inc"]/X_train_f["loan_amnt"]

X_test_f["inc_dev_loan"]=X_test_f["annual_inc"]/X_test_f["loan_amnt"]
#------log化したloan_amntと収入を使って特徴量を作る----　（rev16)

X_train_f["log_inc_minus_loan"]=X_train_f["log_annual_inc"]-X_train_f["log_loan_amnt"]

X_test_f["log_inc_minus_loan"]=X_test_f["log_annual_inc"]-X_test_f["log_loan_amnt"]

X_train_f["log_inc_dev_loan"]=X_train_f["log_annual_inc"]/X_train_f["log_loan_amnt"]

X_test_f["log_inc_dev_loan"]=X_test_f["log_annual_inc"]/X_test_f["log_loan_amnt"]
#現時点で有用な特徴量の上位組をとりあえず割ったりしてみる。（rev18) (rev19)



X_train_f.dti = X_train_f.dti.fillna(0)

X_test_f.dti = X_test_f.dti.fillna(0)



X_train_f["sg_x_dti"]=X_train_f["sub_grade"]*X_train_f["dti"]

X_test_f["sg_x_dti"]=X_test_f["sub_grade"]*X_test_f["dti"]
# installment関連の交互作用特徴量を狙う　(rev27)

X_train_f["inc_minus_inst"]=X_train_f["annual_inc"]-X_train_f["installment"]

X_test_f["inc_minus_inst"]=X_test_f["annual_inc"]-X_test_f["installment"]

X_train_f["inc_dev_inst"]=X_train_f["annual_inc"]/X_train_f["installment"]

X_test_f["inc_dev_inst"]=X_test_f["annual_inc"]/X_test_f["installment"]





#ローンと月払いの額はなんか変なことになりそうだけどやってみる（Rev27)

X_train_f["loan_minus_inst"]=X_train_f["loan_amnt"]-X_train_f["installment"]

X_test_f["loan_minus_inst"]=X_test_f["loan_amnt"]-X_test_f["installment"]



##initial_list_status = w :1

X_train_w=X_train_f[X_train_f.initial_list_status == 1]

X_train_w.head()

X_test_w=X_test_f[X_test_f.initial_list_status == 1]

X_test_w.head()



##initial_list_status = f :0

X_train_f=X_train_f[X_train_f.initial_list_status == 0]

X_train_f.head()

X_test_f=X_test_f[X_test_f.initial_list_status == 0]

X_test_f.head()



#被っているもの

X_train_f=X_train_f.drop(columns=['annual_inc','dti','loan_amnt','zip_code','box_loan_amnt',

                              'tag_application_type','box_inq_last_6mths','title','inc_dev_loan','inc_minus_loan'])



#欠損値が多すぎるもの

X_train_f=X_train_f.drop(columns=['tot_cur_bal','boxed_tot_cur_bal','tot_coll_amt',"mths_since_last_record","mths_since_last_delinq","mths_since_last_major_derog"])

#精度あがらなかったもの

X_train_f=X_train_f.drop(columns=['small_business_flag','tag_initial_list_status','title_count','inc_minus_inst',

                                  'loan_minus_inst','sg_x_dti','log_inc_minus_loan','grade','tag_title','log_inc_dev_loan','purpose_count','emp_title'])



#------------

#被っているもの

X_test_f=X_test_f.drop(columns=['annual_inc','dti','loan_amnt','zip_code','box_loan_amnt',

                              'tag_application_type','box_inq_last_6mths','title','inc_dev_loan','inc_minus_loan'])



#欠損値が多すぎるもの

X_test_f=X_test_f.drop(columns=['tot_cur_bal','boxed_tot_cur_bal','tot_coll_amt',"mths_since_last_record","mths_since_last_delinq","mths_since_last_major_derog"])

#精度あがらなかったもの

X_test_f=X_test_f.drop(columns=['small_business_flag','tag_initial_list_status','title_count','inc_minus_inst',

                                'loan_minus_inst','sg_x_dti','log_inc_minus_loan','grade','tag_title','log_inc_dev_loan','purpose_count','emp_title'])





# ------------

#被っているもの

X_train_w=X_train_w.drop(columns=['annual_inc','dti','loan_amnt','zip_code','box_loan_amnt',

                              'tag_application_type','box_inq_last_6mths','title','inc_dev_loan','inc_minus_loan'])



#欠損値が多すぎるもの

X_train_w=X_train_w.drop(columns=['boxed_tot_cur_bal','mths_since_last_delinq',

                              'mths_since_last_record','mths_since_last_major_derog'])

#精度あがらなかったもの

X_train_w=X_train_w.drop(columns=['small_business_flag','tag_initial_list_status','title_count','inc_minus_inst','loan_minus_inst'])



#------------

#被っているもの

X_test_w=X_test_w.drop(columns=['annual_inc','dti','loan_amnt','zip_code','box_loan_amnt',

                              'tag_application_type','box_inq_last_6mths','title','inc_dev_loan','inc_minus_loan'])



#欠損値が多すぎるもの

X_test_w=X_test_w.drop(columns=['boxed_tot_cur_bal','mths_since_last_delinq',

                              'mths_since_last_record','mths_since_last_major_derog'])

#精度あがらなかったもの

X_test_w=X_test_w.drop(columns=['small_business_flag','tag_initial_list_status','title_count','inc_minus_inst','loan_minus_inst'])





#---------------欠損値補完--------- （ver39 補完した方がスコア良かった）

# #欠損値は特に根拠も無くとりあえず0で埋めてみる

imp = SimpleImputer(missing_values = np.nan,

             strategy = "constant", fill_value=0)

#                strategy = "mean")

imp.fit(X_train_f)

X_train_f=pd.DataFrame(imp.transform(X_train_f),columns=X_train_f.columns.values)

# X_train_f.isnull().any(axis=0)
print(set(X_train_f.columns)-set(X_test_f.columns))

print(set(X_train_w.columns)-set(X_test_w.columns))
X_test_f=pd.DataFrame(imp.transform(X_test_f),columns=X_test_f.columns.values)

# X_test_f.isnull().any(axis=0)
imp.fit(X_train_w)



X_train_w=pd.DataFrame(imp.transform(X_train_w),columns=X_train_w.columns.values)

# X_train_w.isnull().any(axis=0)
X_test_w=pd.DataFrame(imp.transform(X_test_w),columns=X_test_w.columns.values)

# X_test_w.isnull().any(axis=0)
# print(X_train_w.head(),X_test_w.head())
print(set(X_train_f.columns)-set(X_test_f.columns))

print(set(X_train_w.columns)-set(X_test_w.columns))
##通常版のDataDrop

# y_train=X_train['loan_condition']

# X_train=X_train.drop(columns=['loan_condition'])

# X_test=X_test.drop(columns=['loan_condition'])

# print(X_train.shape,X_test.shape)



##Initial_list版のDataDrop

y_train_f=X_train_f['loan_condition']

X_train_f=X_train_f.drop(columns=['loan_condition'])

X_test_f=X_test_f.drop(columns=['loan_condition'])

print(X_train_f.shape,X_test_f.shape)



y_train_w=X_train_w['loan_condition']

X_train_w=X_train_w.drop(columns=['loan_condition'])

X_test_w=X_test_w.drop(columns=['loan_condition'])

print(X_train_w.shape,X_test_w.shape)



#initial_list版のID削除

X_train_f_id=X_train_f

X_train_f=X_train_f.drop(columns=['ID'])

X_test_f_id=X_test_f

X_test_f=X_test_f.drop(columns=['ID'])

print(X_train_f.shape,X_test_f.shape)



X_train_w_id=X_train_w

X_train_w=X_train_w.drop(columns=['ID'])

X_test_w_id=X_test_w

X_test_w=X_test_w.drop(columns=['ID'])

print(X_train_w.shape,X_test_w.shape)

print(X_train_w.head(),X_train_f.head())
# #通常のLGBM

# clf =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#         importance_type='split', learning_rate=0.1, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



# clf2 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#         importance_type='split', learning_rate=0.1, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



# clf3 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#         importance_type='split', learning_rate=0.1, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



# clf4 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#         importance_type='split', learning_rate=0.1, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



# clf5 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#         importance_type='split', learning_rate=0.1, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



# clf6 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#         importance_type='split', learning_rate=0.1, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

#予測用

clf =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

        importance_type='split', learning_rate=0.1, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=114, n_jobs=-1, num_leaves=31, objective=None,

        random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf2 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

        importance_type='split', learning_rate=0.1, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=239, n_jobs=-1, num_leaves=31, objective=None,

        random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf3 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

        importance_type='split', learning_rate=0.1, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=173, n_jobs=-1, num_leaves=31, objective=None,

        random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

clf4 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

        importance_type='split', learning_rate=0.1, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=92, n_jobs=-1, num_leaves=31, objective=None,

        random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf5 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

        importance_type='split', learning_rate=0.1, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=151, n_jobs=-1, num_leaves=31, objective=None,

        random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf6 =LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

        importance_type='split', learning_rate=0.1, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=112, n_jobs=-1, num_leaves=31, objective=None,

        random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

##initial_list_statusごとに分けた場合のSplit

X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train_f,

                                                   y_train_f,

                                                  test_size=0.20,

                                                  random_state=0)





X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_f,

                                                   y_train_f,

                                                  test_size=0.20,

                                                  random_state=42)





X_train_3, X_val_3, y_train_3, y_val_3 = train_test_split(X_train_f,

                                                   y_train_f,

                                                  test_size=0.20,

                                                  random_state=71)





X_train_4, X_val_4, y_train_4, y_val_4 = train_test_split(X_train_w,

                                                   y_train_w,

                                                  test_size=0.20,

                                                  random_state=0)





X_train_5, X_val_5, y_train_5, y_val_5= train_test_split(X_train_w,

                                                   y_train_w,

                                                  test_size=0.20,

                                                  random_state=42)





X_train_6, X_val_6, y_train_6, y_val_6 = train_test_split(X_train_w,

                                                   y_train_w,

                                                  test_size=0.20,

                                                  random_state=71)

# # ###通常

# clf.fit(X_train_1, y_train_1, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val_1,y_val_1)])

# clf2.fit(X_train_2, y_train_2, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val_2,y_val_2)])

# clf3.fit(X_train_3, y_train_3, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val_3,y_val_3)])

# clf4.fit(X_train_4, y_train_4, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val_4,y_val_4)])

# clf5.fit(X_train_5, y_train_5, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val_5,y_val_5)])

# clf6.fit(X_train_6, y_train_6, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val_6,y_val_6)])



# ###予測用

clf.fit(X_train_f, y_train_f, early_stopping_rounds=None, eval_metric='auc')

clf2.fit(X_train_f, y_train_f, early_stopping_rounds=None, eval_metric='auc')

clf3.fit(X_train_f, y_train_f, early_stopping_rounds=None, eval_metric='auc')

clf4.fit(X_train_w, y_train_w, early_stopping_rounds=None, eval_metric='auc')

clf5.fit(X_train_w, y_train_w, early_stopping_rounds=None, eval_metric='auc')

clf6.fit(X_train_w, y_train_w, early_stopping_rounds=None, eval_metric='auc')

# ### Cross validation コーナー





# cv_results = cross_val_score(clf4,

#                              X_train_w,

#                              y_train_w,

#                              cv=5,

#                              scoring='roc_auc')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())



# #######Check用

# cv_results = cross_val_score(clf,

#                              X_train_f,

#                              y_train_f,

#                              cv=5,

#                              scoring='roc_auc')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())



# cv_results = cross_val_score(clf2,

#                              X_train_f,

#                              y_train_f,

#                              cv=5,

#                              scoring='roc_auc')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())



# cv_results = cross_val_score(clf3,

#                              X_train_f,

#                              y_train_f,

#                              cv=5,

#                              scoring='roc_auc')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())

###　予測出力コーナー(Data split用) only for light GBM



y_predA=0.4*clf.predict_proba(X_test_f)+0.3*clf2.predict_proba(X_test_f)+0.3*clf3.predict_proba(X_test_f)

y_predB=0.4*clf4.predict_proba(X_test_w)+0.3*clf5.predict_proba(X_test_w)+0.3*clf6.predict_proba(X_test_w)





# X_test_f_id["loan_condition"]=y_predA[:,1]

# X_test_f_id=X_test_f_id[["ID","loan_condition"]]

# submission = pd.read_csv('../input/sample_submission.csv')

# submission=pd.merge(submission, X_test_f_id, on='ID', how='left')

# X_test_w_id["loan_condition"]=y_predB[:,1]

# X_test_w_id=X_test_w_id[["ID","loan_condition"]]

# submission=pd.merge(submission, X_test_w_id, on='ID', how='left')

# submission=submission[["ID","loan_condition","loan_condition_y"]].fillna(0)

# submission.loan_condition=submission["loan_condition"]+submission['loan_condition_y']

# submission=submission.drop(columns=["loan_condition_y"])

# # submission.to_csv('submission.csv',index=False)

# # print(submission.head(10))

# # ###　特徴量の重要度確認コーナー

# import lightgbm as lgb

# y_pred = clf.predict_proba(X_val_1)[:,1]

# score=roc_auc_score(y_val_1,y_pred)

# print(score)





# fig, ax = plt.subplots(figsize=(10,15))

# lgb.plot_importance(clf,max_num_features=50, ax=ax, importance_type='gain')

# from sklearn.naive_bayes import GaussianNB

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import StratifiedKFold





# def Stacking(model,train,y,n_fold):

#     folds=StratifiedKFold(n_splits=n_fold,random_state=1)

#     train_pred=np.empty((0,1),float)

#     for train_indices,(val_indices) in folds.split(train,y.values):

#         x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]

#         y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

#         model.fit(X=x_train,y=y_train)

#         train_pred=np.append(train_pred,model.predict_proba(x_val))

#     return train_pred







# model1 = clf



# train_pred1 =Stacking(model=model1,n_fold=3, train=X_train_f,y=y_train_f)

# model1.fit(X_train_f,y_train_f)

# test_pred1=model1.predict_proba(X_test_f)



# # train_pred1=pd.DataFrame(train_pred1)

# # test_pred1=pd.DataFrame(test_pred1)



# model2 = GaussianNB()



# train_pred2=Stacking(model=model2,n_fold=3,train=X_train_f,y=y_train_f)

# model2.fit(X_train_f,y_train_f)

# test_pred2=model2.predict_proba(X_test_f)



# # train_pred2=pd.DataFrame(train_pred2)

# # test_pred2=pd.DataFrame(test_pred2)



# X_train_f["stack_LGBM"]=train_pred1

# X_train_f["stack_Gaus"]=train_pred2

# X_test_f["stack_LGBM"]=test_pred1[:,1]

# X_test_f["stack_Gaus"]=test_pred2[:,1]



# LR = LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,

#            intercept_scaling=1, max_iter=100, multi_class='warn',

#            n_jobs=None, penalty='l2', random_state=1, solver='warn',

#            tol=0.0001, verbose=0, warm_start=False)



# pipe1 = Pipeline([['sc', StandardScaler()],

#                   ['clf', LR]])





# X_train_f
# cv_results = cross_val_score(pipe1,

#                              X_train_f,

#                              y_train_f,

#                              cv=5,

#                              scoring='roc_auc')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())

from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='warn',

          n_jobs=None, penalty='l2', random_state=1, solver='warn',

          tol=0.0001, verbose=0, warm_start=False)



pipe1 = Pipeline([['sc', StandardScaler()],

                  ['clf', clf1]])



# ####GridSearchCVコーナー



# param_grid = { 'clf__penalty':['l1','l2'],

#                         'clf__C':[0.01,0.5,1],}



# gs = GridSearchCV(estimator=pipe1,

#                   param_grid=param_grid,

#                   scoring='roc_auc',

#                   cv=5,

#                   return_train_score=False)

# gs.fit(X_train_f, y_train_f)



# # 探索した結果のベストスコアとパラメータの取得

# print('Best Score:', gs.best_score_)

# print('Best Params', gs.best_params_)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, VotingClassifier



# clf1 = LogisticRegression(penalty='l1', 

#                           C=0.001,

#                           random_state=1)



# clf2 = RandomForestClassifier(n_estimators=100,

#                               random_state=0)



# clf3 = GaussianNB()



# pipe1 = Pipeline([['sc', StandardScaler()],

#                   ['clf', clf1]])

# pipe3 = Pipeline([['sc', StandardScaler()],

#                   ['clf', clf3]])



cv_results = cross_val_score(pipe1,

                             X_train_w,

                             y_train_w,

                             cv=5,

                             scoring='roc_auc')

print(cv_results)

print(cv_results.mean(),'+-', cv_results.std())



# print('10-fold cross validation:\n')

# for clf, label in zip([pipe1, clf2, pipe3], clf_labels):

#     scores = cross_val_score(estimator=clf,

#                              X=X_train_f,

#                              y=y_train_f,

#                              cv=3,

#                              scoring='roc_auc')

#     print("ROC AUC: %0.2f (+/- %0.2f) [%s]"

#           % (scores.mean(), scores.std(), label))
# mv_clf = VotingClassifier(estimators=[('lr', pipe1), ('dt', clf2), ('gNB', pipe3)], voting='soft')

# mv_clf = mv_clf.fit(X_train_f, y_train_f)



# all_clf = [pipe1, clf2, pipe3, mv_clf]

# clf_labels = ['Logistic regression', 'RandomForestClassifier', 'GaussianNB']

# for clf, label in zip(all_clf, clf_labels):

#     scores = cross_val_score(estimator=clf,

#                              X=X_train_f,

#                              y=y_train_f,

#                              cv=3,

#                              scoring='roc_auc')

#     print("ROC AUC: %0.2f (+/- %0.2f) [%s]"

#           % (scores.mean(), scores.std(), label))
# vote_f= VotingClassifier(estimators=[('lr', pipe1), ('dt', clf2), ('gNB', pipe3)], voting='soft')

# vote_w= VotingClassifier(estimators=[('lr', pipe1), ('dt', clf2), ('gNB', pipe3)], voting='soft')
# vote_f.fit(X_train_f,y_train_f)

# y_predC=vote_f.predict_proba(X_test_f)



# vote_w.fit(X_train_w,y_train_w)

# y_predD=vote_w.predict_proba(X_test_w)
pipe1.fit(X_train_f,y_train_f)

y_predC=pipe1.predict_proba(X_test_f)



pipe1.fit(X_train_w,y_train_w)

y_predD=pipe1.predict_proba(X_test_w)
import sys

import pandas as pd



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True))
import gc



del X_count_f, X_count,X_target_f,X_target,X_train_tf,X_test_tf,X_train_w_id,X_train_w,X_train_f_id

gc.collect()
X_train_f = pd.read_csv('../input/train.csv')

X_test_f = pd.read_csv('../input/test.csv')

#y_train = df_train.iloc[:,-1]

X_test_f['loan_condition']=0 #dummyで0埋め
##addr_stateに緯度経度をjoin

latlong = pd.read_csv('../input/statelatlong.csv')

latlong= latlong.rename(columns={'State':'addr_state'})

latlong.head()

X_train_f=pd.merge(X_train_f, latlong, on='addr_state', how='left')

X_test_f=pd.merge(X_test_f, latlong, on='addr_state', how='left')



#addr_stateにGDP情報をjoin

GDP = pd.read_csv('../input/US_GDP_by_State.csv')

GDP= GDP.rename(columns={'State':'City'})

X_train_f=pd.merge(X_train_f, GDP.query('year == 2015'), on='City', how='left')

X_test_f=pd.merge(X_test_f, GDP.query('year == 2015'), on='City', how='left')



##Cityは州と重複、yearも今はいらないので捨てておく

X_train_f=X_train_f.drop(columns=['City','year'])

X_test_f=X_test_f.drop(columns=['City','year'])

# #１、０へ。

initial = {'f':0, 'w':1}

X_train_f.initial_list_status = X_train_f.initial_list_status.replace(initial)

X_test_f.initial_list_status = X_test_f.initial_list_status.replace(initial)

print(X_train_f.initial_list_status.value_counts(),X_test_f.initial_list_status.value_counts())





###分ける場合は下

# ##initial_list_status = f :0

# X_train_f=X_train_f[X_train_f.initial_list_status == 0]

# X_train_f.head()

# X_test_f=X_test_f[X_test_f.initial_list_status == 0]

# X_test_f.head()

##initial_list_status = w :1

# X_train_w=X_train[X_train.initial_list_status == 1]

# X_train_w.head()

# X_test_w=X_test[X_test.initial_list_status == 1]

# X_test_w.head()
#----------Grade--------

grades = {'A':6, 'B':5, 'C':4, 'D':3, 'E':2, 'F':1, 'G':0}

X_train_f.grade = X_train_f.grade.replace(grades)

X_test_f.grade=X_test_f.grade.replace(grades)

X_test_f.shape
#----------sub_grade--------

#ドメイン知識でサブグレードを順位化 (Target encordingの的にした時に０埋めした人が不利になるように。影響の無い事確認した rev32) rev45 split対応

sub_grades = {'A1':34, 'A2':33,'A3':32, 'A4':31, 'A5':30, 'B1':29, 'B2':28, 'B3':27, 'B4':26, 'B5':25, 'C1':24, 'C2':23,'C3':22,'C4':21,'C5':20, 'D1':19,'D2':18,

             'D3':17,'D4':16,'D5':15, 'E1':14,'E2':13,'E3':12,'E4':11,'E5':10,'F1':9,'F2':8,'F3':7,'F4':6,'F5':5,'G1':4,'G2':3,'G3':2,'G4':1,'G5':0}

X_train_f.sub_grade = X_train_f.sub_grade.replace(sub_grades)

X_test_f.sub_grade=X_test_f.sub_grade.replace(sub_grades)

# X_train_w.sub_grade = X_train_w.sub_grade.replace(sub_grades)

# X_test_w.sub_grade=X_test_w.sub_grade.replace(sub_grades)

X_train_f.shape
#----------emp_title--------

X_train_f["emp_title"]=X_train_f["emp_title"].str.lower()

X_test_f["emp_title"]=X_test_f["emp_title"].str.lower()



#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)(rev45 split対応)####File49　輸入

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("emp_title")["emp_title"].count().reset_index(name='emp_title_count')

grouped_grade = X_target_f.groupby("emp_title")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "emp_title")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "emp_title")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "emp_title")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "emp_title")

X_train_f["tag_emp_titles"] = X_train_f["grade_counts"]/X_train_f["emp_title_count"]

X_test_f["tag_emp_titles"] = X_test_f["grade_counts"]/X_test_f["emp_title_count"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])

#----------emp_length--------

emp_length = {'10+ years':10,'2 years':2, '< 1 year':0, '3 years':3, '1 year':1, '5 years':5, '4 years':4, '7 years':7, '8 years':8, '6 years':6,'9 years':9}

X_train_f.emp_length = X_train_f.emp_length.replace(emp_length)

X_test_f.emp_length = X_test_f.emp_length.replace(emp_length)





#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)(ver45)

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("emp_length")["emp_length"].count().reset_index(name='emp_length_counts')

grouped_grade = X_target_f.groupby("emp_length")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "emp_length")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "emp_length")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "emp_length")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "emp_length")

X_train_f["tag_emp_length"] = X_train_f["grade_counts"]/X_train_f["emp_length_counts"]

X_test_f["tag_emp_length"] = X_test_f["grade_counts"]/X_test_f["emp_length_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])
#----------home_ownership--------

# print(X_train.home_ownership.value_counts(), X_test.home_ownership.value_counts())

#Home_ownershipの回答は"MORTAGE","RENT","OWN","OTHER"の四択なのでNONEとANYはOTHERに放り込む ver45 split

ownerships={'NONE':'OTHER', 'ANY':'OTHER'}

X_train_f.home_ownership = X_train_f.home_ownership.replace(ownerships)

X_test_f.home_ownership = X_test_f.home_ownership.replace(ownerships)



#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化) ver45 split　####ver49

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("home_ownership")["home_ownership"].count().reset_index(name='home_owner_counts')

grouped_grade = X_target_f.groupby("home_ownership")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "home_ownership")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "home_ownership")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "home_ownership")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "home_ownership")

X_train_f["tag_home_owner"] = X_train_f["grade_counts"]/X_train_f["home_owner_counts"]

X_test_f["tag_home_owner"] = X_test_f["grade_counts"]/X_test_f["home_owner_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])





#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)(ver42_sub_gradeに変えてみる)ver45 gradeに戻しつつsplit

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("purpose")["purpose"].count().reset_index(name='purpose_counts')

grouped_grade = X_target_f.groupby("purpose")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "purpose")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "purpose")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "purpose")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "purpose")

X_train_f["tag_purpose"] = X_train_f["grade_counts"]/X_train_f["purpose_counts"]

X_test_f["tag_purpose"] = X_test_f["grade_counts"]/X_test_f["purpose_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])



#グレードを標的にしたターゲットエンコーディング(ver 29)(ver36 X_testを混ぜて強化)rev45 split

X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("zip_code")["zip_code"].count().reset_index(name='zip_counts')

grouped_grade = X_target_f.groupby("zip_code")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "zip_code")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "zip_code")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "zip_code")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "zip_code")

X_train_f["tag_zip_code"] = X_train_f["grade_counts"]/X_train_f["zip_counts"]

X_test_f["tag_zip_code"] = X_test_f["grade_counts"]/X_test_f["zip_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])





X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("Latitude")["Latitude"].count().reset_index(name='Latitude_counts')

grouped_grade = X_target_f.groupby("Latitude")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "Latitude")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "Latitude")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "Latitude")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "Latitude")

X_train_f["tag_Latitude"] = X_train_f["grade_counts"]/X_train_f["Latitude_counts"]

X_test_f["tag_Latitude"] = X_test_f["grade_counts"]/X_test_f["Latitude_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])



X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("Longitude")["Longitude"].count().reset_index(name='Longitude_counts')

grouped_grade = X_target_f.groupby("Longitude")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "Longitude")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "Longitude")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "Longitude")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "Longitude")

X_train_f["tag_Longitude"] = X_train_f["grade_counts"]/X_train_f["Longitude_counts"]

X_test_f["tag_Longitude"] = X_test_f["grade_counts"]/X_test_f["Longitude_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])



X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("addr_state")["addr_state"].count().reset_index(name='addr_counts')

grouped_grade = X_target_f.groupby("addr_state")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "addr_state")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "addr_state")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "addr_state")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "addr_state")

X_train_f["tag_addr_state"] = X_train_f["grade_counts"]/X_train_f["addr_counts"]

X_test_f["tag_addr_state"] = X_test_f["grade_counts"]/X_test_f["addr_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts"])



X_target_f=pd.concat([X_train_f, X_test_f])

grouped_cat = X_target_f.groupby("Real State Growth %")["Real State Growth %"].count().reset_index(name='cat_counts')

grouped_grade = X_target_f.groupby("Real State Growth %")["grade"].sum().reset_index(name='grade_counts')

X_train_f = X_train_f.merge(grouped_cat, how = "left", on = "Real State Growth %")

X_train_f = X_train_f.merge(grouped_grade, how = "left", on = "Real State Growth %")

X_test_f = X_test_f.merge(grouped_cat, how = "left", on = "Real State Growth %")

X_test_f  = X_test_f.merge(grouped_grade, how = "left", on = "Real State Growth %")

X_train_f["tag_Growth"] = X_train_f["grade_counts"]/X_train_f["cat_counts"]

X_test_f["tag_Growth"] = X_test_f["grade_counts"]/X_test_f["cat_counts"]

X_train_f=X_train_f.drop(columns=["grade_counts","cat_counts"])

X_test_f=X_test_f.drop(columns=["grade_counts","cat_counts"])



###特徴量を輸入 ver49

X_train_f["inc_dev_inst"]=X_train_f["annual_inc"]/X_train_f["installment"]

X_test_f["inc_dev_inst"]=X_test_f["annual_inc"]/X_test_f["installment"]



X_train_f = pd.get_dummies(X_train_f,

                       dummy_na = True,

                       columns = ["home_ownership","purpose","zip_code","addr_state","Latitude","Longitude"])

X_test_f = pd.get_dummies(X_test_f,

                       dummy_na = True,

                       columns = ["home_ownership","purpose","zip_code","addr_state","Latitude","Longitude"])

X_train_f
#consistent with columns set

Xt_X= list(set(X_test_f)-set(X_train_f))

X_test_f= X_test_f.drop(Xt_X, axis =1)



X_Xt= list(set(X_train_f)-set(X_test_f))

X_train_f=X_train_f.drop(X_Xt, axis =1)



# re-order the score data columns

X_test_f = X_test_f.reindex(X_train_f.columns.values,axis=1)

# X_test_f

# X_te = pd.DataFrame(imp.transform(Xs_exp), columns = X_ohe_columns)

#     Xs_exp_selected = Xs_exp.loc[:, X_ohe_columns[selector.support_]]
#---------------欠損値補完---------

X_train_f["emp_title"]=X_train_f["emp_title"].fillna("None")

X_test_f["emp_title"]=X_test_f["emp_title"].fillna("None")

X_train_f=X_train_f.fillna(0)

X_test_f=X_test_f.fillna(0)
nums=['loan_amnt','installment','grade','sub_grade','emp_length','annual_inc','dti','delinq_2yrs','inq_last_6mths','mths_since_last_delinq',

     'mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','collections_12_mths_ex_med','mths_since_last_major_derog',

     'acc_now_delinq','tot_coll_amt','tot_cur_bal','State & Local Spending','Gross State Product','Population (million)',

     "emp_title_count","tag_emp_titles","emp_length_counts","tag_emp_length","home_owner_counts",

          "tag_home_owner","purpose_counts","tag_purpose","zip_counts","tag_zip_code","inc_dev_inst",

     "Latitude_counts","tag_Latitude","Longitude_counts","tag_Longitude","addr_counts","tag_addr_state","tag_Growth"]





for i in nums:

    X_train_f[i]=np.log(X_train_f[i]+0.01)

    X_test_f[i]=np.log(X_test_f[i]+0.01)



#外れ値の除去

# for i in nums:

#     X_train_f=X_train_f[(abs(X_train_f[i] - np.mean(X_train_f[i]))/np.std(X_train_f[i]) <=5)].reset_index(drop=True)
##標準化

for i in nums:

    X_copy = np.copy(X_train_f[i])

    X_train_f[i]=(X_train_f[i]-X_train_f[i].mean())/ X_train_f[i].std()

    

for i in nums:

    X_copy = np.copy(X_test_f[i])

    X_test_f[i]=(X_test_f[i]-X_test_f[i].mean())/ X_test_f[i].std()

    
###効かなそうなものを捨ててみる rev49

dro_lis=["Population (million)","State & Local Spending","Gross State Product",

         "addr_counts","Latitude_counts","Longitude_counts","zip_counts"]

X_train_f=X_train_f.drop(columns=dro_lis)

X_test_f=X_test_f.drop(columns=dro_lis)
from sklearn.feature_extraction.text import TfidfVectorizer

X_train_f["emp_title"]=X_train_f["emp_title"].fillna("None")

X_test_f["emp_title"]=X_test_f["emp_title"].fillna("None")





#####落ちるので分割

##initial_list_status = w :1

X_train_w=X_train_f[X_train_f.initial_list_status == 1]

X_train_w.head()

X_test_w=X_test_f[X_test_f.initial_list_status == 1]

print(X_test_w.head())



##initial_list_status = f :0

X_train_f=X_train_f[X_train_f.initial_list_status == 0]

X_train_f.head()

X_test_f=X_test_f[X_test_f.initial_list_status == 0]

X_test_f.head()





vectorizer = TfidfVectorizer(min_df=3,norm='l2',max_features=100) 

X_train_f_vec_emp=vectorizer.fit_transform(X_train_f["emp_title"])

X_test_f_vec_emp=vectorizer.transform(X_test_f["emp_title"])

X_train_w_vec_emp=vectorizer.fit_transform(X_train_w["emp_title"])

X_test_w_vec_emp=vectorizer.transform(X_test_w["emp_title"])

X_train_f=X_train_f.drop(columns=["emp_title","issue_d","title","earliest_cr_line","application_type","Real State Growth %"])

X_test_f=X_test_f.drop(columns=["emp_title","issue_d","title","earliest_cr_line","application_type","Real State Growth %"])

X_train_w=X_train_w.drop(columns=["emp_title","issue_d","title","earliest_cr_line","application_type","Real State Growth %"])

X_test_w=X_test_w.drop(columns=["emp_title","issue_d","title","earliest_cr_line","application_type","Real State Growth %"])
# import sys

# import pandas as pd



# print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

#                    index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True))
# X_train_f=X_train_f.drop(columns=["tot_coll_amt","tot_cur_bal","mths_since_last_record","mths_since_last_delinq","mths_since_last_major_derog","collections_12_mths_ex_med"])

# X_train_w=X_train_w.drop(columns=["tot_coll_amt","tot_cur_bal","mths_since_last_record","mths_since_last_delinq","mths_since_last_major_derog","collections_12_mths_ex_med"])

# X_test_f=X_test_f.drop(columns=["tot_coll_amt","tot_cur_bal","mths_since_last_record","mths_since_last_delinq","mths_since_last_major_derog","collections_12_mths_ex_med"])
# DataDrop

X_train_f_id=X_train_f



y_train_f=X_train_f['loan_condition']

X_train_f=X_train_f.drop(columns=['loan_condition','ID'])



y_train_w=X_train_w['loan_condition']

X_train_w=X_train_w.drop(columns=['loan_condition','ID'])



X_test_f=X_test_f.drop(columns=['loan_condition','ID'])

X_test_w=X_test_w.drop(columns=['loan_condition','ID'])

print(X_train_f.shape,X_test_f.shape,y_train_f.shape, y_train_w.shape,X_test_f.shape)
print(X_train_f.shape,X_test_f.shape,y_train_f.shape,y_train_w.shape,X_train_w.shape)
from scipy.sparse import coo_matrix, hstack,vstack

X_train_f=hstack([X_train_f,X_train_f_vec_emp])

X_test_f=hstack([X_test_f,X_test_f_vec_emp])

X_train_w=hstack([X_train_w,X_train_w_vec_emp])

X_test_w=hstack([X_test_w,X_test_w_vec_emp])



#####全部使う場合

# X_train_f=vstack([X_train_f,X_train_w])

# y_train_f=hstack([y_train_f,y_train_w])
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train_f,

                                                   y_train_f.T,

                                                  test_size=0.20,

                                                  random_state=0)



X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_w,

                                                   y_train_w.T,

                                                  test_size=0.20,

                                                  random_state=0)
import sys

import pandas as pd



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True))
import gc



del ___, _70,_,

gc.collect()
from keras.layers import Input,Dense,Dropout,BatchNormalization

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping



def create_model(input_len):

    inp = Input(shape=(input_len,), sparse=True)

    x = Dense(256,activation='relu')(inp)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, output=outp)

    model.compile(loss='binary_crossentropy',optimizer='adam') 

    return model



es = EarlyStopping(monitor='val_loss',patience=3)

model_f = create_model(X_train_1.shape[1])

model_w = create_model(X_train_2.shape[1])
model_f.fit(X_train_1,y_train_1, batch_size=512, epochs=99, callbacks=[es],validation_data=(X_val_1, y_val_1))

y_pred = model_f.predict(X_val_1)

print(roc_auc_score(y_val_1, y_pred))

# print(roc_auc_score(y_val_1, y_pred))
model_w.fit(X_train_2,y_train_2, batch_size=512, epochs=99, callbacks=[es],validation_data=(X_val_2, y_val_2))

y_pred = model_w.predict(X_val_2)

print(roc_auc_score(y_val_2, y_pred))
# 予測出力データマージ版



###LGBM

# y_predA=0.4*clf.predict_proba(X_test_f)+0.3*clf2.predict_proba(X_test_f)+0.3*clf3.predict_proba(X_test_f)

# y_predB=0.4*clf4.predict_proba(X_test_w)+0.3*clf5.predict_proba(X_test_w)+0.3*clf6.predict_proba(X_test_w)

###Voting

# y_predC=my_clf.predict_proba(X_test_f)

# y_predD=my_clf.predict_proba(X_test_w)



###NN

y_pred_E=model_f.predict(X_test_f.tocsr())

y_pred_F=model_w.predict(X_test_w.tocsr())



y_pred_f=0.4*y_predA[:,1] + 0.3*y_predC[:,1] + 0.3*y_pred_E.T

y_pred_w=0.3*y_predB[:,1] + 0.3*y_predD[:,1] + 0.4*y_pred_F.T



X_test_f_id["loan_condition"]=y_pred_f.T

X_test_f_id=X_test_f_id[["ID","loan_condition"]]

X_test_w_id["loan_condition"]=y_predB[:,1]

X_test_w_id=X_test_w_id[["ID","loan_condition"]]



submission = pd.read_csv('../input/sample_submission.csv')

submission=pd.merge(submission, X_test_f_id, on='ID', how='left')

submission=pd.merge(submission, X_test_w_id, on='ID', how='left')



submission=submission[["ID","loan_condition","loan_condition_y"]].fillna(0)

submission.loan_condition=submission["loan_condition"]+submission['loan_condition_y']

submission=submission.drop(columns=["loan_condition_y"])

submission.to_csv('submission.csv',index=False)

print(submission.head(10))
