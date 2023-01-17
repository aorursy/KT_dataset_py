import numpy as np 

import pandas as pd

from os import listdir 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

import pylab 

import math

import re

import random

from tqdm import tqdm



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline



from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.utils.multiclass import type_of_target

from sklearn.metrics import accuracy_score



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import optimizers

from tensorflow.keras.models import Model,Sequential

from keras.layers import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers.core import Dense, Flatten, Dropout, Lambda



from kaggle_datasets import KaggleDatasets



import warnings

warnings.filterwarnings('ignore') 



sns.set(rc={'figure.figsize': (20, 5)})
print("Tensorflow version " + tf.__version__)



AUTO = tf.data.experimental.AUTOTUNE



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tpu

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu) 

    tf.tpu.experimental.initialize_tpu_system(tpu) 

    strategy = tf.distribute.experimental.TPUStrategy(tpu) 

else:

    strategy = tf.distribute.get_strategy()

    

print("REPLICAS: ", strategy.num_replicas_in_sync)





DATASET = '512x512-melanoma-tfrecords-70k-images'

GCS_PATH = KaggleDatasets().get_gcs_path(DATASET) 
SEED = 42

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

SIZE = [512,512]

EPOCHS = 3

LABEL_SMOOTHING = 0.05

N_SPLITS = 5

I = 1
basepath = "../input/siim-isic-melanoma-classification/"

imagesRGBpath = "../input/melanoma-classification-rgb-image/" 
print(listdir(basepath))

print(listdir(imagesRGBpath))
train_info = pd.read_csv(basepath + "train.csv")

test_info = pd.read_csv(basepath + "test.csv")
print(train_info.isnull().sum())

print(test_info.isnull().sum())
patient_id_count_train = train_info.groupby('patient_id').aggregate({'patient_id': 'count'})

patient_id_count_test = test_info.groupby('patient_id').aggregate({'patient_id': 'count'})
plt.subplot(221)

sns.distplot(patient_id_count_train, color="green", kde_kws = {'color': 'gray', 'lw':1, 'label': 'patient id count' })

plt.subplot(222)

sns.distplot(patient_id_count_test, color="green", kde_kws = {'color': 'gray', 'lw':1, 'label': 'patient id count' })

plt.subplot(223)

sns.boxplot(patient_id_count_train)

plt.subplot(224)

sns.boxplot(patient_id_count_test)
sex_count_train = train_info.groupby('sex').aggregate({'sex': 'count'}).rename(columns={'sex': 'sex_count'}).reset_index()

sex_count_test = test_info.groupby('sex').aggregate({'sex': 'count'}).rename(columns={'sex': 'sex_count'}).reset_index()
plt.subplot(121)

sns.barplot('sex', 'sex_count', data = sex_count_train)

plt.subplot(122)

sns.barplot('sex', 'sex_count', data = sex_count_test)
age_count_train = train_info.groupby('age_approx').aggregate({'age_approx': 'count'})

age_count_test = test_info.groupby('age_approx').aggregate({'age_approx': 'count'})
sns.set(rc={'figure.figsize': (20, 12)})

plt.subplot(221)

sns.distplot(train_info.age_approx, color="green", kde_kws = {'color': 'gray', 'lw':1, 'label': 'patient age count' })

plt.subplot(222)

sns.distplot(test_info.age_approx, color="green", kde_kws = {'color': 'gray', 'lw':1, 'label': 'patient age count' })

plt.subplot(223)

stats.probplot(age_count_train.age_approx, dist="norm", plot=pylab)

plt.subplot (224)

stats.probplot(age_count_test.age_approx, dist="norm", plot=pylab)

pylab.show()
anatom_count_train = train_info.groupby('anatom_site_general_challenge').aggregate(

    {'anatom_site_general_challenge': 'count'}).rename(columns={'anatom_site_general_challenge': 'anatom_site_count'}).reset_index()

anatom_count_test = test_info.groupby('anatom_site_general_challenge').aggregate(

    {'anatom_site_general_challenge': 'count'}).rename(columns={'anatom_site_general_challenge': 'anatom_site_count'}).reset_index()
sns.set(rc={'figure.figsize': (20, 5)})



plt.subplot(121)

anatom_count_train = sns.barplot('anatom_site_general_challenge', 'anatom_site_count', data = anatom_count_train)

for item in anatom_count_train.get_xticklabels():

    item.set_rotation(45)  

    

plt.subplot(122)

anatom_count_test = sns.barplot('anatom_site_general_challenge', 'anatom_site_count', data = anatom_count_test)

for item in anatom_count_test.get_xticklabels():

    item.set_rotation(45)
diagnosis_count_train = train_info.groupby('diagnosis').aggregate({'diagnosis': 'count'}).rename(

    columns={'diagnosis': 'count_diagnosis'}).reset_index()
diagnosis_count = sns.barplot('diagnosis', 'count_diagnosis', data = diagnosis_count_train)

for item in diagnosis_count.get_xticklabels():

    item.set_rotation(45)
benign_malignant_count_train = train_info.groupby('benign_malignant').aggregate({'benign_malignant': 'count'}).rename(

    columns={'benign_malignant': 'count_benign_malignant'}).reset_index()



target_count_train = train_info.groupby('target').aggregate({'target': 'count'}).rename(

    columns={'target': 'count_target'}).reset_index()
plt.subplot(121)

sns.barplot('benign_malignant', 'count_benign_malignant', data = benign_malignant_count_train)

plt.subplot(122)

sns.barplot('target', 'count_target', data = target_count_train)
# Confidence interval calculation function: 

def derf(sample, mean, std):

    age_shape = sample['age_approx'].shape[0] 

    standard_error_ofthe_mean = std / math.sqrt(age_shape)

    random_mean = random.uniform(mean-(1.96*standard_error_ofthe_mean), mean+(1.96*standard_error_ofthe_mean))

    return round(random_mean, 2) 
T_index = []

for i, t in enumerate(train_info['sex'].isnull()):

    if t == True:

        T_index.append(i)

        

# select only those values in which there are gaps

train_info_NanSEX = train_info.loc[T_index] 

train_info_NanSEX.isnull().sum()
train_info_SeAgId = train_info[['patient_id', 'sex', 'age_approx']].dropna() 



Count_env = 0

for u in train_info_NanSEX['patient_id']:

    if u in list(train_info_SeAgId['patient_id']):

        Count_env+=1

        

print(Count_env)
train_info[['sex']] = train_info['sex'].fillna('male')
# target 0 girls:

Sex_female_target0 = train_info.loc[(train_info.sex == 'female') & (train_info.target == 0)]

# target 1 girls:

Sex_female_target1 = train_info.loc[(train_info.sex == 'female') & (train_info.target == 1)]

#  target 0 guys:

Sex_male_target0 = train_info.loc[(train_info.sex == 'male') & (train_info.target == 0)] 

#  target 1 guys:

Sex_male_target1 = train_info.loc[(train_info.sex == 'male') & (train_info.target == 1)] 
print(Sex_female_target0.isnull().sum())

print(Sex_male_target0.isnull().sum())
sns.set(rc={'figure.figsize': (20, 10)})



plt.subplot (221)

sns.distplot(Sex_female_target0['age_approx'], color="green", kde_kws = {'color': 'g', 'lw':1, 'label': 'distribution Age female_target 0' })

plt.subplot (222)

sns.distplot(Sex_female_target1['age_approx'], color="r", kde_kws = {'color': 'r', 'lw':1, 'label': 'distribution Age female_target 1' })

plt.subplot (223)

sns.distplot(Sex_male_target0['age_approx'], color="blue", kde_kws = {'color': 'blue', 'lw':1, 'label': 'distribution Age male_target 0' })

plt.subplot (224)

sns.distplot(Sex_male_target1['age_approx'], color="gray", kde_kws = {'color': 'gray', 'lw':1, 'label': 'distribution Age male_target 1' })
female_target0_mean, female_target1_mean = Sex_female_target0['age_approx'].mean(), Sex_female_target1['age_approx'].mean()

male_target0_mean, male_target1_mean = Sex_male_target0['age_approx'].mean(), Sex_male_target1['age_approx'].mean()



female_target0_std, female_target1_std = Sex_female_target0['age_approx'].std(), Sex_female_target1['age_approx'].std()

male_target0_std, male_target1_std = Sex_male_target0['age_approx'].std(), Sex_male_target1['age_approx'].std()
for i in train_info.loc[(train_info['sex']=='female') & (train_info['target']==0) & 

                        (train_info['age_approx'].isnull())].index:

    train_info.at[i, 'age_approx'] = derf(Sex_female_target0, female_target0_mean, female_target0_std)

    

for i in train_info.loc[(train_info['sex']=='male') & (train_info['target']==0) & 

                        (train_info['age_approx'].isnull())].index:

    train_info.at[i, 'age_approx'] = derf(Sex_male_target0, male_target0_mean, male_target0_std)    
anatom_Sex_female_target0 = Sex_female_target0.groupby('anatom_site_general_challenge').aggregate({

    'anatom_site_general_challenge': 'count'}).rename(columns={

    'anatom_site_general_challenge': 'count_anatom'}).reset_index()



anatom_Sex_female_target1 = Sex_female_target1.groupby('anatom_site_general_challenge').aggregate({

    'anatom_site_general_challenge': 'count'}).rename(columns={

    'anatom_site_general_challenge': 'count_anatom'}).reset_index()





anatom_Sex_male_target0 = Sex_male_target0.groupby('anatom_site_general_challenge').aggregate({

    'anatom_site_general_challenge': 'count'}).rename(columns={

    'anatom_site_general_challenge': 'count_anatom'}).reset_index()



anatom_Sex_male_target1 = Sex_male_target1.groupby('anatom_site_general_challenge').aggregate({

    'anatom_site_general_challenge': 'count'}).rename(columns={

    'anatom_site_general_challenge': 'count_anatom'}).reset_index()
sns.set(rc={'figure.figsize': (22, 7)})



plt.subplot(221)

anatom_count_Sex_female_target0 = sns.barplot('anatom_site_general_challenge', 'count_anatom', data = anatom_Sex_female_target0)

for item in anatom_count_Sex_female_target0.get_xticklabels():

    item.set_rotation(45)

    

plt.subplot(222)       

anatom_count_Sex_female_target1 = sns.barplot('anatom_site_general_challenge', 'count_anatom', data = anatom_Sex_female_target1)

for item in anatom_count_Sex_female_target1.get_xticklabels():

    item.set_rotation(45)
plt.subplot(221)

anatom_count_Sex_male_target0 = sns.barplot('anatom_site_general_challenge', 'count_anatom', data = anatom_Sex_male_target0)

for item in anatom_count_Sex_male_target0.get_xticklabels():

    item.set_rotation(45)

    

plt.subplot(222)       

anatom_count_Sex_male_target1 = sns.barplot('anatom_site_general_challenge', 'count_anatom', data = anatom_Sex_male_target1)

for item in anatom_count_Sex_male_target1.get_xticklabels():

    item.set_rotation(45)
train_info[['anatom_site_general_challenge']] = train_info['anatom_site_general_challenge'].fillna('torso')

train_info.isnull().sum()
test_info[['anatom_site_general_challenge']] = test_info['anatom_site_general_challenge'].fillna('torso')

test_info.isnull().sum()
patient_gender_train = train_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])

patient_gender_test = test_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])



train_patients = pd.DataFrame(index=patient_gender_train.index.values, data=patient_gender_train.values, columns=["sex"])

test_patients = pd.DataFrame(index=patient_gender_test.index.values, data=patient_gender_test.values, columns=["sex"])



train_patients.loc[:, "num_images"] = train_info.groupby("patient_id").size() 

test_patients.loc[:, "num_images"] = test_info.groupby("patient_id").size() 



train_patients.loc[:, "min_age"] = train_info.groupby("patient_id").age_approx.min()

train_patients.loc[:, "max_age"] = train_info.groupby("patient_id").age_approx.max()

test_patients.loc[:, "min_age"] = test_info.groupby("patient_id").age_approx.min()

test_patients.loc[:, "max_age"] = test_info.groupby("patient_id").age_approx.max()



train_patients.loc[:, "age_span"] = train_patients["max_age"] - train_patients["min_age"] 

test_patients.loc[:, "age_span"] = test_patients["max_age"] - test_patients["min_age"]



train_patients.loc[:, "benign_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "benign"]

train_patients.loc[:, "malignant_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "malignant"]



train_patients["min_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.min().loc[:, "malignant"]

train_patients["max_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.max().loc[:, "malignant"]
train_patients.sort_values(by="malignant_cases", ascending=False).head()   
train_patients_age_span = train_patients.groupby('age_span').aggregate({'age_span': 'count'}).rename(columns={

    'age_span': 'count_age_span'}).reset_index()



test_patients_age_span = test_patients.groupby('age_span').aggregate({'age_span': 'count'}).rename(columns={

    'age_span': 'count_age_span'}).reset_index()
fig, ax = plt.subplots(2,2,figsize=(20,12))

sns.countplot(train_patients.sex, ax=ax[0,0], palette="Reds")

ax[0,0].set_title("Gender counts with unique patient ids in train")

sns.countplot(test_patients.sex, ax=ax[0,1], palette="Blues");

ax[0,1].set_title("Gender counts with unique patient ids in test");



train_patients_age_span_ax = sns.barplot('age_span', 'count_age_span', data = train_patients_age_span, ax=ax[1,0]);

for item in train_patients_age_span_ax.get_xticklabels():

    item.set_rotation(45)

    

train_patients_age_span_ay = sns.barplot('age_span', 'count_age_span', data = test_patients_age_span, ax=ax[1,1]);

for item in train_patients_age_span_ay.get_xticklabels():

    item.set_rotation(45)



ax[1,0].set_title("Patients age span in train")

ax[1,1].set_title("Patients age span in test")
def creation_ofstatistical_tables(tables, statistical_tables):

    for i, rgb_id in tqdm(enumerate(tables.index)):

        x = tables.loc['{}'.format(rgb_id)].values

    

        statistical_tables.loc[i, 'mean'] = x.mean()

        statistical_tables.loc[i, 'des'] = np.var(x)

        statistical_tables.loc[i, 'std'] = x.std()

        statistical_tables.loc[i, 'max'] = x.max()

        statistical_tables.loc[i, 'min'] = x.min()



        statistical_tables.loc[i, 'quan0.25'] = np.quantile(x, 0.25)

        statistical_tables.loc[i, 'quan0.5'] = np.quantile(x, 0.5)

        statistical_tables.loc[i, 'quan0.75'] = np.quantile(x, 0.75)

        

    return statistical_tables
DFrame_ISIC_in_RGB1 = pd.read_csv(imagesRGBpath + 'DFrame_ISIC_in_RGB1.csv')

DFrame_ISIC_in_RGB2 = pd.read_csv(imagesRGBpath + 'DFrame_ISIC_in_RGB2.csv')

DFrame_ISIC_in_RGB3 = pd.read_csv(imagesRGBpath + 'DFrame_ISIC_in_RGB3.csv') 

DFrame_ISIC_in_TEST = pd.read_csv(imagesRGBpath + 'DFrame_ISIC_in_TEST_RGB.csv') 



DTrain_patient_Static = pd.read_csv(imagesRGBpath + 'Train_RGB_Static.csv') 

DTest_patient_Static = pd.read_csv(imagesRGBpath + 'Test_RGB_Static.csv') 



DTrain_patient_Static = DTrain_patient_Static.rename(columns={'image_name': 'ST_image_name'})

DTest_patient_Static = DTest_patient_Static.rename(columns={'image_name': 'ST_image_name'})
RGB_Table = np.concatenate((DFrame_ISIC_in_RGB1, DFrame_ISIC_in_RGB2, DFrame_ISIC_in_RGB3), axis=1) 

columns_RGM = list(DFrame_ISIC_in_RGB1.columns) + list(DFrame_ISIC_in_RGB2.columns) + list(DFrame_ISIC_in_RGB3.columns)

RGB_Table = pd.DataFrame(data=RGB_Table, columns=columns_RGM) 
RGB_Table = pd.DataFrame.transpose(RGB_Table)

RGB_Table_Test = pd.DataFrame.transpose(DFrame_ISIC_in_TEST)



RGB_Table  = RGB_Table.rename(columns={0: 'R', 1: 'G', 2: 'B'})

RGB_Table_Test  = RGB_Table_Test.rename(columns={0: 'R', 1: 'G', 2: 'B'})
RGB_Table_stat = pd.DataFrame(index=range(RGB_Table.shape[0]), dtype=np.float64,

                       columns=['mean', 'des', 'std', 'max', 'min', 'quan0.25', 'quan0.5', 'quan0.75'])



RGB_Table_stat_TEST = pd.DataFrame(index=range(RGB_Table_Test.shape[0]), dtype=np.float64,

                       columns=['mean', 'des', 'std', 'max', 'min', 'quan0.25', 'quan0.5', 'quan0.75'])
RGB_Table_stat = creation_ofstatistical_tables(RGB_Table, RGB_Table_stat)

RGB_Table_stat_TEST = creation_ofstatistical_tables(RGB_Table_Test, RGB_Table_stat_TEST)
RGB_Table_commonEnd = np.concatenate((RGB_Table, RGB_Table_stat), axis=1) 

RGB_Table_commonEnd_test = np.concatenate((RGB_Table_Test, RGB_Table_stat_TEST), axis=1) 



RGB_Table_commonEnd = pd.DataFrame(data=RGB_Table_commonEnd, columns=list(RGB_Table.columns) + list(RGB_Table_stat.columns),

                                  index=RGB_Table.index) 



RGB_Table_commonEnd_test = pd.DataFrame(data=RGB_Table_commonEnd_test, columns=list(RGB_Table_Test.columns) + 

                                        list(RGB_Table_stat_TEST.columns), index=RGB_Table_Test.index) 



RGB_Table_commonEnd['l_image_name'] = RGB_Table_commonEnd.index.map(lambda x: str(x)[:-4]) 

RGB_Table_commonEnd_test['l_image_name'] = RGB_Table_commonEnd_test.index.map(lambda x: str(x)[:-4]) 



train_F = train_info.merge(RGB_Table_commonEnd, left_on='image_name', right_on='l_image_name')

test_F = test_info.merge(RGB_Table_commonEnd_test, left_on='image_name', right_on='l_image_name')



DTrain = np.concatenate((train_F, DTrain_patient_Static), axis=1) 

DTest = np.concatenate((test_F, DTest_patient_Static), axis=1) 



DTrain_F = pd.DataFrame(data=DTrain, columns=list(train_F.columns) + list(DTrain_patient_Static.columns))                             

DTest_F = pd.DataFrame(data=DTest, columns=list(test_F.columns) + list(DTest_patient_Static.columns))
Y = DTrain_F.target

DTrain_F = DTrain_F.drop(['image_name', 'patient_id', 'diagnosis', 'benign_malignant', 'target', 'l_image_name', 

                        'ST_image_name'], axis=1) 

DTest_F = DTest_F.drop(['image_name', 'patient_id', 'l_image_name', 'ST_image_name'], axis=1) 
Y = Y.astype(float)

type_of_target(Y)
DTrain_F_encoder = DTrain_F[['sex', 'anatom_site_general_challenge']]

DTest_F_encoder = DTest_F[['sex', 'anatom_site_general_challenge']]

DTrain_F_encoder = pd.get_dummies(DTrain_F_encoder) 

DTest_F_encoder = pd.get_dummies(DTest_F_encoder) 



DTrain_F = pd.merge(DTrain_F.reset_index(), DTrain_F_encoder.reset_index())

DTest_F = pd.merge(DTest_F.reset_index(), DTest_F_encoder.reset_index())



DTrain_F = DTrain_F.drop(['index', 'sex', 'anatom_site_general_challenge'], axis=1) 

DTest_F = DTest_F.drop(['index', 'sex', 'anatom_site_general_challenge'], axis=1) 
print(DTrain_F.shape)

print(Y.shape)

print(DTest_F.shape)
scaler_imput = Pipeline([

        ("scaler", MinMaxScaler())

    ])
DTrain_F = pd.DataFrame(scaler_imput.fit_transform(DTrain_F), columns=DTrain_F.columns).astype(float)

DTest_F = pd.DataFrame(scaler_imput.transform(DTest_F), columns=DTest_F.columns).astype(float)
DTrain_F.head(5)  
# distribution of values in the target variable

print('Y==0', '{}: {}'.format(sum(Y==0), sum(Y==0)/len(Y)))

print('Y==1', '{}: {}'.format(sum(Y==1), sum(Y==1)/len(Y)))
X_train, X_test, y_train, y_test = train_test_split(DTrain_F, Y, test_size=0.20, random_state=42)
print('{}: {}'.format(sum(y_train==0), sum(y_train==0)/len(Y)))

print('{}: {}'.format(sum(y_train==1), sum(y_train==1)/len(Y)))



print('{}: {}'.format(sum(y_test==0), sum(y_test==0)/len(Y)))

print('{}: {}'.format(sum(y_test==1), sum(y_test==1)/len(Y)))
param = {'colsample_bytree': 0.7887514489701739, 

         'learning_rate': 0.03952688683476441, 

         'max_depth': 5, 

         'min_child_weight': 3, 

         'n_estimators': 185, 

         'num_class': 2, 

         'objective': 'multi:softprob', 

         'subsample': 0.928924055966708,

         'seed': 42 }



clf_xgb = XGBClassifier(**param)  

clf_xgb.fit(X_train, y_train, verbose = True, early_stopping_rounds=10, eval_metric='merror', eval_set=[(X_test, y_test)])
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf_xgb, X_test, y_test, values_format='d', display_labels=['0', '1'])
param_dist = {'n_estimators': [385], #185

              'learning_rate': [0.03952688683476441],

              'subsample': [0.928924055966708],

              'max_depth': [6], 

              'colsample_bytree': [0.7887514489701739],

              'min_child_weight': [2],

              'num_class': [2],

              'objective': ['multi:softprob'],

              'scale_pos_weight': [50],

              'reg_lambda': [0.09893832910164219],

             }





my_model = XGBClassifier()
skfolds = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED)

    

for train_index, test_index in skfolds.split(X_train, y_train):

    print('\n{} of kfold {}'.format(I, skfolds.n_splits)) 

    

    X_train_folds = DTrain_F.T[train_index]

    y_train_folds = Y[train_index]

    

    #print(X_train_folds.shape, y_train_folds.shape)

       

    X_test_fold = DTrain_F.T[test_index]

    y_test_fold = Y[test_index]

    

    print('{}: {}'.format(sum(y_train_folds==0), sum(y_train_folds==0)/len(Y)))

    print('{}: {}'.format(sum(y_train_folds==1), sum(y_train_folds==1)/len(Y)))



    print('{}: {}'.format(sum(y_test_fold==0), sum(y_test_fold==0)/len(Y)))

    print('{}: {}'.format(sum(y_test_fold==1), sum(y_test_fold==1)/len(Y)))

    

    XGB_model = GridSearchCV(my_model, param_dist, cv=2,  scoring = 'roc_auc')

    

    XGB_model.fit(X_train_folds.T, y_train_folds)

    print (XGB_model.best_params_)

    

    y_pred = XGB_model.predict(X_test_fold.T)

    

    print('accuracy_score', accuracy_score(y_test_fold, y_pred))

    print('correct', sum(y_pred==y_test_fold)/len(y_pred))

    I += 1

best_model = XGB_model.best_estimator_

best_model
plot_confusion_matrix(best_model, X_test, y_test, values_format='d', display_labels=['0', '1'])
best_model.fit(DTrain_F, Y)
ypred2 = best_model.predict_proba(DTest_F)[:,1]
sub2 = pd.DataFrame({'image_name': test_info['image_name'],

                    'target': ypred2})

#sub2.to_csv('submission2.csv',index = False)
sub2.head()
def seed_everything(SEED):

    np.random.seed(SEED)

    tf.random.set_seed(SEED) 



seed_everything(SEED)

train_filenames = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

test_filenames = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')
train_filenames[0:4]
train_filenames,valid_filenames = train_test_split(train_filenames,test_size = 0.2,random_state = SEED)
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3) 

    image = tf.cast(image, tf.float32)/255.0 

    image = tf.reshape(image, [*SIZE, 3]) 

    return image
def count_data_items(filenames): 

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
def read_labeled_tfrecord(example):

    

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "target": tf.io.FixedLenFeature([], tf.int64),  } 

    

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32) 

    return image, label 



def read_unlabeled_tfrecord(example):

    

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string), }

    

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT) 

    image = decode_image(example['image'])

    image_name = example['image_name']

    return image, image_name



def load_dataset(filenames, labeled=True, ordered=False): 

    

    ignore_order = tf.data.Options() 

    if not ordered:

        ignore_order.experimental_deterministic = False

    

    dataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

              .with_options(ignore_order) 

              .map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO))

            

    return dataset
train_dataset = (load_dataset(train_filenames, labeled=True)

    .shuffle(SEED)

    .batch(BATCH_SIZE,drop_remainder=True)

    .repeat()

    .prefetch(AUTO))



valid_dataset = (load_dataset(valid_filenames, labeled=True)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO))
print(train_dataset)

print(valid_dataset)
def ret(a):

    return  a
with strategy.scope():

    

    model= Sequential()



    model.add(Lambda(ret, input_shape = (*SIZE, 3)))



    model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

    model.add(BatchNormalization())



    model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    model.add(BatchNormalization())



    model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

    model.add(BatchNormalization())



    model.add(Conv2D(32, (3,3), padding= 'same', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    model.add(BatchNormalization())

    model.add(Flatten())



    model.add(Dense(400, activation = 'relu'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.4))

    

    model.add(Dense(300, activation = 'relu'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.4))



    model.add(Dense(100, activation = 'softmax'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.4))



    model.add(Dense(1, activation='sigmoid'))



    model.compile(loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING), 

                  metrics = ['accuracy',tf.keras.metrics.AUC(name='auc')], optimizer = 'adam')
model.summary()
STEPS_PER_EPOCH = count_data_items(train_filenames) // BATCH_SIZE

print(STEPS_PER_EPOCH)

model_fit = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_dataset) 
num_test_images = count_data_items(test_filenames)

test_dataset = (load_dataset(test_filenames, labeled=False,ordered=True)

    .batch(BATCH_SIZE))



test_dataset_images = test_dataset.map(lambda image, image_name: image)

test_dataset_image_name = test_dataset.map(lambda image, image_name: image_name).unbatch()

test_ids = next(iter(test_dataset_image_name.batch(num_test_images))).numpy().astype('U')
test_pred = model.predict(test_dataset_images, verbose=1) 
test_pred
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(test_pred)})   
pred_df.head()
#pred_df.to_csv('pred_df.csv',index = False)
pred_df_mean = sub2.merge(pred_df, left_on='image_name', right_on='image_name')
pred_df_mean.head()
#pred_mean = pd.DataFrame.from_dict({'image_name': list(pred_df_mean.image_name), 

#                                    'target': pred_df_mean[['target_x', 'target_y']].mean(axis=1)})



pred_mean3 = pd.DataFrame.from_dict({'image_name': list(pred_df_mean.image_name), 

                                    'target': 0.3 * pred_df_mean['target_x'] + pred_df_mean['target_y']})
#pred_mean.to_csv('pred_mean.csv',index = False)



pred_mean3.to_csv('pred_mean3.csv',index = False)