import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score
APP_INPUT_LIB  = "../input/data-for-investing-type-prediction"

APP_INPUT_FILE = "investing_program_prediction_data.csv"



data = pd.read_csv(os.path.join(APP_INPUT_LIB,APP_INPUT_FILE))
display_settings = {'display.max_rows' : 20 , 'display.max_columns' : 50 , 'display.width' : 200}

for op,value in display_settings.items():

    pd.set_option(op,value)
# Print data dimensions

print("Data dimensions :")

print(data.shape)

# List of columns

print("List of columns :")

print(data.columns)
# Create feature groups

APP_TARGET_FEATURE = "InvType"

se_features = [f for f in data.columns if 'SE' in f]

ba_features = [f for f in data.columns if 'BA' in f]

pe_features = [f for f in data.columns if 'PE' in f]

ia_features = [f for f in data.columns if 'IA' in f]



#### List sample customer data - Age and Geographic location

print("Sample customer data :")

print(data[se_features].head())



#### List sample banking activity data

print("Sample banking activity data :")

print(data[ba_features].head())



#### List sample investing history data

print("Sample investing portfolio data :")

print(data[pe_features].head())



#### List sample investing activity data

print("Sample investing activity data :")

print(data[ia_features].head())
fig = plt.figure(figsize=(10,10))

sns.set(style="darkgrid")

ax = sns.countplot(x = APP_TARGET_FEATURE , data = data).set_title(APP_TARGET_FEATURE + " " + "Distribution")
selected_input_features = ba_features

corr = data[selected_input_features].corr()

fig = plt.figure(figsize=(15,15))

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
label_encoders = dict()



for f in data.columns :

    f_type = data[f].dtype.name

    if (f_type == 'object'):

        f_enc = LabelEncoder() ; data[f] = f_enc.fit_transform(data[f]) ; label_encoders[f] = f_enc
from sklearn.model_selection import RepeatedStratifiedKFold

num_folds = 3 ; num_rep = 1 ; splits = dict()

rkfs_partitioner  = RepeatedStratifiedKFold(n_splits = num_folds , n_repeats = num_rep , random_state = 1234)

split_cnt = 0

for train_idxs , test_idxs in rkfs_partitioner.split(data , data[APP_TARGET_FEATURE]):

    # print("TRAIN:", len(train_idxs), "TEST:", len(test_idxs))

    i_rep  = split_cnt // num_folds

    i_part = split_cnt - i_rep * num_folds

    # print(cnt) ; print(i_rep) ; print(i_part)

    splits['R' + str(i_rep) + 'P' + str(i_part)] = {'Train' : train_idxs , 'Test' : test_idxs}

    split_cnt += 1
performance_metrics    = dict()



for i_split in splits.keys() :

    X_train = data.loc[splits[i_split]['Train'],[i for i in data.columns if i not in [APP_TARGET_FEATURE]]]

    y_train = data.loc[splits[i_split]['Train'],APP_TARGET_FEATURE]

    i_model = RandomForestClassifier(n_estimators = 100 ,  max_features = 'sqrt' , max_depth = 15 , random_state = 0) 

    i_model.fit(X_train,y_train)

    X_test  = data.loc[splits[i_split]['Test'],[i for i in data.columns if i not in [APP_TARGET_FEATURE]]]

    y_test  = data.loc[splits[i_split]['Test'],APP_TARGET_FEATURE]

    y_test_cl_preds = i_model.predict(X_test).flatten() 

    performance_metrics[i_split] = balanced_accuracy_score(y_test,y_test_cl_preds)
print("Average Balanced accuracy is : %1.3f" % np.mean(list(performance_metrics.values())))