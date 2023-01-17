import warnings, sys

warnings.filterwarnings("ignore")



# # Chris's RAPIDS dataset

# !cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

# !cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

# sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

# sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

# sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

# !cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('fivethirtyeight')



type_colors = sns.color_palette("hls", 16)
# Read the Dataset



data_train=pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

data_test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

target_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

target_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

# First Five rows



data_train.head()
# Shape of the training data



data_train.shape
# Check for Null



data_train.isna().sum()
# Check if id is unique



data_train.sig_id.nunique()
print('There are  {:} rows in training data.'.format(len(data_train)))
# Describe the training Dataset



data_train.describe()
def plot_fn(df, feature):



    ## Create a chart

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    ## create a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
# c-90 cell

plot_fn(data_train, 'c-90')
# c-0 cell

plot_fn(data_train, 'c-0')
# c-93 cell

plot_fn(data_train, 'c-93')
# g-90 gene

plot_fn(data_train, 'g-90')
# g-0 gene

plot_fn(data_train, 'g-0')
# g-93 gene

plot_fn(data_train, 'g-93')
cp_plot = data_train.cp_type.value_counts()

ax = cp_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Category wise Contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(cp_plot.index, cp_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
cp_dose_plot = data_train.cp_dose.value_counts()

ax = cp_dose_plot.plot(kind='bar', figsize=(10, 5),   # barh -> for Horizontal rectangles plot & bar -> Vertical rectangles plot

          title='Category wise Contribution',

          color=type_colors)

for i, (p, pr) in enumerate(zip(cp_dose_plot.index, cp_dose_plot.values)):

    

    plt.text(s=str(pr), y=pr-5, x=i, color="b",

             horizontalalignment='center', verticalalignment='top',

              size=14)

ax.set_xlabel("Group")

ax.set_ylabel("Count")

plt.xticks(rotation= 45) 

plt.show()
## Target Scored Multi-Label data

target_scored.head()
## Target non-Scored Multi-Label data

target_nonscored.head()
target_scored.sum()[1:].sort_values()
# # drop columns that have only one label for 1's in target

# target_scored =target_scored.drop(['atp-sensitive_potassium_channel_antagonist',

#                     'erbb2_inhibitor'], axis=1)
## Skewness in target_scored



target_scored.skew().sort_values()
target_nonscored.sum()[1:].sort_values()  # remove the first column id(sig_id) and sort the values
## Skewness in target_nonscored



target_nonscored.skew().sort_values()
data_train.skew().sort_values()
def sig_fn(data):

    e = np.exp(1)

    y = 1/(1+e**(-data))

    return y
## Get all numerical columns and create new dataset

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeri_train = data_train.select_dtypes(include=numerics)
#numeri_train.head()
## Apply the sigmoid function on training data

sig_data = sig_fn(numeri_train)



## Find the Skewness

#sig_data.skew()

sig_data.skew().sort_values()
sig_data.head()
# First Five rows

data_test.head()
# Shape of the training data



data_test.shape
data_test.skew().sort_values()
## Get numerical columns on test data

numeri_test = data_test.select_dtypes(include=numerics)



## Apply the sigmoid function on test data

sig_data_test = sig_fn(numeri_test)



## Find the Skewness

#sig_data_test.skew()

sig_data_test.skew().sort_values()
data_train=data_train[list(data_test)]

all_data=pd.concat((data_train, data_test))

print(data_train.shape, data_test.shape, all_data.shape)
## Apply Dummies



all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_dose'], prefix='cp_dose', dtype=float)],axis=1)

all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_time'], prefix='cp_time', dtype=float)],axis=1)

all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_type'], prefix='cp_type', dtype=float)],axis=1)

all_data = all_data.drop(['cp_dose', 'cp_time', 'cp_type'], axis=1)
## After Dummies



## Create a copy of data

full_data = all_data.copy()

all_data.head()
## Check Skewness on whole dataset (training and test)

all_data.skew().sort_values()
## Get numerical columns from whole dataset

numeri_all = all_data.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_all = sig_fn(numeri_all)



## Find the Skewness



sig_data_all.skew().sort_values()
sig_data_all.head()
def normalize_fn(data):

    upper = data.max()

    lower = data.min()

    y = (data - lower)/(upper-lower)

    return y

data_normalized = normalize_fn(sig_data_all)



data_normalized.skew().sort_values()
data_normalized.head()
# data_log = np.log(numeri_all + 1)

# data_log_normalized = normalize_fn(data_log)

# data_log_normalized.describe()





# Divide by Zero Occurs
# def sig_inf_fn(data):

#     e = np.exp(1)

#     y = 2/(1+e**(-data))

#     return y







## Apply the  function

#sig_data_al = sig_inf_fn(numeri_all)



## Find the Skewness

#sig_data_al.skew().sort_values()

## Plot the Skewness values to check the value range

plt.plot(sig_data_all.skew())
all_data = all_data.drop(['g-213', 'cp_type_ctl_vehicle', 'cp_type_trt_cp'], axis=1)

numeri_all = all_data.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_all = sig_fn(numeri_all)



## Find the Skewness



#sig_data_all.skew().sort_values()



plt.plot(sig_data_all.skew())
Xtrain=all_data[:len(data_train)]

Xtest=all_data[len(data_train):]

plt.plot(Xtrain.skew())
## Separate the data

Xtrain=full_data[:len(data_train)]

Xtest=full_data[len(data_train):]
## Get numerical columns from training dataset

numerical_train = Xtrain.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_train = sig_fn(numerical_train)



## Find the Skewness



plt.plot(sig_data_train.skew())
## Get numerical columns from test dataset

numerical_test = Xtest.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_test = sig_fn(numerical_test)



## Find the Skewness



plt.plot(sig_data_test.skew())
final_train = Xtrain.drop(['g-213', 'cp_type_ctl_vehicle', 'cp_type_trt_cp'], axis=1)

numeri_final_train = final_train.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_final_train = sig_fn(numeri_final_train)



## Find the Skewness



#sig_data_final_train.skew().sort_values()



plt.plot(sig_data_final_train.skew())

#sig_data_final_train.head()
final_test = Xtest.drop(['g-213', 'cp_type_ctl_vehicle', 'cp_type_trt_cp'], axis=1)

numeri_final_test = final_test.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_data_final_test = sig_fn(numeri_final_test)



## Find the Skewness



#sig_data_final_test.skew().sort_values()



plt.plot(sig_data_final_test.skew())

#sig_data_final_test.head()
numeri_target_score = target_scored.select_dtypes(include=numerics)



## Apply the sigmoid function

sig_target_score = sig_fn(numeri_target_score)



## Plot the Skewness



#plt.plot(sig_target_score.skew())

sig_target_score.skew()

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

import scipy

# from sklearn.linear_model import LogisticRegression

# cuml uses GPU so it is much faster than sklearn 

# import cuml



import optuna



from sklearn.metrics import log_loss, make_scorer

ftwo_scorer = make_scorer(log_loss)

import pickle





from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

# kf=StratifiedKFold(n_splits=5)

kf = KFold(n_splits=10, shuffle=True, random_state=0)



X_train = sig_data_final_train
def sampling_fn(train, target):

    Xtrain = train

    Ytrain = target

    



    oversample = SMOTE(sampling_strategy=0.1)

    undersample = RandomUnderSampler(sampling_strategy=0.5)



    steps = [('o', oversample), ('u', undersample)]

    pipeline = Pipeline(steps=steps)





    xx, yy = pipeline.fit_resample(Xtrain, Ytrain)

    

    return xx, yy
# # Select First two columns from target(not id)

# select = target_scored.iloc[:,1:2]

# #select = sig_target_score.iloc[:,1:3]



# for i in select:

#     Y_train = select[i]

# #     print(column.values)

    

#     def objective(trial):

#         C=trial.suggest_loguniform('C', 10e-10, 10)

# #         model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

#         model = cuml.linear_model.LogisticRegression(C=C)

#         score=-cross_val_score(model, X_train, Y_train, cv=kf, scoring=ftwo_scorer).mean()

#         return score

#     study=optuna.create_study()

    

#     study.optimize(objective, n_trials=20, show_progress_bar= True)

#     #print(study.best_params)

#     params=study.best_params

# #     model=LogisticRegression(C=params['C'], class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

#     model = cuml.linear_model.LogisticRegression(C=params['C'])

#     model.fit(X_train, Y_train)

    

#     pick_file_name = "model"+str(i)+".pkl"

#     with open(pick_file_name, 'wb') as file:

#         pickle.dump(model, file)
from tqdm.notebook import tqdm
# # Select First two columns from target(not id)

# # select = target_scored.iloc[:,1:207]

# select = target_scored.iloc[:,9:11]



# for i in tqdm(select):

#     Y_train = select[i]

# #     print(column.values)

#     xx, yy = sampling_fn(X_train, Y_train)

#     def objective(trial):

#         C=trial.suggest_loguniform('C', 8e-05, 10)

# #         model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

#         model = cuml.linear_model.LogisticRegression(C=C)

#         score=-cross_val_score(model, xx, yy, cv=kf, scoring=ftwo_scorer).mean()

#         return score

#     study=optuna.create_study()

    

#     study.optimize(objective, n_trials=20, show_progress_bar= True)

#     #print(study.best_params)

#     params=study.best_params

# #     model=LogisticRegression(C=params['C'], class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

#     model = cuml.linear_model.LogisticRegression(C=params['C'])

#     model.fit(xx, yy)

    

#     pick_file_name = "model"+str(i)+".pkl"

#     with open(pick_file_name, 'wb') as file:

#         pickle.dump(model, file)
# import cuml

# # Select all columns from target(not id)

# select = target_scored.iloc[:,9:11]

# MODEL_DIR = "../working/"

# #select = target_scored.iloc[:,1:3]



# submit_df = pd.DataFrame()





# preds = []

# for _col in select:

# # for _col in target_scored.columns:

# #     if _col != "sig_id":

#         pkl_filename = str(MODEL_DIR)+"model"+str(_col)+".pkl"

#         print(pkl_filename)

    

#         with open(pkl_filename, 'rb') as file:

#             pickle_model = pickle.load(file)

#             prediction = pickle_model.predict_proba(sig_data_final_test)[:,1]

#             preds.append(prediction)

        

#             #submit_df[_col]=sum(preds)/float(10)/len(preds)

            

        
# submit_df
import optuna.integration.lightgbm as lgb

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

param = {

         'objective': 'binary',

         'metric': 'binary_logloss',

         'lambda_l1': 8.013085172825831e-05,

         'lambda_l2': 0.00023033771942293548,

         'num_leaves': 169,

         'feature_fraction': 0.6865601329592624,

         'bagging_fraction': 0.9052541959018207,

         'bagging_freq': 3,

         'min_child_samples': 24,

#          "device":"gpu",

         'verbose': -1

}
from sklearn.kernel_approximation import Nystroem

kernel = Nystroem(kernel = 'rbf', n_components = 100, random_state = 0)

# X = kernel.fit_transform(X)

# x_tt = kernel.transform(x_tt)

# select = target_scored.iloc[:,1:3]



# for i in tqdm(select):

for i in tqdm(range(target_scored.shape[1])):

    if i != 0:

#         print(i)

        #     Y_train = select[i]

        Y_train = target_scored.values[:, i].astype(int)   # to convert array object 

#     print(column.values)

#         xx, yy = sampling_fn(X_train, Y_train)

        xx, yy = X_train, Y_train

        X = kernel.fit_transform(xx)

        train_x, test_x, train_y, test_y = train_test_split(X, yy, test_size=0.25)

        dtrain = lgb.Dataset(train_x, label=train_y)

        dval = lgb.Dataset(test_x, label=test_y)

    

#     model = lgb.train(

#         param, dtrain, valid_sets=[dtrain, dval], verbose_eval=100, early_stopping_rounds=10

#     )

    

#     prediction = np.rint(model.predict(test_x, num_iteration=model.best_iteration))

#     accuracy = accuracy_score(test_y, prediction)

    

    

        # Cross validation

        gbm_cross =lgb.cv(param, 

                          dtrain, 

                          nfold=5, 

                          stratified=True,

                         verbose_eval=False

                         )

        #print(gbm_cross)

        num_boost_rounds_lgb = len(gbm_cross['binary_logloss-mean'])

 

        model = lgb.train(param,

                          dtrain,

                          valid_sets=[dtrain, dval],

                          num_boost_round=num_boost_rounds_lgb,

                          verbose_eval=False

                         )



        preds = model.predict(test_x)

        pred_labels = np.rint(preds)

        accuracy = accuracy_score(test_y, pred_labels)

        





        print("  Accuracy = {}".format(accuracy))



        pick_file_name = "model"+str(i)+".pkl"

        with open(pick_file_name, 'wb') as file:

            pickle.dump(model, file)
# import cuml

# # Select all columns from target(not id)

# select = target_scored.iloc[:,1:3]

# MODEL_DIR = "../working/"

# #select = target_scored.iloc[:,1:3]



# submit_df = pd.DataFrame()





# preds = []

# for _col in select:

# # for _col in target_scored.columns:

# #     if _col != "sig_id":

#         pkl_filename = str(MODEL_DIR)+"model"+str(_col)+".pkl"

#         print(pkl_filename)

    

#         with open(pkl_filename, 'rb') as file:

#             pickle_model = pickle.load(file)

#             prediction = pickle_model.predict(sig_data_final_test)

#             preds.append(prediction)

        

# #             submit_df[_col]=sum(preds)/len(preds)

#             lgb.plot_importance(pickle_model,  max_num_features=20)

#             plt.rcParams['figure.figsize'] = [7, 7]

#             plt.show()



# submit_df.head()
# for i in tqdm(range(target_scored.shape[1])):

#     if i != 0:

#         print(i)

#         #     Y_train = select[i]

#         Y_train = target_scored.values[:, i].astype(int) 

# #   
pkl_filename = "../working/model1.pkl" 

x_tt = kernel.transform(sig_data_final_test)

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)

    prediction = pickle_model.predict(x_tt)

    

        

#             submit_df[_col]=sum(preds)/len(preds)

    lgb.plot_importance(pickle_model,  max_num_features=20)

    plt.rcParams['figure.figsize'] = [15, 14]

    plt.show()

    



# 