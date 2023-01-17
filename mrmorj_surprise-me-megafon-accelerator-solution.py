import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.metrics import f1_score

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import xgboost as xgb



seed = 47
train_data = pd.read_csv('../input/mf-accelerator/contest_train.csv')

test_data = pd.read_csv('../input/mf-accelerator/contest_test.csv')

sample_subm = pd.read_csv('../input/mf-accelerator/sample_subm.csv')
print('Size of training set: {} rows and {} columns'.format(*train_data.shape))

train_data.head()
train_data['TARGET'].value_counts(normalize=True)
def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

det = basic_details(train_data)

det.sort_values(by='Missing value', ascending=False)[:10]
start_drop = ['FEATURE_189', 'FEATURE_190', 'FEATURE_191', 'FEATURE_194']



train = train_data.drop(start_drop, axis=1)

test = test_data.drop(start_drop, axis=1)
cols = [c for c in train_data.columns[2:]]

print('Number of features: {}'.format(len(cols)))



print('Feature types:')

train_data[cols].dtypes.value_counts()
counts = [[], [], []]

for c in [c for c in train.columns[2:]]:

    typ = train[c].dtype

    uniq = len(np.unique(train_data[c]))

    if uniq == 1: counts[0].append(c)

    elif uniq == 2: counts[1].append(c)

    elif uniq<15: counts[2].append(c)



print('Constant features: {} Binary features: {} Categorical features: {}\n'.format(*[len(c) for c in counts]))



print('Constant features:', counts[0])

print('Binary features:', counts[1])

print('Categorical features:', counts[2])
conts_drop = ['FEATURE_3', 'FEATURE_144', 'FEATURE_249', 'FEATURE_256']

train = train.drop(conts_drop, axis=1)

test = test.drop(conts_drop, axis=1)
binary_means = [np.mean(train[c]) for c in counts[1]]

binary_names = np.array(counts[1])[np.argsort(binary_means)]

binary_means = np.sort(binary_means)



plt.title('Mean values of binary variables')



names, means = binary_names, binary_means

plt.barh(range(len(means)), means)

plt.xlabel('Mean value')

plt.yticks(range(len(means)), names)

plt.show()
count = []

for c in [c for c in train.columns[2:]]:

    typ = train[c].dtype

    uniq = len(np.unique(train[c]))

    if uniq<15 and train[c].value_counts(normalize=True).values[0]>0.94: count.append(c)



print('Almost const features:', count)
almost_const = ['FEATURE_2', 'FEATURE_5', 'FEATURE_6', 'FEATURE_31', 'FEATURE_140', 'FEATURE_156', 'FEATURE_157', 'FEATURE_159']

train = train.drop(almost_const, axis=1)

test = test.drop(almost_const, axis=1)
for c in ['FEATURE_9', 'FEATURE_10', 'FEATURE_213', 'FEATURE_214', 'FEATURE_218', 'FEATURE_219', 'FEATURE_220', 'FEATURE_257', 'FEATURE_258', 'FEATURE_259']:

    value_counts = train[c].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.title('Categorical feature {} - Cardinality {}'.format(c, len(np.unique(train[c]))))

    plt.xlabel('Feature value')

    plt.ylabel('Occurences')

    plt.bar(range(len(value_counts)), value_counts.values)

    ax.set_xticks(range(len(value_counts)))

    ax.set_xticklabels(value_counts.index, rotation='vertical')

    plt.show()
count_num = []

for c in [c for c in train.columns[2:]]:

    typ = train[c].dtype

    uniq = len(np.unique(train[c]))

    if uniq>=100: count_num.append(c)



print('Number of numerical features:', len(count_num))
from sklearn.impute import SimpleImputer



impute = SimpleImputer(strategy='most_frequent')

X_impute = impute.fit_transform(train.drop(['TARGET', 'ID'], axis=1))

X_test_impute = impute.fit_transform(test.drop(['ID'], axis=1))



X = pd.DataFrame(X_impute)

X.columns = train.drop(['TARGET', 'ID'], axis=1).columns

X.index = train.drop(['TARGET', 'ID'], axis=1).index



X_test = pd.DataFrame(X_test_impute)

X_test.columns = test.drop(['ID'], axis=1).columns

X_test.index = test.drop(['ID'], axis=1).index



y = train['TARGET']
mms = MinMaxScaler()



X[count_num] = mms.fit_transform(X[count_num])

X_test[count_num] = mms.fit_transform(X_test[count_num])
from optuna import Trial

import gc

import optuna

from sklearn.model_selection import train_test_split

import lightgbm as lgb





def objective(trial:Trial):

    

    gc.collect()

    models=[]

    validScore=0

   

    model,log = fitXGB(trial,X,y)

    

    models.append(model)

    gc.collect()

    validScore+=log

    validScore/=len(models)

    

    return validScore
def fitXGB(trial,X, y):

    

    params={

    'n_estimators':trial.suggest_int('n_estimators', 0, 1000), 

    'max_depth':trial.suggest_int('max_depth', 2, 128),

    'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.2),

    'subsample': trial.suggest_loguniform('subsample', 0.01, 1),

    'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1),

    'min_child_weight ': trial.suggest_int('min_child_weight', 1, 256),

    'reg_alpha':trial.suggest_uniform('reg_alpha', 0, 5),

    'reg_lambda':trial.suggest_uniform('reg_lambda', 0, 5),

    'random_state':seed

    }

    stkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    model = xgb.XGBClassifier(

                           n_estimators=params['n_estimators'],

                           max_depth=params['max_depth'],

                           learning_rate=params['learning_rate'],

                           random_state =params['random_state '],

                           min_child_weight =params['min_child_weight '],

                           subsample=params['subsample'],

                           colsample_bytree=params['colsample_bytree'],

                           reg_alpha=params['reg_alpha'], 

                           reg_lambda=params['reg_lambda'],

                           objective="multi:softprob")

    res=[]

    local_probs=pd.DataFrame()



    for i, (tdx, vdx) in enumerate(stkfold.split(X, y)):

        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]

        model.fit(X_train, y_train,

                 eval_set=[(X_train, y_train), (X_valid, y_valid)],

                 verbose=False)   

        preds = pd.DataFrame(model.predict(X_valid))

        

        res.append(f1_score(y_valid, preds, average='macro'))

    

    err = np.mean(res)

    print('**score :',err)

    return model, 1-err
#study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

#study.optimize(objective, timeout=60*60*2)
strfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

params={

    'n_estimators':719, 

    'num_leaves':19,

    'max_depth':59,

    'learning_rate': 0.056142869266916216,

    'subsample': 0.013006267502359514,

    'colsample_bytree': 0.027391500037335745,

    'min_data_in_leaf': 125,

    'feature_fraction': 0.8466331464702941,

    'bagging_fraction': 0.5683222789764869,

    'bagging_freq':1,

    'random_state':seed

    }

lgbm = LGBMRegressor(num_leaves=params['num_leaves'],

                    n_estimators=params['n_estimators'],

                    max_depth=params['max_depth'],

                    learning_rate=params['learning_rate'],

                    random_state=params['random_state'],

                    min_data_in_leaf=params['min_data_in_leaf'],

                    subsample=params['subsample'],

                    colsample_bytree=params['colsample_bytree'],

                    bagging_freq=params['bagging_freq'],

                    bagging_fraction=params['bagging_fraction'],

                    feature_fraction=params['feature_fraction'],

                    verbose_eval=20,

                    objective='multiclass',

                    num_class=3)
def calc(X,y,X_test, model, cv, cols, oof):

    

    if cols is None:

        cols = X.columns

    X=X[cols]

    

    res=[]

    local_probs = pd.DataFrame()

    for i, (tdx, vdx) in enumerate(cv.split(X, y)):

        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]

        model.fit(X_train, y_train,

                 eval_set=[(X_train, y_train), (X_valid, y_valid)],

                 early_stopping_rounds=30, verbose=False)   

        preds = pd.DataFrame(model.predict(X_valid)).idxmax(axis=1).values

        

        if oof==1:   

            X_test=X_test[cols]

            oof_predict = model.predict(X_test)

            local_probs[f'fold_{i+1}'] = pd.DataFrame(model.predict(X_test)).idxmax(axis=1)

        ll = f1_score(y_valid, preds, average='macro')

        print(f'{i} Fold: {ll}')

        res.append(ll)

    print(f'AVG score: {round(np.mean(res), 5)}')

    return np.mean(res), local_probs.mode(axis=1)
_, _ = calc(X, y, X_test, lgbm, strfold, None, 0)
lgbm.fit(X, y)

preds_lgbm = pd.DataFrame(lgbm.predict(X_test)).idxmax(axis=1).values



sample_subm['Predicted'] = preds_lgbm

sample_subm.to_csv('subm_lgbm.csv', index=False)
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_,X.columns)), columns=['Value','Feature'])[-50:]



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features Top 50')

plt.tight_layout()

plt.show()
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train10 = h2o.import_file("../input/mf-accelerator/contest_train.csv")

test10 = h2o.import_file("../input/mf-accelerator/contest_test.csv")



train10[train10["TARGET"] ==2, "TARGET"] = 0

x = test10.columns[1:]

y = 'TARGET'



train10[y] = train10[y].asfactor()
aml10 = H2OAutoML(max_runtime_secs = 2*60, seed = seed)

aml10.train(x=x, y=y, training_frame=train10)
lb = aml10.leaderboard

lb.head()
model_ids = list(aml10.leaderboard['model_id'].as_data_frame().iloc[:,0])

se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])

metalearner = h2o.get_model(se.metalearner()['name'])

metalearner.std_coef_plot()
pred10 = aml10.predict(test10)

pred10.head()
pred_ml = pd.read_csv('../input/mf-accelerator/sample_subm.csv')

pred_ml['1_0'] = pred10.as_data_frame()['predict'].values
train20 = h2o.import_file("../input/mf-accelerator/contest_train.csv")

test20 = h2o.import_file("../input/mf-accelerator/contest_test.csv")



train20[train20["TARGET"] ==1, "TARGET"] = 0

x = test20.columns[1:]

y = 'TARGET'



train20[y] = train20[y].asfactor()
aml20 = H2OAutoML(max_runtime_secs = 2*60, seed = seed)

aml20.train(x=x, y=y, training_frame=train20)
lb = aml20.leaderboard

lb.head()
model_ids = list(aml20.leaderboard['model_id'].as_data_frame().iloc[:,0])

se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])

metalearner = h2o.get_model(se.metalearner()['name'])

metalearner.std_coef_plot()
pred20 = aml20.predict(test20)

pred_ml['2_0'] = pred20.as_data_frame()['predict'].values
pred_ml['Predicted'] = pred_ml['1_0'] + pred_ml['2_0']

pred_ml["Predicted"].replace({3:2}, inplace=True)

pred_ml = pred_ml[['ID', 'Predicted']]

pred_ml.to_csv('subm_automl.csv', index=False)