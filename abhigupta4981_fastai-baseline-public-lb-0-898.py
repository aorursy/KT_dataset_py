from fastai.tabular import *
PATH = Path('../input')
!ls {PATH}
df = pd.read_csv(PATH/'train.csv')

test_df = pd.read_csv(PATH/'test.csv')
features = [feature for feature in df.columns if 'var_' in feature]

len(features)
def augment_df(df):

    for feature in features:

        df[f'sq_{feature}'] = df[feature]**2

        df[f'repo_{feature}'] = df[feature].apply(lambda x: 0 if x==0 else 1/x)

    df['min'] = df[features].min(axis=1)

    df['mean'] = df[features].mean(axis=1)

    df['max'] = df[features].max(axis=1)

    df['median'] = df[features].median(axis=1)

    df['std'] = df[features].std(axis=1)

    df['var'] = df[features].var(axis=1)

    df['abs_mean'] = df[features].abs().mean(axis=1)

    df['abs_median'] = df[features].abs().median(axis=1)

    df['abs_std'] = df[features].abs().std(axis=1)

    df['skew'] = df[features].skew(axis=1)

    df['kurt'] = df[features].kurt(axis=1)

    df['sq_kurt'] = df[[f'sq_{feature}' for feature in features]].kurt(axis=1)
augment_df(df)

df.head()
augment_df(test_df)

test_df.head()
features = features + [f'sq_{feature}' for feature in features] + [f'repo_{feature}' for feature in features]

num_features = len(features)
random.seed(2)

valid_idx = random.sample(list(df.index.values), int(len(df)*0.05))

train_idx = df.drop(valid_idx).index
summary = df.iloc[train_idx].describe()
class roc(Callback):

    def on_epoch_begin(self, **kwargs):

        self.total = 0

        self.batch_count = 0

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        preds = F.softmax(last_output, dim=1)

        try: 

            roc_score = roc_auc_score(to_np(last_target), to_np(preds[:, -1]))

            self.total+=roc_score

            self.batch_count+=1

        except:

            pass

    def on_epoch_end(self, num_batch, **kwargs):

        self.metric = self.total/self.batch_count
def train_and_eval_tabular_learner(train_df, train_features, valid_idx, add_noise=False, lr=0.02, epochs=1, layers=[200, 100], ps=[0.5, 0.2], name='learner'):

    data = TabularDataBunch.from_df(path='.', df=train_df, dep_var='target', valid_idx=valid_idx,

                                    cat_names=[], cont_names=train_features, bs=bs, procs=[FillMissing, Normalize], test_df=test_df)

    learn = tabular_learner(data, layers=layers, ps=ps, metrics=[roc()])

    if add_noise:

        learn.data = None

        data = None

        noise = np.random.normal(summary[features].loc['mean'].values, summary[features].loc['std'].values, (len(train_df), num_features))/100

        train_df[features]+=noise

        data = TabularDataBunch.from_df(path='.', df=train_df, dep_var='target', valid_idx=valid_idx,

                                    cat_names=[], cont_names=train_features, bs=bs, procs=[FillMissing, Normalize], test_df=test_df)

        learn.data = data

        learn.fit_one_cycle(epochs, lr)

        train_df[features]-=noise

        noise = None

    learn.fit_one_cycle(epochs, lr)

    learn.save(name, with_opt=False)

    valid_preds, _ = learn.get_preds(ds_type=DatasetType.Valid)

    valid_probs = np.array(valid_preds[:, -1])

    valid_targets = train_df.loc[valid_idx].target.values

    valid_score = roc_auc_score(valid_targets, valid_probs)

    test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    test_probs = to_np(test_preds[:, -1])

    del(data)

    del(learn)

    gc.collect()

    return valid_score, valid_probs, test_probs
sub_features = []

valid_scores = []

valid_preds = []

preds = []

num_epochs = 10

cv_counts = len(df)//num_epochs

saved_model_prefix = 'learner'
augmented_features = ['min', 'mean', 'max', 'median', 'std', 'abs_mean', 'abs_median', 'abs_std', 'skew', 'kurt', 'sq_kurt']
bs = 2048
import gc
gc.collect()
from sklearn.metrics import roc_auc_score
for i in range(num_epochs):

    print('Training model: ', i)

    sub_features.append(random.sample(list(features), int(num_features*0.75)) + augmented_features)

    name = f'{saved_model_prefix}_{i}'

    score, valid_probs, test_probs = train_and_eval_tabular_learner(df, sub_features[-1], valid_idx,

                                                                    add_noise=True, epochs=3, lr=0.02, name=name)

    valid_scores.append(score)

    valid_preds.append(valid_probs)

    preds.append(test_probs)
import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold
target = df['target']
import time
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

oof = np.zeros(len(df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(skf.split(df.values, target.values)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / 5
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
predictions
preds1 = np.array(sum(preds)/num_epochs)

preds1
predictions.shape, preds1.shape
all_ensemble_values = [0., 0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0]
sample_ensemble_array = np.array(all_ensemble_values[0]*preds1 + (1-all_ensemble_values[0])*predictions)

sample_ensemble_array
for i in range(len(all_ensemble_values)):

    predict_array = np.array(all_ensemble_values[i]*preds1 + (1-all_ensemble_values[i])*predictions)

    sub_df = pd.DataFrame({'ID_code': test_df['ID_code'].values})

    sub_df['target'] = predict_array

    sub_df.to_csv(f'submission_{i}.csv', index=False)