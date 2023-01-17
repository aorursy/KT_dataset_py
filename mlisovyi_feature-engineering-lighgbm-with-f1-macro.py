import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])
def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('bed_density', 'bedrooms', 'rooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                 ('male_over_female', 'r4h3', 'r4m3'),
                 ('man12plus_over_women12plus', 'r4h2', 'r4m2'),
                 ('pesioner_over_working', 'hogar_mayor', 'hogar_adul'),
                 ('children_over_working', 'hogar_nin', 'hogar_adul'),
                 ('education_fraction', 'escolari', 'age')
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('non_bedrooms', 'rooms', 'bedrooms'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean', 'count'],
                'escolari': ['min', 'max', 'mean', 'std'],
                'fe_education_fraction': ['min', 'max', 'mean', 'std']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df
def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_cat = tmp_df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df
train = pd.read_csv('../input/train.csv', nrows=None)
test = pd.read_csv('../input/test.csv')
train.info()
def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(-1)
    # do feature engineering and drop useless columns
    return do_features(df_)

train = process_df(train)
test = process_df(test)
train.info()
def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_
train, test = train_test_apply_func(train, test, convert_OHE2LE)
train.info()
cols_2_ohe = ['eviv_LE', 'etecho_LE', 'epared_LE', 'elimbasu_LE', 
              'energcocinar_LE', 'sanitario_LE', 'manual_elec_LE',
              'pared_LE']
cols_nums = ['age', 'meaneduc', 'dependency', 
             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',
             'bedrooms', 'overcrowding']

def convert_geo2aggs(df_):
    tmp_df = pd.concat([df_[(['lugar_LE', 'idhogar']+cols_nums)],
                        pd.get_dummies(df_[cols_2_ohe], 
                                       columns=cols_2_ohe)],axis=1)
    geo_agg = tmp_df.groupby(['lugar_LE','idhogar']).mean().groupby('lugar_LE').mean().astype(np.float32)
    geo_agg.columns = pd.Index(['geo_' + e + '_MEAN' for e in geo_agg.columns.tolist()])
    
    del tmp_df
    return df_.join(geo_agg, how='left', on='lugar_LE')

train, test = train_test_apply_func(train, test, convert_geo2aggs)
train.info()
X = train.query('parentesco1==1')
#X = train

# pull out the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)
cols_2_drop = ['abastagua_LE', 'agg18_estadocivil1_MEAN', 'agg18_instlevel6_MEAN', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_MEAN', 'agg18_parentesco12_MEAN', 'agg18_parentesco4_MEAN', 'agg18_parentesco5_MEAN', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_MEAN', 'agg18_parentesco8_MEAN', 'agg18_parentesco9_MEAN', 'fe_people_not_living', 'fe_people_weird_stat', 'geo_elimbasu_LE_3_MEAN', 'geo_elimbasu_LE_4_MEAN', 'geo_energcocinar_LE_0_MEAN', 'geo_energcocinar_LE_1_MEAN', 'geo_energcocinar_LE_2_MEAN', 'geo_epared_LE_0_MEAN', 'geo_epared_LE_2_MEAN', 'geo_etecho_LE_2_MEAN', 'geo_eviv_LE_0_MEAN', 'geo_hogar_mayor_MEAN', 'geo_hogar_nin_MEAN', 'geo_manual_elec_LE_1_MEAN', 'geo_manual_elec_LE_2_MEAN', 'geo_manual_elec_LE_3_MEAN', 'geo_pared_LE_0_MEAN', 'geo_pared_LE_1_MEAN', 'geo_pared_LE_3_MEAN', 'geo_pared_LE_4_MEAN', 'geo_pared_LE_5_MEAN', 'geo_pared_LE_6_MEAN', 'geo_pared_LE_7_MEAN', 'hacapo', 'hacdor', 'mobilephone', 'parentesco1', 'parentesco_LE', 'rez_esc', 'techo_LE', 'v14a', 'v18q']
#cols_2_drop = ['agg18_estadocivil1_MEAN', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_MEAN', 'agg18_parentesco12_MEAN', 'agg18_parentesco4_MEAN', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_MEAN', 'agg18_parentesco8_MEAN', 'fe_people_weird_stat', 'hacapo', 'hacdor', 'mobilephone', 'parentesco1', 'parentesco_LE', 'rez_esc', 'v14a']
#cols_2_drop=[]

X.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
test.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
XY = pd.concat([X,y], axis=1)
max_corr = XY.corr()['Target'].loc[lambda x: abs(x)>0.2].index
#min_corr = XY.corr()['Target'].loc[lambda x: abs(x)<0.05].index
_ = plt.figure(figsize=(10,7))
_ = sns.heatmap(XY[max_corr].corr(), vmin=-0.5, vmax=0.5, cmap='coolwarm')
from sklearn.metrics import f1_score
def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.99, current_iter)
    return max(lr, min_learning_rate)

import lightgbm as lgb
fit_params={"early_stopping_rounds":300, 
            "eval_metric" : 'multiclass',
            "eval_metric" : evaluate_macroF1_lgb, 
            #"eval_set" : [(X_train,y_train), (X_test,y_test)],
            'eval_names': ['train', 'early_stop'],
            'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_power_0997)],
            'verbose': False,
            'categorical_feature': 'auto'}

#fit_params['verbose'] = 200
#v8
#opt_parameters = {'colsample_bytree': 0.93, 'min_child_samples': 56, 'num_leaves': 19, 'subsample': 0.84}
#v9
#opt_parameters = {'colsample_bytree': 0.89, 'min_child_samples': 70, 'num_leaves': 17, 'subsample': 0.96}
#v14
#opt_parameters = {'colsample_bytree': 0.88, 'min_child_samples': 90, 'num_leaves': 16, 'subsample': 0.94}
#v17
opt_parameters = {'colsample_bytree': 0.89, 'min_child_samples': 90, 'num_leaves': 14, 'subsample': 0.96}
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

class VotingPrefitClassifier(VotingClassifier):
    '''
    This implements the VotingClassifier with prefitted classifiers
    '''
    def fit(self, X, y, sample_weight=None, **fit_params):
        self.estimators_ = [x[1] for x in self.estimators]
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        return self    
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def train_single_model(clf_, X_, y_, random_state_=314, opt_parameters_={}, fit_params_={}):
    c = clone(clf_)
    c.set_params(**opt_parameters_)
    c.set_params(random_state=random_state_)
    return c.fit(X_, y_, **fit_params_)

def train_model_in_nestedCV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            inner_seed=31416, inner_n=10, outer_seed=314, outer_n=10,
                            opt_parameters_={}, fit_params_={},
                            verbose=True):
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': [],
                 'score_i_ave': [],
                 'score_i_std': [],
                 'score_i_early_stop_ave': [],
                 'score_o_early_stop': [],
                 'score_o_early_stop_vc_w0_soft': [],
                 'score_o_early_stop_vc_w0_hard': []
                }
    # full-sample oof prediction
    y_full_oof = pd.Series(np.zeros(shape=(y.shape[0],)), 
                          index=y.index)
    
    if 'sample_weight' in metric_args:
        sample_weight=metric_args['sample_weight']

    outer_cv = StratifiedKFold(outer_n, shuffle=True, random_state=outer_seed)
    for n_outer_fold, (outer_trn_idx, outer_val_idx) in enumerate(outer_cv.split(X,y)):
        print('--- Outer loop iteration: {} ---'.format(n_outer_fold))
        X_out, y_out = X.iloc[outer_trn_idx], y.iloc[outer_trn_idx]
        X_stp, y_stp = X.iloc[outer_val_idx], y.iloc[outer_val_idx]

        inner_cv = StratifiedKFold(inner_n, shuffle=True, random_state=inner_seed+n_outer_fold)
        # The out-of-fold (oof) prediction for the k-1 sample in the outer CV loop
        y_outer_oof = pd.Series(np.zeros(shape=(X_out.shape[0],)), 
                          index=X_out.index)
        scores_inner = []
        clfs_inner = []

        for n_inner_fold, (inner_trn_idx, inner_val_idx) in enumerate(inner_cv.split(X_out,y_out)):
            X_trn, y_trn = X_out.iloc[inner_trn_idx], y_out.iloc[inner_trn_idx]
            X_val, y_val = X_out.iloc[inner_val_idx], y_out.iloc[inner_val_idx]

            if fit_params_:
                # use _stp data for early stopping
                fit_params_["eval_set"] = [(X_trn,y_trn), (X_stp,y_stp)]
                fit_params_['verbose'] = False

            clf = train_single_model(model, X_trn, y_trn, 314+n_inner_fold, opt_parameters_, fit_params_)

            clfs_inner.append(('{}{}_inner'.format(model_name,n_inner_fold), clf))
            # evaluate performance
            y_outer_oof.iloc[inner_val_idx] = clf.predict(X_val)
            if 'sample_weight' in metric_args:
                metric_args['sample_weight'] = y_val.map(sample_weight)
            scores_inner.append(metric(y_val, y_outer_oof.iloc[inner_val_idx], **metric_args))
            #cleanup
            del X_trn, y_trn, X_val, y_val

        # Store performance info for this outer fold
        if 'sample_weight' in metric_args:
            metric_args['sample_weight'] = y_outer_oof.map(sample_weight)
        perf_eval['score_i_oof'].append(metric(y_out, y_outer_oof, **metric_args))
        perf_eval['score_i_ave'].append(np.mean(scores_inner))
        perf_eval['score_i_std'].append(np.std(scores_inner))
        
        # Do the predictions for early-stop sub-sample for comparison with VotingPrefitClassifier
        if 'sample_weight' in metric_args:
            metric_args['sample_weight'] = y_stp.map(sample_weight)
        score_inner_early_stop = [metric(y_stp, clf_.predict(X_stp), **metric_args)
                                   for _,clf_ in clfs_inner]
        perf_eval['score_i_early_stop_ave'].append(np.mean(score_inner_early_stop))
        
        # Record performance of Voting classifiers
        w = np.array(scores_inner)
        for w_, w_name_ in [(None, '_w0')#,
                            #(w/w.sum(), '_w1'),
                            #((w**2)/np.sum(w**2), '_w2')
                           ]:
            vc = VotingPrefitClassifier(clfs_inner, weights=w_).fit(X_stp, y_stp)
            for vote_type in ['soft', 'hard']:
                vc.voting = vote_type
                if 'sample_weight' in metric_args:
                    metric_args['sample_weight'] = y_stp.map(sample_weight)
                perf_eval['score_o_early_stop_vc{}_{}'.format(w_name_, vote_type)].append(metric(y_stp, vc.predict(X_stp), **metric_args))

        if fit_params_:
            # Train main model for the voting average
            fit_params_["eval_set"] = [(X_out,y_out), (X_stp,y_stp)]
            if verbose:
                fit_params_['verbose'] = 200
        #print('Fit the final model on the outer loop iteration: ')
        clf = train_single_model(model, X_out, y_out, 314+n_outer_fold, opt_parameters_, fit_params_)
        if 'sample_weight' in metric_args:
            metric_args['sample_weight'] = y_stp.map(sample_weight)
        perf_eval['score_o_early_stop'].append(metric(y_stp, clf.predict(X_stp), **metric_args))
        clfs.append(('{}{}'.format(model_name,n_outer_fold), clf))
        y_full_oof.iloc[outer_val_idx] = clf.predict(X_stp)
        # cleanup
        del inner_cv, X_out, y_out, X_stp, y_stp, clfs_inner

    return clfs, perf_eval, y_full_oof

def print_nested_perf_clf(name, perf_eval):
    print('Performance of the inner-loop model (the two should agree):')
    print('  Mean(mean(Val)) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                      np.mean(perf_eval['score_i_ave']),
                                                                      np.std(perf_eval['score_i_ave'])
                                                                     ))
    print('  Mean(mean(EarlyStop)) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                      np.mean(perf_eval['score_i_early_stop_ave']),
                                                                      np.std(perf_eval['score_i_early_stop_ave'])
                                                                     ))
    print('Mean(inner OOF) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                       np.mean(perf_eval['score_i_oof']), 
                                                                       np.std(perf_eval['score_i_oof'])
                                                                      ))
    print('Mean(EarlyStop) score outer {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                      np.mean(perf_eval['score_o_early_stop']),
                                                                      np.std(perf_eval['score_o_early_stop'])
                                                                     ))
    print('Mean(EarlyStop) outer VotingPrefit SOFT: {:.4f}+-{:.4f}'.format(np.mean(perf_eval['score_o_early_stop_vc_w0_soft']),
                                                                           np.std(perf_eval['score_o_early_stop_vc_w0_soft'])                                                                    
                                                                    ))
    print('Mean(EarlyStop) outer VotingPrefit HARD: {:.4f}+-{:.4f}'.format(np.mean(perf_eval['score_o_early_stop_vc_w0_hard']),
                                                                           np.std(perf_eval['score_o_early_stop_vc_w0_hard'])
                                                                    ))
clf  = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                         random_state=1, silent=True, metric='None', 
                         n_jobs=4, n_estimators=5000, class_weight='balanced')

clfs_, perf_eval, y_full_oof = train_model_in_nestedCV(clf, X, y, f1_score, 
                                                      metric_args={'average':'macro'},
                                                      model_name='lgbm', 
                                                      opt_parameters_=opt_parameters,
                                                      fit_params_=fit_params, 
                                                      inner_n=10, outer_n=10,
                                                      verbose=False)
w = np.array(perf_eval['score_o_early_stop'])
ws = [(None, '_w0'),
  (w/w.sum(), '_w1'),
  ((w**2)/np.sum(w**2), '_w2')
 ]
vc = {}
for w_, w_name_ in ws:
    vc['vc{}'.format(w_name_)] = VotingPrefitClassifier(clfs_, weights=w_).fit(X, y)
clf_final = clfs_[0][1]
global_score = np.mean(perf_eval['score_i_oof'])
global_score_std = np.std(perf_eval['score_i_oof'])

print_nested_perf_clf('lgbm', perf_eval)
print('Outer OOF score {} Classifier: {:.4f}'.format('lgbm', f1_score(y, y_full_oof, average='macro')))
perf_eval_df = pd.DataFrame(perf_eval)
perf_eval_df
from sklearn.metrics import precision_score, recall_score, classification_report
#print(classification_report(y_test, clf_final.predict(X_test)))
#vc.voting = 'hard'
#print(classification_report(y_test, vc.predict(X_test)))
#vc.voting = 'soft'
#print(classification_report(y_test, vc.predict(X_test)))
def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    # Plot feature importances
    if not doWorst:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_feat].index        
    else:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[-n_feat:].index
    
    mean_imp = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    df_2_neglect = mean_imp[mean_imp['importance'] < 1e-3]
    print('The list of features with 0 importance: ')
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    #plt.savefig('lgbm_importances.png')
    
importance_df = pd.DataFrame()
importance_df["feature"] = X.columns.tolist()      
importance_df["importance"] = clf_final.booster_.feature_importance('gain')
display_importances(feature_importance_df_=importance_df, n_feat=20)
#display_importances(feature_importance_df_=importance_df, doWorst=True, n_feat=20)
import shap
shap_values = shap.TreeExplainer(clf_final.booster_).shap_values(X)

#shap_df = pd.DataFrame()
#shap_df["feature"] = X_train.columns.tolist()    
#shap_df["importance"] = np.sum(np.abs(shap_values), 0)[:-1]
#display_importances(feature_importance_df_=shap_df, n_feat=20)
shap.summary_plot(shap_values, X, plot_type='bar')
y_subm = pd.read_csv('../input/sample_submission.csv')
from datetime import datetime
now = datetime.now()

sub_file = 'submission_LGB_{:.4f}_{}.csv'.format(global_score, str(now.strftime('%Y-%m-%d-%H-%M')))
y_subm['Target'] = clf_final.predict(test) + 1
y_subm.to_csv(sub_file, index=False)

# Store predictions with voting classifiers
for vc_name_,vc_ in vc.items():
    for vc_type_ in ['soft', 'hard']:
        vc_.voting = vc_type_
        name = '{}_{}'.format(vc_name_, vc_type_)
        y_subm_vc = y_subm.copy(deep=True)
        y_subm_vc.loc[:,'Target'] = vc_.predict(test) + 1
        sub_file = 'submission_{}_LGB_{:.4f}_{}.csv'.format(name, 
                                                            global_score, 
                                                            str(now.strftime('%Y-%m-%d-%H-%M'))
                                                           )
        y_subm_vc.to_csv(sub_file, index=False)
!ls
