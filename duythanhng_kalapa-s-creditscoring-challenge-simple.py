import pandas as pd

from lightgbm import LGBMClassifier

import category_encoders as ce

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/kalapas/train.csv')

test_df = pd.read_csv('/kaggle/input/kalapas/test.csv')
train_df.head()
def feature_engineering(train, test):

    labels = train['label']

    data = train.drop(columns=['label']).append(test, ignore_index=True)

    remove_features = ['Field_1', 'Field_2', 'Field_4', 'Field_5', 'Field_6', 'Field_7', 'Field_8', 'Field_9',

                       'Field_11', 'Field_12', 'Field_15', 'Field_18', 'Field_25', 'Field_32', 'Field_33',

                       'Field_34', 'Field_35', 'gioiTinh', 'diaChi', 'Field_36', 'Field_38', 'Field_40',

                       'Field_43', 'Field_44', 'Field_45', 'Field_46', 'Field_47', 'Field_48', 'Field_49',

                       'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62', 'Field_65', 'Field_66',

                       'Field_68', 'maCv', 'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',

                       'currentLocationCountry', 'currentLocationName', 'currentLocationState', 'homeTownCity',

                       'homeTownCountry', 'homeTownName', 'homeTownState', 'F_startDate', 'F_endDate',

                       'E_startDate', 'E_endDate', 'C_startDate', 'C_endDate', 'G_startDate', 'G_endDate',

                       'A_startDate', 'A_endDate', 'brief']



    cat_features_count_encode = ['Field_4', 'Field_12', 'Field_18', 'Field_34', 'gioiTinh', 'diaChi', 'Field_36',

                                 'Field_38', 'Field_45', 'Field_46', 'Field_47', 'Field_48', 'Field_49',

                       'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62', 'Field_65', 'Field_66',

                       'Field_68', 'maCv', 'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',

                       'currentLocationCountry', 'currentLocationName', 'currentLocationState', 'homeTownCity',

                       'homeTownCountry', 'homeTownName', 'homeTownState', 'brief']

    

    cat_date_array = ['Field_1', 'Field_2', 'Field_5', 'Field_6', 'Field_7', 'Field_8', 'Field_9', 'Field_11',

                      'Field_15', 'Field_25', 'Field_32', 'Field_33', 'Field_35', 'Field_40', 'Field_43',

                      'Field_44', 'F_startDate', 'F_endDate', 'E_startDate', 'E_endDate', 'C_startDate',

                      'C_endDate', 'G_startDate', 'G_endDate', 'A_startDate', 'A_endDate']

    for col in cat_date_array:

        data[col+'Year'] = pd.DatetimeIndex(data[col]).year

        data[col+'Month'] = pd.DatetimeIndex(data[col]).month

        data[col+'Day'] = pd.DatetimeIndex(data[col]).day

    

    data[remove_features].fillna("Missing", inplace=True)

    count_en = ce.CountEncoder()

    cat_ce = count_en.fit_transform(data[cat_features_count_encode])

    data = data.join(cat_ce.add_suffix("_ce"))

    

    data.replace("None", -1, inplace=True)

    data.replace("Missing", -999, inplace=True)

    data.fillna(-999, inplace=True)



    _train = data[data['id'] < 53030]

    _test = data[data['id'] >= 53030]

    

    _train["label"] = labels



    _train.drop(columns=remove_features, inplace=True)

    _test.drop(columns=remove_features, inplace=True)

    

    return _train, _test
train_data, test_data = feature_engineering(train_df, test_df)
def calculate_woe_iv(dataset, feature, target):

    lst = []

    for i in range(dataset[feature].nunique()):

        val = list(dataset[feature].unique())[i]

        lst.append({

            'Value': val,

            'All': dataset[dataset[feature] == val].count()[feature],

            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],

            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]

        })

    dset = pd.DataFrame(lst)

    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()

    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()

    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])

    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']

    iv = dset['IV'].sum()

    dset = dset.sort_values(by='WoE')

    return dset, iv



USELESS_PREDICTOR = []

WEAK_PREDICTOR = []

MEDIUM_PREDICTOR = []

STRONG_PREDICTOR = []

GOOD_PREDICTOR = []

IGNORE_FEATURE = USELESS_PREDICTOR + WEAK_PREDICTOR

for col in train_data.columns:

    if col == 'label' or col == 'id': continue

    elif col in IGNORE_FEATURE: continue

    else:

        print('WoE and IV for column: {}'.format(col))

        final, iv = calculate_woe_iv(train_data, col, 'label')

        iv = round(iv,2)

        print('IV score: ' + str(iv))

        print('\n')

        if (iv < 0.02) and col not in USELESS_PREDICTOR:

            USELESS_PREDICTOR.append(col)

        elif iv >= 0.02 and iv < 0.1 and col not in WEAK_PREDICTOR:

            WEAK_PREDICTOR.append(col)

        elif iv >= 0.1 and iv < 0.3 and col not in MEDIUM_PREDICTOR:

            MEDIUM_PREDICTOR.append(col)

        elif iv >= 0.3 and iv < 0.5 and col not in STRONG_PREDICTOR:

            STRONG_PREDICTOR.append(col)

        elif iv >= 0.5 and col not in GOOD_PREDICTOR:

            GOOD_PREDICTOR.append(col)
print('USELESS_PREDICTOR')

print(len(USELESS_PREDICTOR))

print('WEAK_PREDICTOR')

print(len(WEAK_PREDICTOR))

print('MEDIUM_PREDICTOR')

print(len(MEDIUM_PREDICTOR))

print('STRONG_PREDICTOR')

print(len(STRONG_PREDICTOR))

print('GOOD_PREDICTOR')

print(len(GOOD_PREDICTOR))
IGNORE_FEATURE = USELESS_PREDICTOR

final_train_data = train_data.drop(columns=IGNORE_FEATURE)

final_test_data = test_data.drop(columns=[col for col in IGNORE_FEATURE if col not in ['label']])
import gc

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, recall_score, classification_report

import seaborn as sns





# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances01.png')

    

def display_roc_curve(y_, oof_preds_,sub_preds_,folds_idx_):

    # Plot ROC curves

    plt.figure(figsize=(6,6))

    scores = [] 

    for n_fold, (_, val_idx) in enumerate(folds_idx_):  

        # Plot the roc curve

        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])

        score = 2 * auc(fpr, tpr) -1

        scores.append(score)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (Gini = %0.4f)' % (n_fold + 1, score))

    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)

    score = 2 * auc(fpr, tpr) -1

    plt.plot(fpr, tpr, color='b',

             label='Avg ROC (Gini = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),

             lw=2, alpha=.8)

    

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('LightGBM ROC Curve')

    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig('roc_curve.png')





# LightGBM GBDT with Stratified KFold

def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    gc.collect()

    # Cross validation model

    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=500)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['label','id']]

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['label'])):        

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['label'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['label'].iloc[valid_idx]



        clf = LGBMClassifier(

            nthread=4,

            n_estimators=10000,

            learning_rate=0.02,

            num_leaves=128,

            colsample_bytree=0.9497036,

            subsample=0.8715623,

            max_depth=8,

            reg_alpha=0.041545473,

            reg_lambda=0.0735294,

            min_split_gain=0.0222415,

            min_child_weight=39.3259775,

            silent=-1,

            verbose=-1

        )



        clf.fit(train_x, train_y.ravel(), eval_set=[(train_x, train_y), (valid_x, valid_y)], 

            eval_metric='auc', verbose= 1000, early_stopping_rounds= 200)



        oof_pred = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        

        pred = clf.predict(valid_x, num_iteration=clf.best_iteration_)

        print('F1 Score: ' + str( f1_score(valid_y, pred) ))

        print('Recall Score: ' + str( recall_score(valid_y, pred) ))

        

        sub_pred = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        oof_preds[valid_idx] = oof_pred

        sub_preds += sub_pred

                

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()



    print('Full AUC score %.6f' % roc_auc_score(train_df['label'], oof_preds))

    

    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_df[feats], train_df['label'])]

    display_roc_curve(y_=train_df['label'],oof_preds_=oof_preds,sub_preds_ = sub_preds, folds_idx_=folds_idx)

    

    # Write submission file and plot feature importance

    test_df['label'] = sub_preds

    test_df[['id', 'label']].to_csv('submission.csv', index= False)

    display_importances(feature_importance_df)
kfold_lightgbm(final_train_data, final_test_data, 5)