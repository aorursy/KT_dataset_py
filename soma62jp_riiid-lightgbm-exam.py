# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



import warnings

warnings.filterwarnings('ignore')



import riiideducation



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
env = riiideducation.make_env()
train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5,

                      dtype={

                          'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                      })
train_df
# 追加するデータ1 user_answers_df

train_questions_only_df = train_df[train_df['answered_correctly'] != -1]

grouped_by_user_df = train_questions_only_df.groupby('user_id')

grouped_by_user_df.head()
user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']

user_answers_df
# 追加するデータ2 questions_df

questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')



grouped_by_content_df = train_questions_only_df.groupby('content_id')



content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] }).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked']



questions_df = questions_df.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')



bundle_dict = questions_df['bundle_id'].value_counts().to_dict()



# right_answers 正解数

questions_df['right_answers'] = questions_df['mean_accuracy'] * questions_df['question_asked']



questions_df['bundle_size'] = questions_df['bundle_id'].apply(lambda x: bundle_dict[x])
questions_df
# 追加するデータ3 bundle_answers_df

grouped_by_bundle_df = questions_df.groupby('bundle_id')



bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

bundle_answers_df.columns = ['bundle_right_answers', 'bundle_questions_asked']



bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_right_answers'] / bundle_answers_df['bundle_questions_asked']
bundle_answers_df
# 追加するデータ4 part_answers_df

grouped_by_part_df = questions_df.groupby('part')



part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()



part_answers_df.columns = ['part_right_answers', 'part_questions_asked']

part_answers_df['part_accuracy'] = part_answers_df['part_right_answers'] / part_answers_df['part_questions_asked']
part_answers_df
features = [

    'timestamp',

    'mean_user_accuracy', 

    'questions_answered',

    'mean_accuracy',

    'question_asked',

    'prior_question_elapsed_time', 

    'prior_question_had_explanation',

    'bundle_size', 

    'bundle_accuracy',

    'part_accuracy', 

    'right_answers',

    #'user_answer',

    'correct_answer'

]



target = 'answered_correctly'
# 講義(-1)以外を抽出 train

train_part_df = train_df[train_df[target] != -1]

train_part_df
# user_answers_df

train_part_df = train_part_df.merge(user_answers_df, how='left', on='user_id')



# questions_df

train_part_df = train_part_df.merge(questions_df, how='left', left_on='content_id', right_on='question_id')



# bundle_answers_df

train_part_df = train_part_df.merge(bundle_answers_df, how='left', on='bundle_id')



# part_answers_df

train_part_df = train_part_df.merge(part_answers_df, how='left', on='part')
# ユーザーが質問に回答した後、説明と正しい回答を確認したかどうか 欠損値をFalseと置く、 astypeでデータ型の変換(キャスト)

train_part_df['prior_question_had_explanation'] = train_part_df['prior_question_had_explanation'].fillna(value=False).astype(int)



train_part_df.fillna(value = -1, inplace = True)
train_part_df
import lightgbm as lgb

import optuna

from sklearn.model_selection import train_test_split,KFold,cross_validate,cross_val_score

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc



train_part_df.info()
train_part_df = train_part_df.drop('tags',axis=1)

#X_train = train_part_df.drop(target,axis=1)

X_train = train_part_df[features] 

y_train = train_part_df[target]





# 学習データ = 75% テストデータ = 25%　に分割

train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.25, 

                                                    shuffle = True , random_state = 0)
train_x.info()
train_y
test_x
test_y
# LightGBM用のDatasetに格納

dtrain = lgb.Dataset(train_x, label=train_y)

dtest = lgb.Dataset(test_x, label=test_y)
#------------------------LightGBM Model 最適化-----------------------

# ハイパーパラメータ検索ライブラリ「Optuna」を使用



def objectives(trial):

    

    #--optunaでのハイパーパラメータサーチ範囲の設定

    # 二値分類

    # binary_logloss(クロスエントロピー)最適化

    # 勾配ブースティングを使用する

    params = {

        #fixed

        'boost_from_average': True, ## ONLY NEED FOR LGB VERSION 2.1.2

        "objective": "binary",

        'boosting_type':'gbdt',

        'max_depth':-1,

        'learning_rate':0.1,

        'n_estimators': 1000,

        'metric':'binary_logloss',



        #variable

        'num_leaves': trial.suggest_int('num_leaves', 10, 300),

        'reg_alpha': trial.suggest_loguniform('reg_alpha',0.001, 10),

        'reg_lambda':trial.suggest_loguniform('reg_lambda', 0.001, 10),

    }



    # LightGBMで学習+予測

    model = lgb.LGBMClassifier(**params,random_state=0)

    

    # kFold交差検定で決定係数を算出し、各セットの平均値を返す

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    scores = cross_validate(model, X=train_x, y=train_y,scoring='r2',cv=kf)   



    # 最小化問題とするので1.0から引く

    return 1.0 - scores['test_score'].mean()
# optunaによる最適化呼び出し

opt = optuna.create_study(direction='minimize')

opt.optimize(objectives, n_trials=20)
# 実行結果表示

print('最終トライアル回数:{}'.format(len(opt.trials)))

print('ベストトライアル:')

trial = opt.best_trial

print('値:{}'.format(trial.value))

print('パラメータ:')

for key, value in trial.params.items():

    print('{}:{}'.format(key, value))
from sklearn.metrics import mean_squared_error,r2_score,f1_score



gbm_best = lgb.train(trial.params, dtrain)



def model_Eval_Reg(testX,testY):

    

    predict_best = gbm_best.predict(testX)

    print(predict_best.shape)



    np.set_printoptions(threshold=10)        # 10件表示設定

    pd.set_option('display.max_rows',10)      # 10件表示設定



    preds = pd.DataFrame({"preds":predict_best, "true":testY})

    preds



    # 残差プロット

    preds["residuals"] = preds["true"] - preds["preds"]

    preds.plot(x = "preds", y = "residuals",kind = "scatter")



    # モデルのあてはめ

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)



    ax.scatter(preds["true"], preds["preds"],label="LigntGBM Model Fitting")

    ax.set_xlabel('predicted')

    ax.set_ylabel('true')

    ax.set_aspect('equal')



    # rmseとr2を求める

    rmse = np.sqrt(mean_squared_error(preds["true"], preds["preds"]))

    print('rmse:',rmse)

    r2 = r2_score(preds["true"], preds["preds"])

    print('r2:',r2)

    

    # 重要度プロット

    lgb.plot_importance(gbm_best,importance_type='split',max_num_features = 20,figsize=(12,6))

    

def model_Eval_Cls(testX,testY):



    predict_best = gbm_best.predict(testX)

    predictions_lgbm = np.where(predict_best > 0.5, 1, 0)



    #lgb.plot_importance(gbm_best,importance_type='split',max_num_features = 20,figsize=(12,6))

    lgb.plot_importance(gbm_best,importance_type='split',figsize=(12,6))



    # ROC曲線を出力

    plt.figure()

    false_positive_rate, recall, thresholds = roc_curve(testY, predict_best)

    roc_auc = auc(false_positive_rate, recall)

    plt.title('Receiver Operating Characteristic (ROC)')

    plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0,1], [0,1], 'r--')

    plt.xlim([0.0,1.0])

    plt.ylim([0.0,1.0])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

    print('AUC Curve score:', roc_auc)



    # 混同行列出力

    plt.figure()

    cm = confusion_matrix(testY, predictions_lgbm)

    labels = ['True', 'False']

    plt.figure(figsize=(8,6))

    sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2)

    plt.title('Confusion Matrix')

    plt.ylabel('True Class')

    plt.xlabel('Predicted Class')

    plt.show()



    tp, fn, fp, tn = cm.flatten()

    print('\nSpecificity:\n',(tn / (fp + tn)))

    print('\nFalse Negative Rate:\n',(fn / (tp + fn)))

    print('\nFalse Positive Rate:\n',(fp / (fp + tn)))



    # 正解率プロット

    print('\nAccuracy:\n', accuracy_score(testY, predictions_lgbm))

    print('\nAUC:\n', roc_auc_score(testY, predictions_lgbm))

    print('\nConfusion matrix:\n', confusion_matrix(testY, predictions_lgbm))

    #print('\nPrecision:\n', precision_score(testY, predictions_lgbm))

    #print('\nRecall:\n', recall_score(testY, predictions_lgbm))

    print('\nF-measure:\n', f1_score(testY, predictions_lgbm))

    

    print('\nPrecision:\n',(tp / (tp + fp)))

    print('\nRecall:\n',(tp / (tp + fn)))
# モデルデータの検証

model_Eval_Cls(train_x,train_y)
# 過学習の確認

# テストデータの検証

model_Eval_Cls(test_x,test_y)
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    y_preds = []

    

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

    test_df = test_df.merge(bundle_answers_df, how = 'left', on = 'bundle_id')

    test_df = test_df.merge(part_answers_df, how = 'left', on = 'part')

    

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df.fillna(value = -1, inplace = True)

    

    y_pred = gbm_best.predict(test_df[features])

    #y_pred = gbm_best.predict(X_train)

    pred_lgbm = np.where(y_pred > 0.5, 1, 0)

    y_preds.append(pred_lgbm)

    

    #for mdl in gbm_best:

    #    y_pred = mdl.predict(test_df[features], num_iteration=mdl.best_iteration)

    #    y_preds.append(y_pred)

        

    y_preds = sum(y_preds) / len(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])