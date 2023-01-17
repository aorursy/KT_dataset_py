import pandas as pd

sub_lgbm_sk = pd.read_csv('../input/qiita7/submission_lightgbm_skfold_test.csv')
sub_lgbm_ho = pd.read_csv('../input/qiita7/submission_lightgbm_holdout_test.csv')
sub_rf = pd.read_csv('../input/qiita7/submission_randomforest_test.csv')
sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])
sub['Survived'] = sub_lgbm_sk['Survived'] + sub_lgbm_ho['Survived'] + sub_rf['Survived']
sub['Survived'] = (sub['Survived'] >= 2).astype(int)
sub.to_csv('submission_lightgbm_ensemble.csv', index=False)
sub.head()
sub_lgbm_sk.head()
sub_lgbm_ho.head()
sub_rf.head()