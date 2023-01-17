# ライブラリのインポート

import numpy as np

import pandas as pd



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm
# データを読み込んだり特徴量エンジニアリングしたりする。自分でやってください

# ここが空欄なので、必然的に以下は全てエラーになります。
%%time

# CVしてスコアを見てみる

# なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

# 次回Build Modelの内容ですが、是非各自検討してみてください

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = GradientBoostingClassifier() # ここではデフォルトのパラメータになっている。各自の検討項目です

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1] # predict_probaで確率を出力する
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')