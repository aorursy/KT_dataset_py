# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 



from xgboost import XGBClassifier

from sklearn.model_selection import KFold

#category_encodersは様々なカテゴリ特徴量(通常のものからバイナリ、OneHot、ハッシング

#など様々)をいくつかの変換方法でNumeric型の特徴量に変換

#最近のものでsklearn.preprocessingより良さげで、OneHot化も4行くらいでできちゃう

from category_encoders import CountEncoder

#PipelineはscikitAPIの機能の一つ。Estimator(データから学習する機能。fitなど)の処理を

#まとめて実行可能

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss



import matplotlib.pyplot as plt



from sklearn.multioutput import MultiOutputClassifier



import os

import warnings

warnings.filterwarnings('ignore')
#バイナリ分類

#ラベル間の相関情報がなくなり、学習数が多く遅い



SEED = 42

NFOLDS = 5

np.random.seed(SEED)
DATA_DIR = '/kaggle/input/lish-moa/'

train = pd.read_csv(DATA_DIR + 'train_features.csv')

targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')



test = pd.read_csv(DATA_DIR + 'test_features.csv')

sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')
# sigIDのカラムを消し、ndarray化

X = train.iloc[:,1:].to_numpy()

X_test = test.iloc[:,1:].to_numpy()

y = targets.iloc[:,1:].to_numpy() 
#データを確認

print(train.shape, targets.shape, test.shape, sub.shape)
train.head()
#薬物VS薬物を溶かすのに使用した溶剤で、vehicleがその溶剤

train.cp_type.value_counts()
targets.head()
sub.head()
#分類モデル

#時間によってはXGBClassifier()引数にtree_method='gpu_hist'いれてAWSかGCPでGPU使用

#category_encodersのCounterEncoder(cols=リスト)→エンコードする列のリストで

#カラムの特徴量をその特徴の集合数の数値でカテゴリ数値化??

#pipelineは変換(fit,transform)を順番に一度で行う機能でpipeline(ステップ)→ステップは

#リストに、学習機にかけるfit前データと最終的な学習器をそれぞれタプルで設定する

#こんな風に>>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

#このオブジェクトにはいくつかメソッドもある

classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))

clf = Pipeline([('encode', CountEncoder(cols=[0,2])),

                ('classify', classifier)])
#pipelineオブジェにさらにパラメータ設定

#'モデル__estimator__通常のパラ設定'

params = {'classify__estimator__colsample_bytree': 0.6522,#各ステージの決定木の使う特徴量の割合

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }

_ = clf.set_params(**params)
print(y.shape, test.shape)
#モデルの学習(目的変数のクラス数206個を学習させる必要があるためかなり時間かかる)



#oofはout of foldで交差検証の分割で学習に使わなかったデータを指す

oof_preds = np.zeros(y.shape)

#testを3982行206列に揃えて全て0に

test_preds = np.zeros((test.shape[0], y.shape[1]))

oof_losses = []

#K-Fold交差検証でデータをk個に分けてn個を訓練用、残りk-n個をテスト用にわける

#分けられたn個のデータが絶対1回はテスト用として使われるようn回検定する

#n_splitはデータの分割数つまりk

kf = KFold(n_splits=NFOLDS)#NFOLDS=5



#k-fold.split(X, y=None, groups=None)で分割データを訓練、テストセットに代入

#trn_idx(訓練用),val_idx(テスト用)にしてループ

for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):

    print('Starting fold: ', fn)

    X_train, X_val = X[trn_idx], X[val_idx]

    y_train, y_val = y[trn_idx], y[val_idx]

    

    #vehicleは直接的に関係なさそうなので削除

    ctl_mask = X_train[:,0] == 'ctl_vehicle'

    #~(チルダ)は2進数のビット反転が基本だがpandasでは集合Notの役割

    X_train = X_train[~ctl_mask, :]

    y_train = y_train[~ctl_mask]

    

    clf.fit(X_train, y_train)

    #predict_proba(予測したいイテラブルデータ)で最終推定器ゆえサンプル数とクラス数のarray返す

    #つまりテストデータでfit,transformしクラスあたりの推定リストを返す

    val_preds = clf.predict_proba(X_val)

    #陽性クラスを抽出

    #array(奥行,行,列)ゆえ1の列が陽性??

    val_preds = np.array(val_preds)[:,:,1].T

    #0でまっさらなoof_predsの検証データに相当するarrayに前行で推測した陽性クラスをいれる

    oof_preds[val_idx] = val_preds

    

    #sklearn.metrics.log_loss(正解y_true,予測y_pred)で交差エントロピーを使った評価

    #np.ravelはflatten関数みたく多次元を1次元配列にして返すがflattenと違い元データを変更して返す点に注意

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    oof_losses.append(loss)

    #テストデータも同様に予測かける

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T

    test_preds += preds / NFOLDS

    

    print(oof_losses)

    print('Mean OOF loss across folds', np.mean(oof_losses))

    print('STD OOF loss across folds', np.std(oof_losses))

    
#ctl_vehicleを学習に考慮しないようにするためoof_preds内のvehicleタイプを再度0に設定

control_mask = train['cp_type']=='ctl_vehicle'

oof_preds[control_mask] = 0

#正解y(train_targets_scored)と新たに設定した予測oof_predsで交差エントロピーの損失計算

print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
#テストデータでも同様にvehicleを考慮しないようにする

control_mask = test['cp_type']=='ctl_vehicle'



test_preds[control_mask] = 0
#提出ファイルを作成

sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)