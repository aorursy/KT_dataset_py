# いつものライブラリ追加

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# 可視化用ライブラリ追加

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# kaggleの入力ファイル一覧確認

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv(f'{dirname}/train.zip')

sample_submission = pd.read_csv(f'{dirname}/sample_submission.zip')

test = pd.read_csv(f'{dirname}/test.zip')
train
sample_submission
test
# 時間の単位チェック

# 結果を見る限り、trip_durationの単位は秒っぽい

from datetime import datetime as dt

do = dt.strptime(train['dropoff_datetime'][0], '%Y-%m-%d %H:%M:%S')

pu = dt.strptime(train['pickup_datetime'][0], '%Y-%m-%d %H:%M:%S')

(do - pu).seconds
# 乗車時間チェック

# 明らかにおかしいデータが入っている

# 例えば1秒とか1939736秒=約22日とか

# 86392時間 = 23時間ちょっと怪しいので両端のデータは切ってしまったほうがいいかもしれない

# つまり怪しいデータがそこそこ入っているのでそれを弾く前処理がいる

train['trip_duration'].sort_values()
# そもそも計算時間間違っているんじゃないか説もあると思うので、trip_durationを再計算した列を追加する

# from datetime import datetime as dt

# train['datetime_diff'] = (

#     train['dropoff_datetime'].map(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))

#     - train['pickup_datetime'].map(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))

# )/np.timedelta64(1, 's')
# train['duration_diff'] = train['trip_duration'] - train['datetime_diff'] 
# 全部0だったので計算ミスはなさそう

# つまりい打刻時刻が怪しいデータが入っている方で間違いなさそう

# train['duration_diff'].sort_values()
# 紛らわしいので調査で追加しておいた行を消しておく

# del train['datetime_diff']

# del train['duration_diff']
# まずは異様に時間が短いケースを調査する、利用時間が30秒未満のものをみてみる

# 異様に trip_duration が短いものは緯度・経度が変わっていないことがわかる

train[

    train['trip_duration'] < 30

]
# ので緯度経度が変わっていないものを削除する



# 緯度・経度が変わっていない項目を探す列を追加

train['longitude_diff'] = (

    train['pickup_longitude'] == train['dropoff_longitude'])

train['latitude_diff'] = (

    train['pickup_latitude'] == train['dropoff_latitude'])



# 削除対象のインデックス番号を取得

# 削除対象は緯度または経度が変わっていないもの

# 全体に対し、削除対象の割合が小さいことは確認済み

del_indexes = train[

( (train['longitude_diff'] == True)

    | (train['latitude_diff'] == True))

].index





# データからインデックス削除

# dropはリストでまとめて指定ができる

train = train.drop(index = del_indexes)
# 確認、確かに消えている

train
# 続いて異様に渡航時間が短いデータを消す

# とりあえず1分

# 本当はしっかりいくつ以下をけす根拠データがあるといいんだろうな…

del_indexes = train[

    train['trip_duration'] < 60

].index



# データからインデックス削除

train = train.drop(index = del_indexes)
# 異様に渡航時間が長いデータも消す

# こっちもとりあえず20時間

del_indexes = train[

    train['trip_duration'] > 60*60*20

].index



# データからインデックス削除

train = train.drop(index = del_indexes)
# ここまで一回も見ていないpassenger_countと、store_and_fwd_flagについても確認する

# いや0ってなにこれも件数確認して消す

train['passenger_count'].sort_values()
# passenger_count(乗車人数)が0のものを削除 

del_indexes = train[train['passenger_count'] == 0].index

train = train.drop(index = del_indexes)
# store_and_fwd_flag :  旅行記録が車内メモリーに保存されていたかどうか？らしい？

# ちょっと意味が分からないので保留

# とりあえず Y or N を 1 or 0　に変換だけしておく

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].replace(['Y', 'N'], [1, 0])



# testもね

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].replace(['Y', 'N'], [1, 0])

# というわけでざっくり外れ値削除完了

# 下記のデータ削除作業を行った

#    - 緯度または経度が変わっていないデータ、

#    - 旅行時間が異様に短い(1分未満)データと、異様に長い(20時間以上)データ、

#    - 渡航人数が0人のデータ

train
# 回帰問題なので