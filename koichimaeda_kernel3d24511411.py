# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# 最低限のimportは上の項目で済の為、入力はinputに入ってる必要データ読込から。

# ※「../」は今いる所から1つ上に移動しますという意味。◆つまり、この場所から見て「１つ上の階層」にinputフォルダ存在

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# train.head(10)

# ◆上記のheadで中身確認も良いが、最終番号見て、行数合計知りたいのとクールなのでtail使った

# （０行目から始まる為、０〜８９０行で計８９１人）。headやtailは（）内を未入力だと「５」が入る

train.tail()
# この処理前に、NaNの合計表示したら全部０だった為、NULL合計の出力で測定不可の値を検索。ここの表示結果自体は合っている（他サイトで確認済）

train.isnull().sum()



# Age と　Cabin 使う場合、欠損値が多いので、使わない事にしようと決めた



# 後からわかった事で、Fare（旅客運賃）にNULL無だが、０が数個存在し、厳密にはこれ欠損値（タダ乗りは居なかった事も、グーグル先生が言っている）

# できれば旅客運賃Fareも相関高そうで説明変数に入れたかったが、この欠損値「0」有の為、データ整形要するのが面倒で、 除外

# （整形するなら「０」の値について、Fare全体の「中央値」を当てはめるのが良いらしい。なぜ平均値より優先か等、不明な為使わない）
# int型のものだけ使えば、後で型変換不要だと思いint探した

# フロートとint混ざる場合、intに型の変更が必要か気になった

print('データ型の確認\n{}\n'.format(train.dtypes))



# 上項の欠損値の検証も合わせて、Pclassと、SibSpと、Sex（要ダミー変数化）が要因と仮定し、検証しようと思う

# Sexをダミー変数化すれば、たぶんint型になると思う為、型変換は不要

# 現段階の仮定：Pclassが高く、SibSp（兄弟・配偶者）が多く、Sexが男、が生き残ると想定
# 性別の数量を出す（maleが男。男が女の１．８倍以上の人数いる）

train["Sex"].value_counts()
# 資産階級の人数も出してみた。上級中級の合計人数は、全体の約４割いる

# 資産階級Pclass、の数量を出す（１が上級、２が中級、３が下級）

train["Pclass"].value_counts()
# Survived（生存者y）の男、女で平均を求めた（０が死亡、１が生存の為、表示結果がそのまま生存率）

# 以下の結果見ると、◆女の生存確率が７４％以上で、男の３．９倍以上も高い

# 仮定を修正◆確実に、女の方が生き残る結果になる！この相関は強そうだ



#男の、Survived（生存者y）の平均

print(train[train.Sex == 'male'].Survived.mean())



#女の、Survived（生存者y）の平均

print(train[train.Sex == 'female'].Survived.mean())
# 各、性別ごとの「各列の平均値」を求める事できる　

# ◆.groupby('基準を入力。今回は性別で２つに分けた').mean()

train.groupby('Sex').mean()



# 平均値でみると、女性の方が男性よりも、高額の運賃で乗船。女性の貧乏一人旅なんて居ないのだろう
# Pclass列の内容を基準に、平均とった。

train.groupby('Pclass').mean()



# Pclass（資産階級みたいな）は、１が最高である。上級階級が１、中級が２、下級が３

# ◆Pclassが上級なほど、Survivedの数値が高い（上級は下級と比べ、2.5倍以上の生存率

#Pclassが上級なほど、AgeとFareの数値が高い（旅客運賃Fareは、下級の７倍弱多い。金持ちだ）
# SibSp列（兄弟or配偶者）の内容を基準に、平均とった。（０〜５人と、８人のデータ）

train.groupby('SibSp').mean()



# 生き残り順では、SibSp「１」が１番。ついで「2」、「0」、「3」、「４」。５人以上は生存していない

# →◆直線グラフにならないが、相関は強そう。１、２人連れは助かるが、多すぎると助からない

# （注◆生存者８割近くは女性。女性はほぼ複数人で乗船していた点で、データ重複が有る）

# 厳密に「何人連れ」を求めたい場合、兄弟配偶者以外に、Parch親子数も入れる必要有。家族数という新変数作るのも有
# get_dummies(xxx[""])

pd.get_dummies(train["Sex"])



# ダミー変数使うと、以下のような振り分けになる事を認識しよう

# ０番の人は、maleに１が入ってるので男。１〜３はfemaleに１が入ってるので女
# 説明変数３つを入れるよ。

# Pclass（階級。１が最高）のようにダミー変数にする必要ない物は、ダミー変数化されないので一緒に入れてしまう

trainXX = pd.get_dummies(train[["Sex","Pclass","SibSp"]])



# 注◆ダミー変数化処理するとNaNが含まれる行は、削除されるらしい→◆get_dummiesの機能
trainXX.head()



# 並び順は、元の並びのPassengerId順（０番の人は下層階級で、兄弟か妻が１人いた、男）
# trainyy = train["Survived"]

trainyy = pd.get_dummies(train[["Survived"]])



# サバイブは、ダミー変数化の意味ないと思うが、念の為説明変数と同様にダミー変数化した
# 必要なライブラリの読み込み（◆読込済の物が重複するが、気にせずコピペ）

import numpy as np

from scipy import sparse

import pandas as pd



from IPython.display import display

import sys
# trainXX.keys()で中の値を見れる

print("key of trainXX: \n{}".format(trainXX.keys()))



# 注◆出力内容がpandas形式（Indexと表示。dictにしたい）。これ、参考のロジ回帰と同じにしたい

# →結局、これは挫折。pandas形式のままでも、ロジスティック回帰は正常動作しそうな為
# 求めたい値、生存の種類の確認（ま、０死亡か、１生存　しかない）

# traiXXには、生存Survived入ってない為、◆trainの中の、Survivedを指定

# 単なる確認

print("生存の種類: \n{}".format(trainyy['Survived']))
# 念の為、データ型を確認。Sexがダミー変数化されてuint8だが、型変換の必要は無いと思う

print('データ型の確認\n{}\n'.format(trainXX.dtypes))
# ◆ここで使用するハイパーパラメータ設定については、下の項目で補足



# まず使用するモデル、ロジスティック　リグレッションをimport

from sklearn.linear_model import LogisticRegression



# ハイパーパラメータ◆このカッコの中の、C以降全てを呼ぶ。パラメータどう入れるかで、良いデータ取れるか変わるのでプロの仕事

logistic_regression = LogisticRegression(C=1.0, random_state=0, multi_class="ovr", penalty="l2")

# 注◆そもそも訓練用データのみ使うわけだが、そいつを訓練用とテスト用に分割する（この関数、デフォルトで７５：２５比率で分ける）

# 変数名の書き方、説明変数Xは大文字で書く、目的変数yは小文字で書く。Xやyを頭につけた変数名を使う慣習ぽい

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(

  trainXX,trainyy,random_state=0

)
# 分離取得したデータの配列の形状を確認します。

# 新規作成◆訓練用データ、行数と列数を求める

print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))
# 新規作成◆testデータ、行数と列数を求める

print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
# 説明変数Xは、特徴量とも呼ばれる。目的変数yは、ラベルとも呼ばれる。（特徴量とは判断、物事を分ける際の、大事な要素。他と分ける為に大きな違いが出る部分）

logistic_regression.fit(X_train,y_train)
## モデルをテストして精度を確認する



# ロジスティックリグレッションというモデルを使って、X_testを　「predict」すると　y予測になる（前項ではfitしたが、今回はpredictしてる）

y_pred = logistic_regression.predict(X_test)



print("Test set predictions: \n {}".format(y_pred))
### 精度計算

# 結果79％

print("Test set score: {:.2f}".format(logistic_regression.score(X_test,y_test)))



# 上記コードの「logistic_regression.score(X_test　」と、ここの処理で予測値（y_predと同じ値）が出る

# つづく「,y_test」　部分が実際の生存者データ。

# 上記２つのデータを比較して、実際の生存者データに対して、正解率が何パーセントか表示されます

# ◆学習時、投入すべきデータは「train」データの全て

# （「train」データには、生存率Survivedという答えも内在する為「教師あり学習」）

# パラメータ設定終わったモデルに対し、説明変数と、目的変数を分けて投入し、学習させる！

logistic_regression.fit(trainXX,trainyy)
# 使う変数、３つを入れるよ。

# Pclass（階級。１が最高）のように元々数値intの物は、ダミー変数化されないので一緒に入れる

# これまでtrainXXは作ったが、testデータを使った「testXX」を作成するのは、初だ

testXX = pd.get_dummies(test[["Sex","Pclass","SibSp"]])
testXX.tail()

# 行数は０から始まるので、最終行が４１７ならば０〜４１７行で、４１８人分のデータだ
## モデルに、testの説明変数を投入し、予測結果（生存するか否かのデータ）を出す



# ロジスティックリグレッションというモデルを使って、testXXを　predictすると　y予測を出力できる

# 目的変数の予測値をyy_predとした。変数名testyyとしたいところだが、目的変数の正確な値ではない、予測値なのでpredを付けた名前にした

yy_pred = logistic_regression.predict(testXX)



print("Test set predictions: \n {}".format(yy_pred))

# 中身見ると、顧客リスト順で、サバイブの予測値が０１で並んでいる。これが回答。あとは、この結果を提出用のcsvファイルに変換する作業がある
# PassengerIDの列を取得してる

PassengerId = np.array(test["PassengerId"]).astype(int)



# 予測結果をパンダス形式の表にします。yy_pred、PassengerIDを並べます。また名前が「Survived」という列を追加しますの処理

# kekkaという変数に結果を入れた。ネーミングが安易な為、どんな処理使ったかわかる名前に変えても良い

kekka5 = pd.DataFrame(yy_pred, PassengerId, columns = ["Survived"])



# csvファイルの作成処理は、こんな感じ。左端の列index_labelは、PassengerIdという名前にします

kekka5.to_csv("kekka5.csv", index_label = ["PassengerId"])



print ("自分プロフィールの、カーネルズタブ、ユアワーク内の作成済カーネル内を見ると、一番下の方にcssデータが表示されている。この画面からsubmit to competitionを押せば提出操作完了し、結果表示まで行ける")

# 処理終わった目印として、文章出力。この処理後、コミットボタン押す必要がありそう