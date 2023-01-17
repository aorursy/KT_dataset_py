# ライブラリのインストール

!pip install pyicu

!pip install polygot

!pip install pycld2

!pip install morfessor
# このPython3環境には、多くの役立つ分析ライブラリがインストールされています

# kaggle / pythonドッカーイメージによって定義されます：https://github.com/kaggle/docker-python

# たとえば、ロードするのに役立つパッケージがいくつかあります

import numpy as np

import pandas as pd

import polyglot

from polyglot.text import Text, Word

from polyglot.detect import Detector





# 入力データファイルは、「../ input /」ディレクトリにあります。

# たとえば、これを実行する（実行をクリックするか、Shift + Enterを押す）と、入力ディレクトリの下のすべてのファイルが一覧表示されます

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # ディレクトリ名

        print(dirname)

        # ファイル名

        print(filename)
# ファイルを読み込む

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission_df = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



# 特徴量を把握する

print(train_df.columns.values)

print(test_df.columns.values)

print(submission_df.columns.values)
# データセットをざっと見て考える

train_df.head()
# 欠損値がないか調べる

# 空白やnullやNaNは欠損値と呼ばれ、データが欠けていることを表している。何らかの形で訂正する必要がある。

train_df.tail()
# データセットの情報

# データ数・non-null・形式

train_df.info()

# 区切り線

print('_'*40)

test_df.info()
# 統計量を確認

train_df.describe()
#  統計量を確認

train_df.describe(include=['O'])
!pip install pyicu

!pip install pycld2

!pip install morfessor

!polyglot download embeddings2.en

!polyglot download pos2.en
import pycld2 as cld2

from polyglot.text import Text, Word

import collections



disasterWordList = []

noDisasterWordList = []



# トレインデータのTwitterのテキストの配列をループさせる

for i in range(len(train_df.id.values)):

    textItem = train_df.text.values[i]

    if textItem is None:

        continue

    # UTF-8変換

    textItemUTF8 = ''.join(x for x in textItem if x.isprintable())

    # Polygot変換

    textArrayUTF8Polygot = Text(textItemUTF8)

    if textArrayUTF8Polygot.language.code != "en":

        # 解析不可

        continue

    # 品詞解析

    textArrayUTF8Polygot.pos_tags

    for textItemUTF8Polygot in textArrayUTF8Polygot.words:

        # https://polyglot.readthedocs.io/en/latest/POS.html

        if textItemUTF8Polygot.pos_tag == "NOUN" or textItemUTF8Polygot.pos_tag == "ADJ" or textItemUTF8Polygot.pos_tag == "ADV":

            if textItemUTF8Polygot != "http" and textItemUTF8Polygot != "https" and textItemUTF8Polygot != "amp":

                if train_df.target.values[i] == 0:

                    # 災害ではない

                    noDisasterWordList.append(textItemUTF8Polygot)

                else:

                    disasterWordList.append(textItemUTF8Polygot) 



disasterCount = collections.Counter(disasterWordList)

noDisasterCount = collections.Counter(noDisasterWordList)

# 頻出単語計測

print(disasterCount.most_common())            
# 頻出単語計測

print(noDisasterCount.most_common())            
import csv

import pprint

with open('submission.csv', 'w') as f:

    writer = csv.writer(f)

    writer.writerow(['id', 'target'])

    for i in range(len(test_df.id.values)):

        #  テストデータのTwitterのテキストの配列をループさせる

        textItem = test_df.text.values[i]

        if textItem is None:

            # 解析不可

            writer.writerow([test_df.id.values[i], 0])            

            continue

        # UTF-8変換

        textItemUTF8 = ''.join(x for x in textItem if x.isprintable())

        # Polygot変換

        textArrayUTF8Polygot = Text(textItemUTF8)

        if textArrayUTF8Polygot.language.code != "en":

            # 解析不可

            writer.writerow([test_df.id.values[i], 0])            

            continue

        # 品詞解析

        textArrayUTF8Polygot.pos_tags

        realDisasterScore = 0

        for textItemUTF8Polygot in textArrayUTF8Polygot.words:

            # https://polyglot.readthedocs.io/en/latest/POS.html

            if textItemUTF8Polygot.pos_tag == "NOUN" or textItemUTF8Polygot.pos_tag == "ADJ" or textItemUTF8Polygot.pos_tag == "ADV":

                for countItem in disasterCount.most_common():

                    if countItem[0] == textItemUTF8Polygot:

                        realDisasterScore = realDisasterScore + countItem[1]                    

                for countItem in noDisasterCount.most_common():

                    if countItem[0] == textItemUTF8Polygot:

                        realDisasterScore = realDisasterScore - countItem[1]                        

        result = 1 if realDisasterScore > 0 else 0

        writer.writerow([test_df.id.values[i], result])            
