import pandas as pd

import numpy as np

df = pd.read_csv("../input/nagano_2018zitensyatou_loc.csv", encoding="cp932")



df.head()
# delete column

del df['罪名']

del df['手口']

del df['管轄警察署（発生地）']

del df['管轄交番・駐在所（発生地）']

del df['市区町村コード（発生地）']

del df['都道府県（発生地）']

del df['市区町村（発生地）']

del df['町丁目（発生地）']

del df['発生場所の属性']



# replace column name with english

df = df.rename(columns={

    '発生年月日（始期）': 'occurDate',

    '発生時（始期）': 'occurTime',

    '被害者の年齢': 'age',

    '被害者の職業': 'occupation',

    '施錠関係': 'isRock',

})



df.head()
# 列名が英字に変換されている事を確認

# 指定した列が削除されている事を確認

df.head()
# isRock(施錠状態)の数値化



# isRockに登録されている値を確認

print("category:", df["isRock"].unique())



df["isRock"][df["isRock"] == "施錠した"] = 1 # reply "施錠した" with 1

df["isRock"][df["isRock"] == "施錠せず"] = 0 # reply "施錠せず" with 1

# int型に型変換

df["isRock"] = df["isRock"].astype(int)

# 数値化されている事を確認

df.head()
# occupation(職業)の数値化

# isRockに登録されている値を確認

print("カテゴリ値の確認", df["occupation"].unique())

df["occupation"][df["occupation"] == "小学生"] = 0

df["occupation"][df["occupation"] == "中学生"] = 1

df["occupation"][df["occupation"] == "高校生"] = 2

df["occupation"][df["occupation"] == "大学生"] = 3

df["occupation"][df["occupation"] == "その他"] = 4

df["occupation"][df["occupation"] == "法人・団体、被害者なし"] = 5

# int型に型変換

df["occupation"] = df["occupation"].astype(int)

# 数値化されている事を確認

df.head()
# age(年齢)の数値化

# ageに登録されている値を確認

print("カテゴリ値の確認", df["age"].unique())

df["age"][df["age"] == "10歳未満"] = 10

df["age"][df["age"] == "10歳代"] = 10

df["age"][df["age"] == "20歳代"] = 20

df["age"][df["age"] == "30歳代"] = 30

df["age"][df["age"] == "40歳代"] = 40

df["age"][df["age"] == "50歳代"] = 50

df["age"][df["age"] == "60-64歳"] = 60

df["age"][df["age"] == "65-69歳"] = 60

df["age"][df["age"] == "70歳以上"] = 70

# 以下の理由から、age = 法人・団体、被害者なし の行を削除

# ・年齢の値としてふさわしくない

# ・1496件中7件しか存在しない

df = df[df["age"] != "法人・団体、被害者なし"]

# int型に型変換

df["age"] = df["age"].astype(int)

# 数値化されている事を確認

df.head()
# occurTime(発生時刻)の数値化

# occurTimeに登録されている値を確認

print("カテゴリ値の確認", df["occurTime"].unique())

# 8件の発生時刻が不明なデータは削除する(中央値で穴埋めしても良いが、ここでは削除)

df = df[df["occurTime"] != "不明"]

# 整数型に型変換する

df['occurTime'] = df['occurTime'].astype(int)

# 数値化されている事を確認

df.head()
# occurDate(発生時刻)の数値化

# occurDateに登録されている値を確認

print("カテゴリ値の確認", df["occurDate"].unique())

# occurDateから occurMonth(月情報) を生成する

df['occurMonth'] = df['occurDate'].str.replace('-', '').str[4:6].astype(int)

# 以下の理由から日にち情報は扱わない

# ・月によって日数が異なる

# ・日にちで件数比較したときに特徴が見つからない

# occurDateは情報の抽出・数値化を完了し用済みのため削除する

del df['occurDate']

# 数値化されている事を確認

df.head()
# 型の確認 全てが数値の型になっている事を確認

df.dtypes
# csvデータの出力

# csv情報として出力

df.to_csv('transform_bohan_data.csv')