%%time
#  このセル（jupyter でのプログラムの実行単位です。）は、ライブラリを読み込み、sqlite データと接続して、一時的なテーブルを作成します。
# pandas というデータを扱うライブラリを読み込みます。
# このプログラムでは、データ操作はＳＱＬを通じて行うので、pandas は、グラフを書くために使うのが主です。
import pandas as pd
import numpy as np
import pandas.io.sql as psql

# sqliteを読み込むライブラリ
import sqlite3

# 線形回帰　使う頻度はあまりないですが、とりあえず、いれておきます。
from sklearn import linear_model
clf = linear_model.LinearRegression()



# HTMLで表示する　エクセルにコピペするときに便利
from IPython.display import display, HTML

# markdown 用
from tabulate import tabulate

# 日時を扱う
from datetime import datetime as dt
import time

# グラフ
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
%matplotlib inline

# system 関係のライブラリ
import sys
# os の機能を使うライブラリ
import os
# 正規表現
import re

# json,yaml 形式を扱う
import json
import yaml

# 変数の状態を調べる
import inspect

# 文字コード
import codecs

# Web からデータを取得する
import requests

# 貿易統計のデータ
# http://www.customs.go.jp/toukei/info/tsdl_e.htm
# コード　輸出は日本語のみ
# https://www.customs.go.jp/toukei/sankou/code/code_e.htm 

# sqlite に show tables がないので補足するもの
show_tables = "select tbl_name from sqlite_master where type = 'table'"
# describe もないで、補完します。
desc = "PRAGMA table_info([{table}])"
# メモリで、sqlite を使います。kaggle のスクリプト上では、オンメモリでないと新規テーブルがつくれません
# プログラムの一行が長いときは　\　で改行します。
conn = \
    sqlite3.connect(':memory:')

# sql を実行するための変数です。
cursor = conn.cursor()

# 1997 年から、2019 年までの年ベースのデータです。テーブルは、year_from_1997 
# year_from_1997
attach = 'attach "../input/japan-trade-statistics/y_1997.db" as y_1997'
cursor.execute(attach)

# 2018 年の月別集計 テーブル名も ym_2018 
attach = 'attach "../input/japan-trade-statistics/ym_2018.db" as ym_2018'
cursor.execute(attach)

# 2019 年の月別集計 テーブル名も ym_2019
attach = 'attach "../input/japan-trade-statistics/ym_2019.db" as ym_2019'
cursor.execute(attach)

# 2020 年の月別集計 テーブル名も ym_2020
attach = 'attach "../input/japan-trade-statistics/ym_2020.db" as ym_2020'
cursor.execute(attach)

# hs code,country,HSコードです。使いやすいように pandas　に変更しておきます。
attach = 'attach "../input/japan-trade-statistics/codes.db" as code'
cursor.execute(attach)
# import hs,country code as pandas
tmpl = "{hs}_{lang}_df =  pd.read_sql('select * from code.{hs}_{lang}',conn)"
for hs in ['hs2','hs4','hs6','hs6','hs9']:
    for lang in ['jpn','eng']:
        exec(tmpl.format(hs=hs,lang=lang))        

# 国コードも pandas で扱えるようにします。
# country table: country_eng,country_jpn
country_eng_df = pd.read_sql('select * from code.country_eng',conn)
country_eng_df['Country']=country_eng_df['Country'].apply(str)
country_jpn_df = pd.read_sql('select * from code.country_jpn',conn)
country_jpn_df['Country']=country_jpn_df['Country'].apply(str)

# custom  table: code.custom 税関別のコードです
custom_df = pd.read_sql('select * from code.custom',conn)
attach = 'attach "../input/japan-trade-statistics/custom_from_2012.db" as custom_from'
cursor.execute(attach)
attach = 'attach "../input/custom-2016/custom_2018.db" as custom_2018'
cursor.execute(attach)
attach = 'attach "../input/custom-2016/custom_2019.db" as custom_2019'
cursor.execute(attach)

attach = 'attach "../input/japan-trade-statistics/custom_2020.db" as custom_2020'
cursor.execute(attach)

# 計算時間を節約するために、年のデータから、2019 年を切り出します。
# 最初のはエラー処理です。y_2019 というテーブルが存在すると、新規に y_2019 を作ろうとするとエラーになります。
# error の場合は、何もせず、次にすすみます。
try:
    cursor.execute('drop table y_2019')
except:
    pass

# これからが、SQl になります。複数行で書くことが多いのでsql という変数に複数行を代入します。
# 最後の [1:-1] は、一行目（改行で空白）と最後の行（これも改行だけで空白）をとりのぞくためです。
# 0 から始まるので、1 だと、２行目から最後の行のひとつ手前までです。
sql = """
create table y_2019 
as select * from year_from_1997
where Year = 2019
"""[1:-1]
# 上記の sql を実行しして、2019 年のデータをつくります。
cursor.execute(sql)

# sql の説明です。
# create table テーブル名 : テーブルを新規作成　ここでは、y_2019 
# as select * from  テーブル名　: テーブル名(year_from_1997)からつくります。
# where Year = 2019 : 2019 年のデータを指定します。Year は、数値なので、2019 と書きます。


# https://www.customs.go.jp/toukei/srch/index.htm?M=01&P=1,1,,,,,,,,4,1,2019,0,0,0,2,020230100,,,,,,,,,,6,120,,,,,,,,,,,,,,,,,,,,,20

# graph 用の　color　https://matplotlib.org/examples/color/named_colors.htmlsql_sample 
# ym_2018,ym_2019 に ym_2020 を加えて,ym_2018_2020 を作成 
# 年＋月のカラム(ym)をつくります。
# 年は、整数、月は、文字列なので、型変換と文字列結合を行います。
# CAST(Year AS  str )||month as ym がその処理をしている部分です。
# 文字列結合が、|| なので違和感ありますが、しょうがないです。

try:
    cursor.execute('drop table ym_2019_2020')
except:
    pass

sql = """
create table ym_2018_2020
as select CAST(Year AS  str )||month as ym,* from ym_2018
"""[1:-1]
cursor.execute(sql)


sql = """
insert into  ym_2018_2020
 select CAST(Year AS  str )||month as ym,* from ym_2019
"""[1:-1]
cursor.execute(sql)

sql = """
insert into  ym_2018_2020
 select CAST(Year AS  str )||month as ym,* from ym_2020
"""[1:-1]
cursor.execute(sql)
# 計算負荷を減らすために先に 豚肉全体(0203) 輸入(import) を年のテーブル(year_from_1997)から作成する
# 作成する一時テーブル名は、y_hs_0203_import 
# 一時テーブルがあったときは、削除するのが、次の try ～ except までです。
# 
try:
    cursor.execute('drop table y_hs_0203_import')
except:
    pass

# 条件　HS4 豚肉 hs4 = '0203' 輸入 exp_imp=2 で、year_from_1997 からデータを抽出
# 複数行にわたるので、ヒヤドキュメントという記述方法つかっています。""" から、""" までを
# sql に格納します。最後にある[1:-1]は、最初の行と最後の行を除くためです。

sql="""
create table  y_hs_0203_import
as select Year,hs9,hs6,Country,Value
from year_from_1997 
where
exp_imp=2 and
hs4 = '0203' 
"""[1:-1]
cursor.execute(sql)
# 2か国比較用の関数 sql 以下は、規定値　実際に使うときは、適宜規定値を変えてください。
def graph_c2(sql,period='Year',last_p='2019',c2=[['302','Canada','r'],['304','USA','b']]):
    # pandas というpython のデータ形式に、sql の実行結果を格納します。
    df = pd.read_sql(sql,conn) 
    #　グラフで表示がわかりやすくなるために、名前を付与
    df['name'] = ''
    # Country が数字型になっているので、変換しておきます。（データ作成のミスなんでそのうち元データ修正します）
    df['Country'] = df['Country'].astype('str')
    df.loc[df['Country'] == c2[0][0], 'name'] = c2[0][1]
    df.loc[df['Country'] == c2[1][0], 'name'] = c2[1][1]

    # グラフ作成
    plt.figure(figsize=(20, 10))

    ax = sns.lineplot(x=period,y='Value',hue='name',
                      data=df,linewidth=7.0,
                      palette={c2[0][1]: c2[0][2] ,c2[1][1]: c2[1][2]})
    # 年,年月が途中で省略されないため
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) 
    # sql からつくられたデータフレームを返り値にします。
    return(df)
# sql 文のテンプレート　カナダ(302) アメリカ(304) を 比較　変数は、hsコードの部分 {w}がそれです。
# sql_tmpl は、同じ名前をつかいまわしているので注意が必要です。
# 最終的な表示のの指定は、規定値をつかいできるだけ簡単にするのがいいと思っての運用です。
# どういった運用がいいかは人によって判断が異なりますので、適宜変更をしてください。

sql_tmpl="""
select ym,Country,sum(Value) as Value 
from ym_2018_2020
where {w} and
exp_imp = 2 and
Country in ('302','304')
group by ym,Country
"""[1:-1]
# 年月で、豚全体の比較です。USA が逆転です。
w = "hs4='0203'"
sql = sql_tmpl.format(w=w)
df = graph_c2(sql,'ym','202001')
# 次の df をコメントアウトすると,データフレームの内容表示が抑制されます。
# エクセルにコピペして使う場合は、
# df

1+1