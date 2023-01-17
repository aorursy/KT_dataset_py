#読み込んだデータを表示するだけのテスト

#以下のデータを読み込んで表示させています
#Video Game Sales  https://www.kaggle.com/gregorut/videogamesales

import pandas as pd

df = pd.read_csv("../input/vgsales.csv")

#最初の5件を表示
df.head()
#最初の1件を表示
df.head(1)
#最初の100件を表示
df.head(100)
#全件を表示
#件数が多すぎると途中の表示は省略される？
df