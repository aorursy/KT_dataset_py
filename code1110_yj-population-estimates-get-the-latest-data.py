import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import sleep
# latest data is uploaded here

target_url = "https://dfi-place.west.edge.storage-yahoo.jp/web/report/%E6%9D%B1%E4%BA%AC23%E5%8C%BA%E6%8E%A8%E7%A7%BB0409.xlsx"
# Feb 2020

df2020_02 = pd.read_excel(target_url, sheet_name="2020.02")

print(df2020_02.shape)

df2020_02.head()
# March 2020

sleep(3)

df2020_03 = pd.read_excel(target_url, sheet_name="2020.03")

print(df2020_03.shape)

df2020_03.head()
# April 2020

sleep(3)

df2020_04 = pd.read_excel(target_url, sheet_name="2020.04")

print(df2020_04.shape)

df2020_04.head()