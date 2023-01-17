# import modules

# インポート

import numpy as np 

import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns
# Read in the data

# データの読み込み

INPUT_DIR = '../input/m5-forecasting-accuracy'

cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
# Examples of events on 'calneder.csv'

# カレンダーに出てくるイベントの例

cal.drop_duplicates(['event_name_1', 'event_name_2'])[1:].head()
# Sporting list

cal_spo = cal[cal['event_type_1'] == 'Sporting']

cal_spo.drop_duplicates(['event_name_1'])[['event_name_1', 'event_type_1']].head()
# Cultural list

cal_cul = cal[cal['event_type_1'] == 'Cultural']

cal_cul.drop_duplicates(['event_name_1'])[['event_name_1', 'event_type_1']].head()
# National list

cal_nat = cal[cal['event_type_1'] == 'National']

cal_nat.drop_duplicates(['event_name_1'])[['event_name_1', 'event_type_1']].head()
# Religious list

cal_rel = cal[cal['event_type_1'] == 'Religious']

cal_rel.drop_duplicates(['event_name_1'])[['event_name_1', 'event_type_1']].head()
# The number of days excpet NaN

# 各項目の欠損値でない日数

print("<All data> \n")

print(cal.count())

eve_rate = cal.count()["event_name_1"] / cal.count()["date"]

print("\n" + "event days rate: " + "{:.02f}".format(eve_rate * 100) + "%")
# test data

# テストデータ

cal[-28:]
print("<Test data> \n")

# The number of days excpet NaN (only in test data)

# 各項目の欠損値でない日数 (only in test data)

print(cal[-28:].count())

eve_rate_28 = cal[-28:].count()["event_name_1"] / cal[-28:].count()["date"]

print("\n" + "event days rate: " + "{:.02f}".format(eve_rate_28 * 100) + "%")
# All data, 全データ

print("<All data> \n")



# Each number of appearance (event_name_1)

# 各イベントの日数 (event_name_1)

print(cal['event_name_1'].value_counts())

print("\n")



# Each number of appearance (event_name_2)

# 各イベントの日数 (event_name_2)

print(cal['event_name_2'].value_counts())

print("\n")



# Test data, テストデータ

print("<Test data> \n")



# Each number of appearance (event_name_1, test data)

# 各イベントの日数 (event_name_1, テストデータ)

print(cal[-28:]['event_name_1'].value_counts())

print("\n")



# Each number of appearance (event_name_2, test data)

# 各イベントの日数 (event_name_2, test data)

print(cal[-28:]['event_name_2'].value_counts())
# All data, 全データ

print("<All data> \n")



# Each number of appearance (event_type_1)

# 各イベントの種類の出現回数 (event_type_1)

print(cal['event_type_1'].value_counts())

print("\n")



# Each number of appearance (event_type_2)

# 各イベントの種類の出現回数 (event_type_2)

print(cal['event_type_2'].value_counts())

print("\n")





# Test data, テストデータ

print("<Test data> \n")



# Each number of appearance (event_type_1)

# 各イベントの種類の出現回数 (event_type_1)

print(cal[-28:]['event_type_1'].value_counts())

print("\n")



# Each number of appearance (event_type_2)

# 各イベントの種類の出現回数 (event_type_2)

print(cal[-28:]['event_type_2'].value_counts())
cal[cal['event_name_1']=='NewYear'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='OrthodoxChristmas'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='MartinLutherKingDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='SuperBowl'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='ValentinesDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='OrthodoxChristmas'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='PresidentsDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='LentStart'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='LentWeek2'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='StPatricksDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Purim End'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Easter'][['event_name_1', 'event_type_1', 'date', 'weekday']]
# 'event_name_2' exists.

cal[cal['event_name_2']=='Easter'][['event_name_2', 'event_type_1', 'date', 'weekday', 'event_name_1']]
cal[cal['event_name_1']=='OrthodoxEaster'][['event_name_1', 'event_type_1', 'date', 'weekday']]
# 'event_name_2' exists.

cal[cal['event_name_2']=='OrthodoxEaster'][['event_name_2', 'event_type_1', 'date', 'weekday', 'event_name_1']]
cal[cal['event_name_1']=='Pesach End'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Cinco De Mayo'][['event_name_1','event_type_1', 'date', 'weekday']]
# 'event_name_2' exists.

cal[cal['event_name_2']=='Cinco De Mayo'][['event_name_2', 'event_type_1', 'date', 'weekday', 'event_name_1']]
cal[cal['event_name_1']=='Mother\'s day'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='MemorialDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='NBAFinalsStart'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='NBAFinalsEnd'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Father\'s day'][['event_name_1', 'event_type_1', 'date', 'weekday']]
# 'event_name_2' exists.

cal[cal['event_name_2']=='Father\'s day'][['event_name_2', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='IndependenceDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Ramadan starts'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Eid al-Fitr'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='LaborDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='ColumbusDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Halloween'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='EidAlAdha'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='VeteransDay'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Thanksgiving'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Christmas'][['event_name_1', 'event_type_1', 'date', 'weekday']]
cal[cal['event_name_1']=='Chanukah End'][['event_name_1', 'event_type_1', 'date', 'weekday']]