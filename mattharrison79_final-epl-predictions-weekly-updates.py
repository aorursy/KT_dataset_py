import pandas as pd

Week_2_table = pd.read_csv('../input/predicted-tables/predicted_table_week2.csv')

Week_2_table = Week_2_table.reset_index(drop=True)

Week_2_table = Week_2_table.drop(['Positional difference'], axis=1)

print(Week_2_table)
Week_3_table = pd.read_csv('../input/week3table/predicted_table_week3.csv')

Week_3_table = Week_3_table.reset_index(drop=True)

Week_3_table = Week_3_table.drop(['Positional difference'], axis=1)

Week_3_table = Week_3_table.round(3)

print(Week_3_table)
Week_4_table = pd.read_csv('../input/week4table/predicted_table_week4.csv')

Week_4_table = Week_4_table.reset_index(drop=True)

Week_4_table = Week_4_table.drop(['Positional difference'], axis=1)

Week_4_table = Week_4_table.round(3)

print(Week_4_table)
Week_5_table = pd.read_csv('../input/week5table2/predicted_table_week5.csv')

Week_5_table = Week_5_table.reset_index(drop=True)

Week_5_table = Week_5_table.drop(['Positional difference'], axis=1)

Week_5_table = Week_5_table.round(3)

print(Week_5_table)
Week_6_table = pd.read_csv('../input/week6table/predicted_table_week6.csv')

Week_6_table = Week_6_table.reset_index(drop=True)

Week_6_table = Week_6_table.drop(['Positional difference'], axis=1)

Week_6_table = Week_6_table.round(3)

print(Week_6_table)
Week_7_table = pd.read_csv('../input/week7table/predicted_table_week7.csv')

Week_7_table = Week_7_table.reset_index(drop=True)

Week_7_table = Week_7_table.drop(['Positional difference'], axis=1)

Week_7_table = Week_7_table.round(3)

print(Week_7_table)
Week_8_table = pd.read_csv('../input/week8table/predicted_table_week8.csv')

Week_8_table = Week_8_table.reset_index(drop=True)

Week_8_table = Week_8_table.round(3)

print(Week_8_table)
Week_9_table = pd.read_csv("../input/week9table/predicted_table_week9.csv")

Week_9_table = Week_9_table.reset_index(drop=True)

Week_9_table = Week_9_table.round(3)

print(Week_9_table)
Week_10_table = pd.read_csv("../input/week10to12table/predicted_table_week10.csv")

Week_10_table = Week_10_table.reset_index(drop=True)

Week_10_table = Week_10_table.round(3)

print(Week_10_table)