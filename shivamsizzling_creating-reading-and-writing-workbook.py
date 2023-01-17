import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
dict1 = {"Apples": [30], "Bananas": [21]}

#print(dict1)
df = pd.DataFrame(dict1)
print(df)
check_q1(df)
dict2 = {"Apples": [35,41], "Bananas": [21,34]}

#print(dict1)
df = pd.DataFrame(dict2, index= ["2017 Sales", "2018 Sales"])
print(df)
check_q2(df)
list1 = ["4 cups","1 cup", "2 large", "1 can"]
Di = pd.Series(list1, index=["Flour","Milk","Eggs","Spam"], name="Dinner")
Di
check_q3(Di)
wines = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
wines.columns
wines = wines.drop(columns='Unnamed: 0')

check_q4(wines)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
#check_q5(q6_df)
# Your Code Here