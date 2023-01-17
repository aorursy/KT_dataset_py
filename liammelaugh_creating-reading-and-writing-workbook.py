import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
df = pd.DataFrame({"Apples":30,"Bananas":21}, index=[0])
df
pd.DataFrame({"Apples":[35,41], "Bananas":[21,34]}, index=["2017 Sales","2018 Sales"])
pd.Series(("4 cups","1 cup","2 large","1 can"), index=["Flour","Milk","Eggs","Spam"], name="Dinner")
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")