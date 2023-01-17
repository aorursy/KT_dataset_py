import pandas as pd

import numpy as np

df = pd.read_csv("../input/the-economic-freedom-index/economic_freedom_index2019_data.csv",encoding = "ISO-8859-1")
df.info()
def world_compare(chosen_country):

    country = df.loc[df['Country'] == chosen_country]

    print("  World Economic Freedom values for *", chosen_country,"* compared to worldwide averages.\n")

    for col in country:

        country_val = country.iloc[0][col]

        try:

            world_mean = round(df[col].mean(),2)

        except:

            world_mean = "-"

        print("  ",col, " "*(30-len(col)),country_val," "*(30-len(str(country_val))),world_mean)
world_compare("Ireland")
world_compare("United States")
numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]

non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
print("  Numeric Columns\n  -------------------")

for x in numeric_cols:

    print(" ",x)
print("  Non Numeric Columns\n  -------------------")

for x in non_numeric_cols:

    print(" ",x)
print("  Converting GDP per Capita (PPP) column to numeric.\n  First i will remove the '$' signs and commas.")

print("  Then convert the string column to the integer datatype.\n")

print("Before:")

print(df['GDP per Capita (PPP)'].values)

df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.replace('$', '')

df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.replace(',', '')

print("After:")

print(df['GDP per Capita (PPP)'].values)
print("  The extraneous info.such as '1700 (2015 est.)' will also be removed.\n")

df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.split(' ').str[0]

df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].astype(float)

print(df['GDP per Capita (PPP)'].values)
twelve_measures = ["Property Rights","Judical Effectiveness","Government Integrity","Tax Burden",

  "Gov't Spending","Fiscal Health","Business Freedom","Labor Freedom","Monetary Freedom",

  "Trade Freedom","Investment Freedom ","Financial Freedom"]

print("  Correlation of each of the twelve factors with World Economic Freedom Ranking\n")

for col in twelve_measures:

    print("  ",col," "*(30-len(col)),round(df[col].corr(df["World Rank"]),3))
print("  In general the lower(better) a country's Economic Freedom Ranking the higher its GDP Per Capita.\n")

print("  Correlation =",round(df["World Rank"].corr(df["GDP per Capita (PPP)"]),3))
twelve_measures = ["Property Rights","Judical Effectiveness","Government Integrity","Tax Burden",

  "Gov't Spending","Fiscal Health","Business Freedom","Labor Freedom","Monetary Freedom",

  "Trade Freedom","Investment Freedom ","Financial Freedom"]

print("  Correlation of each of the twelve factors with GDP Per Capita\n")

for col in twelve_measures:

    print("  ",col," "*(30-len(col)),round(df[col].corr(df["GDP per Capita (PPP)"]),3))