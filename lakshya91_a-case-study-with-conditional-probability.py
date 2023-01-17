# Import libraries

import numpy as np

import pandas as pd
# Read the data

df = pd.read_csv("/kaggle/input/kerela-flood/kerala.csv")

df.head()
# Changing the target column to numeric values

df["FLOODS"] = df["FLOODS"].map({"YES": 1, "NO": 0})
# Creating binary data for the months of June and July using the rainfall threshold

df["JUN_GT_500"] = (df["JUN"] > 500).astype("int")

df["JUL_GT_500"] = (df["JUL"] > 500).astype("int")

df_small = df.loc[:, ["YEAR", "JUN_GT_500", "JUL_GT_500", "FLOODS"]]

df_small["COUNT"] = 1

df_small.head()
df_small.shape
# Creating the tabular data based on the counts

pd.crosstab(df_small["FLOODS"], df_small["JUN_GT_500"])
P_F = (6 + 54) / (6 + 54 + 19 + 39)

P_J = (39 + 54) / (6 + 54 + 19 + 39)

P_F_intersect_J = 54 / (6 + 54 + 19 + 39)

print(f"P(F): {P_F}") 

print(f"P(J): {P_J}")

print(f"P(F AND J): {P_F_intersect_J}")
# Now calculate probailitity of flood given it rained more than 500 mm in June (P(A|B))

P_F_J = P_F_intersect_J / P_J

print(f"P(F|J): {P_F_J}")
# Probability of rain more than 500 mm in June given it flooded that year (P(B|A))

P_J_F = (P_F_J * P_J) / P_F

print(f"P(J|F): {P_J_F}")
# We can similarly do it for july

pd.crosstab(df_small["FLOODS"], df_small["JUL_GT_500"])
P_F = (3 + 57) / (3 + 57 + 19 + 39)

P_J = (39 + 57) / (3 + 57 + 19 + 39)

P_F_intersect_J = 57 / (3 + 57 + 19 + 39)

print(f"P(F): {P_F}") 

print(f"P(J): {P_J}")

print(f"P(F AND J): {P_F_intersect_J}")
# Now calculate probailitity of flood given it rained more than 500 mm in July

P_F_J = P_F_intersect_J / P_J

print(f"P(F|J): {P_F_J}")
# Probability of rain more than 500 mm in July given it flooded that year (P(B|A))

P_J_F = (P_F_J * P_J) / P_F

print(f"P(J|F): {P_J_F}")