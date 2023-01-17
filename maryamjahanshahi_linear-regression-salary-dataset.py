import pandas as pd

import statsmodels.formula.api as sm

df = pd.read_csv("../input/salary-dataset/salary_dataset.csv")
df.head()
pd.get_dummies(df["gender"]) 
pd.get_dummies(df["dept"]) 
df["gender_male"] = pd.get_dummies(df["gender"])["Male"]
df_new = pd.get_dummies(df, columns = ["gender", "jobTitle", "edu", "dept"], drop_first=True)

df_new.head()
ignore_variables = ["bonus", "basePay"]
input_variables = [x for x in df_new.columns if x not in ignore_variables]

input_variables
input_variables_formula = " + ".join(input_variables)

input_variables_formula
reg_model = sm.ols(formula= f"basePay ~ {input_variables_formula}", data=df_new).fit()
df.replace(" ", "_", regex=True, inplace=True)
df.head()
df_new = pd.get_dummies(df, columns = ["gender", "jobTitle", "edu", "dept"], drop_first=True)

df_new.head()
input_variables = [x for x in df_new.columns if x not in ignore_variables]

input_variables_formula = " + ".join(input_variables)

input_variables_formula
reg_model = sm.ols(formula= f"basePay ~ {input_variables_formula}", data=df_new).fit()
reg_model.summary()