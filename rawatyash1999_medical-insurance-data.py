import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('/kaggle/input/insurance.csv')
df.head()
df.dtypes
df.describe()
df.isnull().sum()
df.tail()
df.shape
sns.distplot(df['bmi'])
sns.distplot(df['bmi'])
import numpy as np
df["weight_condition"] = np.nan

lst = [df]



for col in lst:

    col.loc[col["bmi"] < 18.5, "weight_condition"] = "Underweight"

    col.loc[(col["bmi"] >= 18.5) & (col["bmi"] < 24.986), "weight_condition"] = "Normal Weight"

    col.loc[(col["bmi"] >= 25) & (col["bmi"] < 29.926), "weight_condition"] = "Overweight"

    col.loc[col["bmi"] >= 30, "weight_condition"] = "Obese"

    
df
pd.crosstab(df["sex"],df["region"],margins=True)
df.loc[(df["sex"]=="male") & (df["smoker"]=="yes") & (df["region"]=="southwest"), ["sex","smoker","region"]].head(10)
df_sorted = df.sort_values(['smoker','region'], ascending=False)

df_sorted[['smoker','region']].head(10)



impute_grps = df.pivot_table(values=["charges"], index=["sex","smoker"], aggfunc=np.mean)
impute_grps
df.boxplot(column="charges",by="region", figsize=(18, 8))
df.plot.scatter(x='charges', y='bmi', figsize=(18, 8))
# Area Plot

df_new = df.drop(columns = 'charges') #dropping charges for the plot

df_new.plot.area(figsize=(18, 8))
df['charges'].plot(kind='kde')
from sklearn.preprocessing import LabelEncoder

#sex

le = LabelEncoder()

le.fit(df.sex.drop_duplicates()) 

df.sex = le.transform(df.sex)

# smoker or not

le.fit(df.smoker.drop_duplicates()) 

df.smoker = le.transform(df.smoker)

#region

le.fit(df.region.drop_duplicates()) 

df.region = le.transform(df.region)
df.corr()['charges'].sort_values()