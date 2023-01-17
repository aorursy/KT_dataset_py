import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('/kaggle/input/us-major-league-soccer-salaries/mls-salaries-2017.csv')

df
df.head(10)
df.index.size # with index size
len(df.index) # with len function
df["base_salary"].sum() / df.index.size # dangerously-calculated average
def get_average_salary(dataframe):
    cl = "base_salary"
    missing_value_count = dataframe[cl].isnull().sum()
    count_of_valid_value = df[cl].size - missing_value_count
    total_of_valid_value = dataframe[cl].dropna().sum()
    return total_of_valid_value / count_of_valid_value

get_average_salary(df) # safe-calculated average
df["base_salary"].mean() # usual way for getting average
df["first_name"].isnull().sum() # count of missing values for first_name
df["base_salary"].isnull().sum() # count of missing values for base_salary
df["base_salary"].max()
df.iloc[df["base_salary"].argmax()] # by index location
df.iloc[[df["base_salary"].argmax()]] # by index location (Attention! Check difference with previous approach.)
df[df["base_salary"] == df["base_salary"].max()] # by filter
df.iloc[df["guaranteed_compensation"].argmax()]
df.iloc[df["guaranteed_compensation"].argmax()]["last_name"]
df[(df["guaranteed_compensation"] == df["guaranteed_compensation"].max())] # Alternative solution to find max guaranteed compansation
df[(df["guaranteed_compensation"] == df["guaranteed_compensation"].max())]["last_name"]
df[(df["guaranteed_compensation"] == df["guaranteed_compensation"].max())]["last_name"].iloc[0]
df[(df["first_name"] == "Will") & (df["last_name"] == "Johnson")] # Row for Will Johnson
df[(df["first_name"] == "Will") & (df["last_name"] == "Johnson")]["position"]
df[(df["first_name"] == "Will") & (df["last_name"] == "Johnson")]["position"].iloc[0]
df.groupby("position")["base_salary"].mean()
len(df.groupby("position")) # with groupby and len function
df["position"].nunique() # with nqunique function
df["club"].value_counts()
df["position"].value_counts()
def starts_with(first_name, letters):
    if first_name is np.nan:
        return False
    elif first_name.lower().startswith(letters.lower()):
        return True
    else:
        return False
df["first_name"].apply(starts_with, letters="pa") # Applying function to first_name column to filter
df[df["first_name"].apply(starts_with, letters="pa")]