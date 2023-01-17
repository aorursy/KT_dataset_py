import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()
# Encoding for Brazilian : https://community.atlassian.com/t5/Jira-questions/File-encoding-for-Portuguese-Brazilian/qaq-p/680609

df =  pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding="ISO-8859-1")

df.head()
df.shape
df.isna().sum()
# converting `number` to int

df.number = df.number.astype('int')



# replacing the month column with its numeric version

df["month"] = df["month"].map({'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6, 'Julho': 7,

       'Agosto': 8, 'Setembro': 9, 'Outubro':10, 'Novembro':11, 'Dezembro':12}).astype('category')
df.head()
fires_by_year = df[["year", "number"]].groupby("year").sum()

fires_by_year.reset_index(inplace=True)



plt.figure(figsize=(18,6))

sns.lineplot(x="year", y="number", data=fires_by_year)

plt.xticks(fires_by_year["year"])

plt.title("Number of forest fires each year")

plt.show()
fires_by_month =  df[["month", "number"]].groupby("month").sum()

fires_by_month.reset_index(inplace=True)



plt.figure(figsize=(18,6))

sns.lineplot(x="month", y="number", data=fires_by_month)

plt.xticks(fires_by_month["month"])

plt.title("Number of forest fires by month")

plt.show()
fires_by_state =  df[["state", "number"]].groupby("state").sum()

fires_by_state.reset_index(inplace=True)



plt.figure(figsize=(18,6))

sns.lineplot(x="state", y="number", data=fires_by_state)

plt.xticks(fires_by_state["state"], rotation="vertical")

plt.title("Number of forest fires by state")

plt.show()
top_10_states = fires_by_state.sort_values(["number"], ascending=False)[0:10]["state"]

fires_in_top_10_states = df[df.state.isin(top_10_states)][["state", "year", "number"]]

fires_in_top_10_states_by_year = fires_in_top_10_states.groupby(["state","year"]).sum().reset_index()
plt.figure(figsize=(18,6))

sns.lineplot(x="year", y="number", hue="state", data=fires_in_top_10_states_by_year)

plt.xticks(fires_in_top_10_states_by_year["year"])

plt.title("Top 10 states with forest fires over the years")

plt.legend(loc="upper left")

plt.show()