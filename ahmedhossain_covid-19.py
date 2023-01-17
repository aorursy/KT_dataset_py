

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
sns.set()
df = pd.read_excel("/kaggle/input/covid-19-dataset-of-the-world-upto-august-22-2020/owid-covid-data.xlsx")
df.head()
df_bd = df[df["location"] == "Bangladesh"]
df_bd.head()
df_bd.info()
df_bd.drop(["iso_code", "continent"], axis= 1, inplace = True)
df_bd.shape
df_bd.head()
df_bd["date"] = pd.to_datetime(df_bd['date'], format='%Y-%m-%d')
df_bd["month"] = df_bd["date"].dt.month
df_bd["month"]
look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

df_bd['month'] = df_bd['month'].apply(lambda x: look_up[x])
df_bd["ratio"] = df_bd["new_cases"] / df_bd["new_tests"]
df_bd.month
import matplotlib.dates as mdates
fig, ax = plt.subplots(3,1,figsize= (16,18))
ax[0].plot(df_bd["date"], df_bd["new_cases"], marker='o', linestyle='-')
ax[1].plot(df_bd["date"], df_bd["new_tests"], marker='o', linestyle='-')
ax[2].plot(df_bd["date"], df_bd["new_deaths"], marker='o', linestyle='-')
ax[0].set_ylabel('Counts of new cases')
ax[1].set_ylabel('Counts of new tests')
ax[2].set_ylabel('Counts of new deaths')
ax[0].set_title('New cases per day')
ax[1].set_title('New tests per day')
ax[2].set_title('New deaths per day')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig, ax = plt.subplots(figsize= (16,6))
ax.plot(df_bd["date"], df_bd["ratio"], marker='o', linestyle='-')
ax.set_ylabel('Ratio of new cases/ new tests')
ax.set_title('Ratio of new cases/ new tests per day')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
month = ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]
cases_by_month = []
tests_by_month = []
deaths_by_month = []
for i in month:
    x = df_bd[df_bd["month"] == i]["new_tests"].sum()
    y = df_bd[df_bd["month"] == i]["new_cases"].sum()
    z = df_bd[df_bd["month"] == i]["new_deaths"].sum()
    tests_by_month.append(x)
    cases_by_month.append(y)
    deaths_by_month.append(z)

plotting = pd.DataFrame({"Months": month, "Tests by Month": tests_by_month, "Cases by Month": cases_by_month, "Deaths by Month": deaths_by_month})
plotting
plotting.plot(x = "Months", y = ["Tests by Month", "Cases by Month", "Deaths by Month"], kind = "bar")
sns.barplot(x = plotting["Months"], y = plotting ["Deaths by Month"])
