
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
cases = pd.read_csv("../input/worldwideaggregated-csv-apr16csv/worldwide-aggregated_csv_Apr16.csv")
cases.head()
cases["Date"] = pd.to_datetime(cases["Date"])
cases.info()
cases.head()
# adding new rate of change columns
cases["Deaths_change"] = cases["Deaths"].pct_change()*100
cases["Recovered_change"] = cases["Recovered"].pct_change()*100
cases.rename(columns={"Deaths_change": "Increase Rate (Deaths)", 
                      "Recovered_change": "Increase rate (Recovered)",
                     "Increase rate": "Increase rate(Confirmed)"}, inplace= True)
cases.tail()

fig, axs = plt.subplots(figsize=(15,5))

plt.plot("Date", "Deaths", data = cases, color='red', marker='o', linestyle='dashed', linewidth=2)
plt.plot("Date", "Recovered", data = cases, color='green', marker='o', linestyle='dashed', linewidth=2)
plt.plot("Date", "Confirmed", data = cases, color='yellow', marker='o', linestyle='dashed', linewidth=2)

plt.legend()


# rate of Increase in death, recovered and confirmed  cases compared to the precious day

fig, axs = plt.subplots(figsize=(15,5))

plt.plot("Date", "Increase Rate (Deaths)", data = cases, color='red', marker='o', linestyle='dashed', linewidth=2)
plt.plot("Date", "Increase rate (Recovered)", data = cases, color='green', marker='o', linestyle='dashed', linewidth=2)
plt.plot("Date", "Increase rate(Confirmed)", data = cases, color='yellow', marker='o', linestyle='dashed', linewidth=2)

plt.legend()

