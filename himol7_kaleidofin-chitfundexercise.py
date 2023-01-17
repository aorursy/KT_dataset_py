import pandas as pd
data = pd.read_excel("../input/chit fund exercise.xlsx")
data.head()
data.shape
data["Actual Contribution"] = data["Contribution"] - data["Amount returned to everyone in the group"]
data.head()
total = data["Actual Contribution"].sum()
total
data["return"] = data["Net amount recd by Bid winner"] - total
data.head()
data["ret%"] = (data["return"]/total)*100
data.head()
data.iloc[24]
((((1 + (data["ret%"][24])/100)) ** (12/25)) - 1) * 100
data.iloc[0]
((((1 + (data["ret%"][0])/100)) ** (12/25)) - 1) * 100
data["ret%"]
import seaborn as sns
sns.lineplot(data["Month"], data["ret%"])
import matplotlib.pyplot as plt
plt.plot(data["Month"], data["ret%"], color = 'red')
plt.plot(data["Month"], data["Net amount recd by Bid winner"]/1000, color = 'blue')
plt.legend(("Return %", "Net Amount Received by Bid Winner in thousands"))
