import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/digital-transactions-in-india/data.csv')

data
data = data.rename(columns={"Volume (Cr)":"Percentage increase","Value (in Lakh Crores)":"Value(in Billions)"})

data["Value(in Billions)"]*=0.14

data
plt.figure(figsize=(15,6))

plt.title("Transaction done over the given period")

plt.xlabel("Months")

plt.ylabel("Amount (in billions)")

plt.plot(data['Month/Year'],data['Value(in Billions)'])

plt.show()
plt.figure(figsize=(15,6))

plt.title("Percentage increase over the dataset")

plt.xlabel("Month/Year range")

plt.ylabel("Percentage")

plt.plot(data['Month/Year'],data['Percentage increase'])

plt.show()