import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
donations = pd.read_csv("../input/Donations.csv")
donations.dropna(inplace=True)
donations.head()
print(len(donations.loc[donations['Donation Amount'] < 1000])/len(donations))
print(len(donations.loc[donations['Donation Amount'] >= 10000])/len(donations))
donationsUnder1K = (donations.loc[donations['Donation Amount'] < 1000])['Donation Amount']
fig = plt.figure()
plt.title("Donations under 1K dollars")
ax = plt.gca()
ax.hist(donationsUnder1K,bins=50)
plt.xlabel("Donation Amount")
plt.ylabel("Count")
plt.show()
# Closer look on donations under 200 dolars
donationsUnder200 = (donations.loc[donations['Donation Amount'] < 200])['Donation Amount']
fig = plt.figure()
plt.title("Donations under 200 dollars")
ax = plt.gca()
ax.hist(donationsUnder200,bins=20)
plt.xlabel("Donation Amount")
plt.ylabel("Count")
plt.show()
sum(donationsUnder200)/sum(donations['Donation Amount'])
# Closer look on donations under 1K by their frequency
results, edges = np.histogram(donationsUnder1K, normed=True,bins=20)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth)
plt.title("Probability distribution on donations")
plt.xlabel("Donation Amount")
plt.ylabel("Probability")
plt.show()
# Even closer look on under 200 dollars donations
plt.title("Donation Amount Distribution Under 200 dollars")
ax = plt.gca()
ax.boxplot(donationsUnder200,vert=False)
plt.xlabel("Donation Amount")
plt.show()