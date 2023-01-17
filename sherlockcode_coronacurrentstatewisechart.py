import pandas as pd
import matplotlib as plt
import seaborn as sns

from matplotlib import pyplot

#always include r infront of the file path so that it hits the root dir directly

df = pd.read_excel(r"C:\Users\Anik Chatterjee\Untitled Folder\Kegal Datasets\Covid19_india\StatewiseTestingDetails.xlsx")
#need to find out a way to display the the data frame without .........
#df
#displays the first 5 rows 
#df.head()

#need to refine the data

total_statewise_sample_captured = df.groupby("State")["TotalSamples"].sum()
total_statewise_sample_captured

#plot a graph for the statewise samples taken

plt.pyplot.figure(figsize=(80,40))
x1 = plt.pyplot.xlabel('State', size = 50)
x1 = df['State']


y2 = plt.pyplot.ylabel('Samples', size = 50)
y2 = df['TotalSamples']

sns.set(font_scale=2.5)

plt.pyplot.bar(x1, y2, width = 0.9, color = ['red'])
plt.pyplot.savefig("Samples_statewise.jpg")
#total_Negative = df["TotalSamples"].sum
#total_Negative

total_Positive_statewise_captured = df.groupby("State")["Positive"].sum()
total_Positive_statewise_captured

plt.pyplot.figure(figsize=(80,40))
x1 = plt.pyplot.xlabel('State', size = 50)
x1 = df['State']


y2 = plt.pyplot.ylabel('Positive', size = 50)
y2 = df['Positive']

sns.set(font_scale=2.5)

plt.pyplot.bar(x1, y2, width = 0.9, color = ['blue'])
