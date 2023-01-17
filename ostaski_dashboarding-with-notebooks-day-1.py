import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
# output of plotting commands is displayed inline
%matplotlib inline
import matplotlib.pyplot as plt # scatterplot tools
import seaborn as sns # boxplot tools

# taking a look at input's content
import os
print(os.listdir("../input"))

# read Competitions.csv into a data frame
df = pd.read_csv("../input/Competitions.csv")

# print the first 3 rows
print(df.head(3))

# summarize the data
df.describe(include = 'all')

# create a scatterplot with TotalCompetitors and NumScoredSubmissions
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['TotalCompetitors'], df['NumScoredSubmissions'])
plt.title('TotalCompetitors and NumScoredSubmissions Distribution')
plt.xlabel('TotalCompetitors')
plt.ylabel('NumScoredSubmissions')
plt.show()

# create a boxplot with TotalCompetitors and NumScoredSubmissions
sns.set(style="ticks", palette="pastel")
sns.boxplot(x = df['TotalCompetitors'], y = df['NumScoredSubmissions'])
sns.despine(offset=10, trim=True)