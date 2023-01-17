import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

my_filepath = "../input/ebolaaa/ebola.csv"
df = pd.read_csv(my_filepath)
df.head()
ebola_сonfirmed = df[df.Indicator == "Cumulative number of confirmed Ebola cases"]
ebola_сonfirmed.head()
date_data = ebola_сonfirmed.groupby(['Date']).sum()

date_data.head()
plt.title("Общее количество зараженных")
#print(date_data.index)
max_x=date_data.loc[['2014-08-29', '2014-09-05', '2014-09-08', '2014-09-12', '2014-09-16']]
#print(max_x)
sns.barplot(x=max_x.index, y=max_x['value'])
#plt.show()
date_data.to_csv("date_data.csv", index=True)