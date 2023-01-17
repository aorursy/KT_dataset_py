import pandas as pd

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/steam123/steam.csv", index_col=0)



df.head()
ownersdeveloper = df[df.owners == "10000000-20000000"]



ownersdeveloper.head()

ownersdeveloper.groupby('developer').size()
plt.figure(figsize=(17,8))

plt.title("Количество негативных отзывов в играх с числом скачиваний 10000000-20000000 человек")

sns.barplot(x=ownersdeveloper['negative_ratings'], y=ownersdeveloper['name'])



plt.show()

ownersdeveloper.to_csv("output.csv", index=True)