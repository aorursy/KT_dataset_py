import pandas as pd

import matplotlib.pyplot as plt

afe_dataset = pd.read_csv("../input/CAERS_ASCII_2004_2017Q2.csv")

afe_dataset.head()

afe_dataset.count()
# data preparation

# count how often each gender was affected

gender_freq_table = afe_dataset["CI_Gender"].value_counts()



list(gender_freq_table)

labels = list(gender_freq_table.index)

positionForBars = list(range(len(labels)))

plt.bar(positionForBars, gender_freq_table.values)

plt.xticks(positionForBars, labels)

plt.title("Adverse Food Events by Gender")
import seaborn as sns

sns.countplot(afe_dataset["CI_Gender"]).set_title("Adverse Food Events by Gender")