import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
DATA_FILE='../input/crime.csv'

df = pd.read_csv(DATA_FILE)

df.head(4)
df.OFFENSE_CATEGORY_ID.value_counts().plot(kind='barh')
df.NEIGHBORHOOD_ID.value_counts()[:10].plot(kind='barh', title='High Crime Occurance')
df.NEIGHBORHOOD_ID.value_counts()[-10:].plot(kind='barh', title='Low Crime Occurance')