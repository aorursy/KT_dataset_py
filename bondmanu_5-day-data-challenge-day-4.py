import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt        # for plotting

import seaborn as sns

sns.set(style="darkgrid")

data = pd.read_csv("../input/anonymous-survey-responses.csv")

# look at the first few rows

data.head()
# sns countplot for categorical couting and plotting

ax = sns.countplot(x="Just for fun, do you prefer dogs or cat?", 

                   data=data, order = 

                   data["Just for fun, do you prefer dogs or cat?"].value_counts().index)

# used value_counts().index to sort the count

ax.set_title("Pet preferences")

# plt.title("Pet preferences")   # we can use plt.title as well