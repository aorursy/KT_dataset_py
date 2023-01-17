# import useful libraries
import numpy as np 
import pandas as pd

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
# Check the encoding of the dataset and then decode it
import chardet
with open("../input/complete.csv", "rb") as f:
    result = chardet.detect(f.read(100000))
print(result)
complete_df = pd.read_csv("../input/complete.csv", encoding="utf-8", index_col=0)
complete_df.head()
# I will merely select some columns that I am interested in
interesting_columns = ["name", "age", "height_cm", "weight_kg", 
                       'eur_value', 'eur_wage', 'eur_release_clause', 
                       'overall', 'potential', 'international_reputation']
df = complete_df[interesting_columns].copy()
# create a column expressing the remaining potential of players
df["remaining_potential"] = df["potential"] - df["overall"]
df.head()
df.sort_values(by="eur_value", ascending=False).head(10)[["name", "eur_value"]]
df.sort_values(by="eur_release_clause", ascending=False).head(10)[["name", "eur_release_clause"]]
top100 = df.sort_values(by="eur_value", ascending=False).head(100)
sns.distplot(top100["eur_value"] / 1e6, 
             kde_kws=dict(cumulative=True))
plt.title("Top 1000 player value distribution")
plt.xlabel("Value (in millions euro)")
plt.ylabel("CDF")
plt.show()
sns.distplot(df["age"], kde=False, fit=stats.gamma)
plt.title("Age distribution of all players")
plt.xlabel("Age")
plt.ylabel("Probability")
plt.show()
cbs = complete_df[complete_df["prefers_cb"] == True]

sns.distplot(cbs["age"], kde=False, fit=stats.gamma)
plt.title("Age distribution of all goalkeepers")
plt.xlabel("Age")
plt.ylabel("Probability")
plt.show()
plt.scatter(df["age"], df["remaining_potential"])
plt.title("Age by remaining potential")
plt.xlabel("Age")
plt.ylabel("Remaining potential")
plt.show()
age = df.sort_values("age")['age'].unique()
remaining_potentials = df.groupby(by="age")["remaining_potential"].mean().values
plt.title("Age vs remaining potential")
plt.xlabel("Age")
plt.ylabel("Remaining potential")
plt.plot(age, remaining_potentials)
plt.show()
overall_skill_reputation = df.groupby(by="international_reputation")["overall"].mean()
potential_skill_reputation = df.groupby(by="international_reputation")["potential"].mean()
plt.plot(overall_skill_reputation, marker='o', c='r', label='Overall Skillpoint')
plt.plot(potential_skill_reputation, marker='x', c='b', label='Potential Skillpoint')
plt.title('Overall, Potential vs Reputation')
plt.xlabel('Reputation')
plt.ylabel('Skill point')
plt.legend(loc='lower right')
plt.show()
