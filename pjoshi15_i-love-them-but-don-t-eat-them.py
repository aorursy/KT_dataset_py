import pandas as pd
import numpy as np
import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
%matplotlib inline
choc_df = pd.read_csv("../input/flavors_of_cacao.csv")
choc_df.head()
choc_df['Rating'].plot.hist(bins = 16)
g = sns.FacetGrid(choc_df, col = 'Review\nDate', col_wrap=4, size=2.5)
g = g.map(plt.hist, "Rating")
choc_df.isnull().sum()
choc_df.dropna(axis=0, how='any', inplace=True)
choc_df['Cocoa\nPercent'] = choc_df['Cocoa\nPercent'].replace("%","", regex = True)
choc_df['Cocoa\nPercent'] = choc_df['Cocoa\nPercent'].apply(pd.to_numeric, errors = 'coerce')
choc_df.head()
sns.jointplot(x = "Rating", y = "Cocoa\nPercent", data = choc_df)
p = sns.FacetGrid(choc_df, col = 'Review\nDate', col_wrap=4, size=2.5)
p = p.map(plt.scatter, "Rating", "Cocoa\nPercent")
sns.boxplot(x = 'Review\nDate', y = 'Cocoa\nPercent', data = choc_df)
