# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/movie_metadata.csv")
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline  
sns.distplot(df['imdb_score'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
cor_dict = corr['imdb_score'].to_dict()

del cor_dict['imdb_score']

print("List the numerical features decendingly by their correlation with Sale Price:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(float(x[1]))):

    print("{0}: \t{1}".format(*ele))