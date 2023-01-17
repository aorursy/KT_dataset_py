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
import seaborn as sns;

data_set = pd.read_csv("../input/movie_metadata.csv")

data_set = data_set.dropna()

data_set
#Question 1

data_set['genres'].value_counts()
#Question 2

bs_plot = sns.regplot(x="budget", y="imdb_score", data=data_set)
#Question 2

bs = {'Budget': data_set['budget'], 'IMDB_score': data_set['imdb_score']}

dataframe_budget_score = pd.DataFrame(bs)

dataframe_budget_score.corr('pearson')
#Question 3

bg_plot = sns.regplot(x="budget", y="gross", data=data_set)
#Question 3

bg = {'Budget': data_set['budget'], 'Gross': data_set['gross']}

dataframe_budget_gross = pd.DataFrame(bg)

dataframe_budget_gross.corr('pearson')