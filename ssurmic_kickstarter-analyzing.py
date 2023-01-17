# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np 

import pandas as pd 

import warnings 

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

live_project = pd.read_csv("../input/live.csv")# the live dataset is now a Pandas DataFrame

most_backed_project = pd.read_csv("../input/most_backed.csv")

most_backed_project.head(10)

#backed_data=most_backed_project.loc[:,"amt.pledged":"num.backers"]#narrow down our focus

#backed_data.head()

most_backed_project.plot(x='goal',y='num.backers',kind='Scatter')

sb.jointplot(x="goal", y="num.backers", data= most_backed_project, size=7, kind='reg',color='red')

#live_project.head(10)

#print('Correlation Matrix')

#live_project.corr()

sb.boxplot(x="goal", y="num.backers", data= most_backed_project,orient="v",saturation=0.8, palette='Greens')

sb.swarmplot(x="goal", y="num.backers", data= most_backed_project,orient="v",palette='Reds')

sb.stripplot(x="goal", y="num.backers", data= most_backed_project,orient="v",palette='Blues')



# Any results you write to the current directory are saved as output.