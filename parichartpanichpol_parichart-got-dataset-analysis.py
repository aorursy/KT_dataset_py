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
import numpy as np # Basic linear algebra functions

import pandas as pd # Structured data operations and manipulations

import seaborn as sn # Statistical data visualization

import matplotlib.pyplot as plt # Python defacto plotting library

%matplotlib inline 

import warnings
data_battles = pd.read_csv("../input/battles.csv")

data_deaths = pd.read_csv("../input/character-deaths.csv")

data_predictions = pd.read_csv("../input/character-predictions.csv")
data_battles
data_battles.shape
data_deaths
data_deaths.shape
data_predictions
data_predictions.shape
df = pd.DataFrame(data_deaths)

deaths = df.dropna() #remove NaN
deaths
deaths.shape
deaths.columns
deaths.head(10)
deaths.describe()
deaths.corr()
# Set up the matplotlib figure

plt.rcParams['figure.figsize']=[12.0, 10.0]

plt.title('Pearson Correlation')

# Draw the heatmap using seaborn

sn.heatmap(deaths.corr(),linewidths=0.5,vmax=1.0, square=True, cmap="coolwarm", linecolor='black', annot=True)
deaths.boxplot(column='Death Chapter', by = 'Gender')