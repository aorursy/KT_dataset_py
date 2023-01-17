import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/science-olympiad-2019-results/2019-nats-divc-results.csv')



plt.figure(figsize=(75, 25))



sns.barplot(x=data['Team'], y=data['Total'])
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/science-olympiad-2019-results/2019-nats-divc-results.csv')



data = data[0:15]



plt.figure(figsize=(75, 25))



sns.barplot(x=data['Team'], y=data['Total'])
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/science-olympiad-2019-results/2019-nats-divc-results.csv')



data = data[data.columns[2:28]]



corr = data.corr() # this generates a correlation matrix for the data



plt.figure(figsize=(75, 25))



sns.heatmap(data=corr, annot=True, cmap="YlGnBu") # annot allows us to see the value in the cell, and as for cmap... I like the yellow-green-blue colorscheme lol 
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/science-olympiad-2019-results/2019-nats-divc-results.csv')



plt.figure(figsize=(75, 25))



diff = data['Thermodynamics'] - data['Overall']



sns.barplot(x=data['Overall'], y=diff) # we'll use the overall placements as the x-axis because it's cleaner and we don't have to count all the bars