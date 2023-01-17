import pandas as pd



USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")
USvideos
USvideos_Sorted = USvideos[["title", "views", "likes"]]

USvideos_Sorted
import seaborn as sns

import matplotlib.pyplot as plt



# Default plot configurations

%matplotlib inline

plt.rcParams['figure.figsize'] = (16,8)

plt.rcParams['figure.dpi'] = 150

sns.set()



from IPython.display import display, Latex, Markdown

plt.xticks(rotation=270) 

sns.scatterplot(x='views', y='likes', data=USvideos_Sorted);