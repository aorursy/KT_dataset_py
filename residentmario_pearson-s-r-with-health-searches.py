import numpy as np



def pearson_r(x, y):

    x_bar, y_bar = np.mean(x), np.mean(y)

    cov_est = np.sum((x - x_bar) * (y - y_bar))

    std_x_est = np.sqrt(np.sum((x - x_bar)**2))

    std_y_est = np.sqrt(np.sum((y - y_bar)**2))

    return cov_est / (std_x_est * std_y_est)
import pandas as pd

searches = pd.read_csv("../input/RegionalInterestByConditionOverTime.csv")

searches.head()
import seaborn as sns

sns.jointplot('2004+cancer', '2017+cancer', data=searches)
import seaborn as sns

sns.jointplot('2016+cancer', '2017+cancer', data=searches)
p_corrs = [pearson_r(searches["{0}+cancer".format(year)], searches["2017+cancer"]) for year 

           in range(2004, 2018)]
p_corrs = [pearson_r(searches["{0}+cancer".format(year)], searches["2017+cancer"]) for year 

           in range(2004, 2018)]

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

pd.Series(p_corrs, index=range(2004, 2018)).plot.line()
df = pd.DataFrame()

for topic in ['cancer', 'cardiovascular', 'stroke', 'depression', 'rehab', 'vaccine', 

              'diarrhea', 'obesity', 'diabetes']:

    p_corrs = [pearson_r(searches["{0}+{1}".format(year, topic)], 

                         searches["2017+{0}".format(topic)]) for year in range(2004, 2018)]

    df[topic] = p_corrs

    

df.index = range(2004, 2018)
df.plot.line(figsize=(12, 6), cmap='viridis', 

             title='Google Medical Searches by Similarity to 2017')