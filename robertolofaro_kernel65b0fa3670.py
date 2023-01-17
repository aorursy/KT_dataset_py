# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization 

%matplotlib inline
content_df = pd.read_csv("../input/sdgeu-datamart-sample/sdgeu_datamart_kaggle.csv")
content_df_presentation = content_df[["SeriesCode","TimePeriod","Value"]]

content_df_presentation.set_index("SeriesCode")

content_df_pivot = content_df_presentation.groupby(["SeriesCode","TimePeriod"]).count().pivot_table(values='Value', index='TimePeriod', columns='SeriesCode').reset_index()

content_df_pivot.set_index("TimePeriod")
plt.figure(figsize=(10, 10))

selected_indicator = "IT_USE_ii99" 

sns.boxplot(x='SeriesCode',y='Value',data=content_df[(content_df['SeriesCode']==selected_indicator)&(content_df['TimePeriod']==2016)],palette='winter')
plt.figure(figsize=(10, 10))

sns.heatmap(content_df_pivot.isnull(),cbar=False, cmap="winter")
selected_indicator = "IT_USE_ii99" 

# a period where data are present within the raw data

content_df[(content_df['TimePeriod']==2017)&(content_df['SeriesCode']==selected_indicator)]
# a period where data are missing within the raw data for 

content_df[(content_df['TimePeriod']==2018)&(content_df['SeriesCode']==selected_indicator)]