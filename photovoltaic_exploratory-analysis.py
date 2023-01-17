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
import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/SalesKaggle3.csv')
### pre-processing



#historical only

df_historical = df.loc[df['File_Type']=='Historical']



# making SoldFlag categorical (str)

df_historical['SoldFlag'] = df_historical['SoldFlag'].astype(int).astype(str)
f, axes = plt.subplots(6, 1, figsize=(10,20), sharex=True)

f.suptitle('Exploratory screening of Inputs by Marketing Type', fontsize=17)

sns.boxplot(x='MarketingType', y='PriceReg', hue="SoldFlag", data=df_historical, palette="PRGn", ax=axes[0])

axes[0].set_ylim(0,1000)

sns.boxplot(x='MarketingType', y='ReleaseNumber', hue="SoldFlag", data=df_historical, palette="PRGn", ax=axes[1])

axes[1].set_ylim(0,50)

sns.boxplot(x='MarketingType', y='StrengthFactor', hue="SoldFlag", data=df_historical, palette="PRGn", ax=axes[2])

axes[2].set_yscale('log')

sns.boxplot(x='MarketingType', y='ItemCount', hue="SoldFlag", data=df_historical, palette="PRGn", ax=axes[3])

axes[3].set_ylim(0,300)

sns.boxplot(x='MarketingType', y='LowUserPrice', hue="SoldFlag", data=df_historical, palette="PRGn", ax=axes[4])

axes[4].set_ylim(0,250)

sns.boxplot(x='MarketingType', y='LowNetPrice', hue="SoldFlag", data=df_historical, palette="PRGn", ax=axes[5])

axes[5].set_ylim(0,250)
corr = df_historical[['PriceReg','ReleaseNumber','StrengthFactor','ItemCount','LowUserPrice','LowNetPrice']].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(200, 1, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, square=True, linewidths=.5, annot=True, ax=ax)