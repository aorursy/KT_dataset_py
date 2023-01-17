# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/metal-by-nation/metal_bands_2017.csv',encoding='latin-1')
data.head(2)
data.shape
data["fans"]
data.dtypes
data.columns[data.isna().any()]
data["origin"].isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
dataset = data.iloc[:, 1:7]
dataset.shape
dataset.band_name.duplicated().sum()
dataset.loc[dataset.band_name.duplicated(keep='first'), :]
dataset.loc[dataset.band_name.duplicated(keep='last'), :]
new_data = dataset.drop_duplicates(keep='first')
new_data.shape
new_data.info()
new_data["origin"].fillna("unknown", inplace=True)
for i in new_data["style"].str.split(','):
    i = i[0]
    print(i)
styles = new_data["style"].str.split(",", expand=True)
new_data['main_style'] = styles[0]
new_data['style_2'] = styles[1]
new_data['style_3'] = styles[2]
new_data['style_4'] = styles[3]
new_data['style_5'] = styles[4]
new_data['style_6'] = styles[5]
new_data['main_style'].value_counts().head()
new_data.head()
new_data.drop("style", axis=1, inplace=True)
new_data.head(2)
plt.subplots(figsize=(25,5))
origin_total = new_data["origin"].value_counts()
sns.barplot(x=origin_total[:20].keys(), y=origin_total[:20].values)
style_count = new_data["main_style"].value_counts()
plt.subplots(figsize=(30,5))
sns.barplot(x=style_count[:20].keys(), y=style_count[:20].values)
split_count = new_data["split"].value_counts(ascending=False)
formed_count = new_data["formed"].value_counts(ascending=False)
split_count
data_ = {'split': split_count, 'formed': formed_count}
split_formed = pd.DataFrame(data=data_)
split_formed.tail()
fig = plt.figure(figsize=(30,10))
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])
ax1.plot(split_formed["formed"], color='teal')
ax2.plot(split_formed["split"], color='green')
ax2.set_title("Split Bands", fontsize=18)
# split_formed.dtypes
plt.subplots(figsize=(30,5))
sns.barplot(x="formed", y="fans", data=new_data)
new_data[(new_data.fans <= 2500) & (new_data.formed.isin(['1975', '1976', '1977'])) & (new_data.main_style == "Heavy")]
