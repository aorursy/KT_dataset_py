import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
playstore_filepath = '/kaggle/input/google-play-store-apps/googleplaystore.csv'
userreviews_filepath = '/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv'

playstore_data = pd.read_csv(playstore_filepath)
userreviews_data = pd.read_csv(userreviews_filepath)

playstore_data.head()
userreviews_data.head()
playstore_data.info()
playstore_data['Category'].value_counts()
playstore_data[playstore_data['Category']=='1.9']
playstore_data = playstore_data.set_index('Category')
playstore_data = playstore_data.drop('1.9')
playstore_data = playstore_data.reset_index()
playstore_data['Category'].value_counts()
plt.figure(figsize=(30,10))
sns.set(font_scale=2)
sns.countplot(x = "Category", data = playstore_data)
plt.xticks(
    rotation=45,
    horizontalalignment='right')
playstore_data['Installs'] = playstore_data['Installs'].map(lambda x: str(x)[:-1])
playstore_data['Installs']=[col.replace(",", "") for col in playstore_data['Installs']]
playstore_data['Installs'] = pd.to_numeric(playstore_data['Installs'], errors='coerce')
playstore_data['Installs']

#playstore_data.groupby['Category']['Installs'].sum()
data_installs = playstore_data.groupby('Category').agg({'Installs':'sum'})
data_installs.sort_values('Installs', ascending=False).head(10)
plt.figure(figsize=(30,10))
sns.set(font_scale=2)
sns.barplot(x = data_installs.index, y = data_installs['Installs'])
plt.xticks(
    rotation=45,
    horizontalalignment='right')
plt.figure(figsize=(30,10))
sns.set(font_scale=2)
sns.countplot(x = "Category", hue='Type', data = playstore_data)
plt.xticks(
    rotation=45,
    horizontalalignment='right')
playstore_data['Price'] = playstore_data['Price'].map(lambda x: str(x).replace("$", ""))
playstore_data['Price'] = pd.to_numeric(playstore_data['Price'], errors='coerce')
data_price = playstore_data.groupby('Category').agg({'Price':['min', 'mean', 'max']}) 
data_price.sort_values(('Price', 'mean'), ascending=False)