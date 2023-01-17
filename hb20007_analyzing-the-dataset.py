import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir('../input'))
data = pd.read_csv('../input/Transformed Data Set - Sheet1.csv');
data.sample(3)
data.info()
# sns.countplot('Gender', data=data, palette='Greens') # Just like 'Greens', there's also 'Reds', 'Purples' etc... Also, you can append to any of these _r which means reverse or _d which means dark.
data['Gender'].value_counts().plot.pie(explode=[0,0.1], shadow=True)
sns.countplot('Favorite Color', hue='Gender', data=data, palette='Greens')
sns.countplot(y='Favorite Music Genre', hue='Gender', data=data, palette='Greens');
sns.countplot('Favorite Beverage', hue='Gender', data=data, palette='Greens');
sns.countplot(y='Favorite Soft Drink', hue='Gender', data=data, palette='Greens');