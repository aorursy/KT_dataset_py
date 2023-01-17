# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        import pandas as pd



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
my_data = "../input/coronavirusdataset/Case.csv"

data = pd.read_csv(my_data)

data.sort_values(by=['confirmed'], ascending=False,inplace=True)

data.head(20)

data=data.groupby(['city'])['confirmed','city'].mean()



data.reset_index(level=0, inplace=True)

plt.figure(figsize=(30,5))



sns.barplot(x=data['city'], y=data['confirmed'])