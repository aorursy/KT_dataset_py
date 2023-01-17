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
data = pd.read_csv('../input/cereal.csv')
data.head()
import matplotlib.pyplot as plt

import seaborn as sns
data.describe()
plt.hist(data['calories'])

plt.title('Calories in Cereal')

plt.xlabel('Calories per serving of Cereal')

plt.ylabel('Count')
plt.hist(data['calories'], bins=9, edgecolor="black")

plt.title('Calories in Cereal')

plt.xlabel('Calories per serving of Cereal')

plt.ylabel('Count')
data.hist(column="calories", figsize=(8, 8))