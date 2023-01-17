import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Data= pd.read_csv("/kaggle/input/2020-indonesian-university-ranking/2020 Indonesian University Ranking.csv")
Data.head()
#The Best 10 Indonesian University by Rank

top_University = Data.sort_values(by='Rank', ascending=True)[:10]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_University.University, x=top_University.Rank)

plt.xticks()

plt.xlabel('Rank')

plt.ylabel('University')

plt.title('The Best 10 Indonesian University by Rank')

plt.show()
#The Best 25 Indonesian University by Rank

top_University = Data.sort_values(by='Rank', ascending=True)[:25]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_University.University, x=top_University.Rank)

plt.xticks()

plt.xlabel('Rank')

plt.ylabel('University')

plt.title('The Best 25 Indonesian University by Rank')

plt.show()
#The Best 50 Indonesian University by Rank

top_University = Data.sort_values(by='Rank', ascending=True)[:50]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_University.University, x=top_University.Rank)

plt.xticks()

plt.xlabel('Rank')

plt.ylabel('University')

plt.title('The Best 50 Indonesian University by Rank')

plt.show()