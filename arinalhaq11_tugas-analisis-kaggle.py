# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
kepopuleran= df.sort_values('popularity', ascending=False)

plt.figure(figsize=(14,7))



plt.barh(kepopuleran['title'].head(10),kepopuleran['popularity'].head(10), align='center',

        color='darkgrey')

plt.gca().invert_yaxis()

plt.xlabel("Kepopuleran")

plt.title("tingkat kepopuleran movie")
import matplotlib.ticker as ticker

penghasilan= df.sort_values('revenue', ascending=False)

formatter = ticker.FormatStrFormatter('$%1.0f')

ay, ax = plt.subplots()

#ax.figure(figsize=(14,7))

ay.set_figheight(7)

ay.set_figwidth(14)

ax.xaxis.set_major_formatter(formatter)

plt.barh(penghasilan['title'].head(10),penghasilan['revenue'].head(10), align='center',

        color='darkgrey')

ax.invert_yaxis()

ax.set(xlabel="penghasilan dalam dolar")

ax.set_title('Penghasilan Tertinggi')
import matplotlib.ticker as ticker

budget= df.sort_values('budget', ascending=False)

formatter = ticker.FormatStrFormatter('$%1.0f')

ay, ax = plt.subplots()

#ax.figure(figsize=(14,7))

ay.set_figheight(7)

ay.set_figwidth(14)

ax.xaxis.set_major_formatter(formatter)

plt.barh(budget['title'].head(10),budget['budget'].head(10), align='center',

        color='darkgrey')

ax.invert_yaxis()

ax.set(xlabel="budget dalam dolar")

ax.set_title('budget Tertinggi')
