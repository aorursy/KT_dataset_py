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
import matplotlib.pyplot as p

import numpy as n

import math

import collections

import pandas as pd





def find_combinations(total):

	combinations_amounts = []

	

	for amount_100 in range(0, total+1, 100):

		for amount_50 in range(0, total+1, 50):

			for amount_25 in range(0, total+1, 25):

				for amount_10 in range(0, total+1, 10):

					for amount_5 in range(0, total+1, 5):

						total_so_far = amount_100 + amount_50 + amount_25 + amount_10 + amount_5

						if total_so_far <= total:

							combinations_amounts.append([amount_100, amount_50, amount_25, amount_10, amount_5, total - total_so_far])

	return combinations_amounts



totals = range(100, 600, 100)

lengths = [len(find_combinations(total)) for total in totals]

p.plot(totals, lengths)

p.show()