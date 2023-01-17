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
# Supervised Learning: Avocado Prices

df_avocado = pd.read_csv("../input/avocado-prices/avocado.csv")

df_avocado.head()
# Unsurpevised Learning: The Cure discography

df_cure = pd.read_csv("../input/the-cure-discography/thecure_discography.csv")

df_cure.head()
# Reinforcement Learning: 3 million Sudoku puzzles with ratings

df_sudoku = pd.read_csv("../input/3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv")

df_sudoku.head()