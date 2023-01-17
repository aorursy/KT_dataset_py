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
import plotly.graph_objects as go

from plotly.subplots import make_subplots
# settign data sources

basedir = '/kaggle/input/kaggle-survey-2019/'

fileName = 'multiple_choice_responses.csv'

path = os.path.join(basedir, fileName)

df = pd.read_csv(path)



# exploring the first few rows (6) of the data

df.head(5)
df.shape