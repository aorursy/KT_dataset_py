# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_json('../input/us-presidents.json')

data



# Any results you write to the current directory are saved as output.
party_colors = {'republican': 'r', 'democratic': 'b'}



data['color'] = data['party'].apply(lambda x: party_colors[x])
data.plot.scatter(x='year', y='cpv', c=data['color'])