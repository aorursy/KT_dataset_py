# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
digimon_list = pd.read_csv("/kaggle/input/digidb/DigiDB_digimonlist.csv")

digimon_list.head()
digimon_list.info()
# Stage

digimon_stage = digimon_list['Stage'].value_counts().reset_index()

digimon_stage.columns = ['Stage', 'value']

px.bar(digimon_stage, x='Stage', y='value')
# Type

digimon_type = digimon_list['Type'].value_counts().reset_index()

digimon_type.columns = ['Stage', 'value']

px.bar(digimon_type, x='Stage', y='value')
# Attribute

digimon_attribute = digimon_list['Attribute'].value_counts().reset_index()

digimon_attribute.columns = ['Attribute', 'value']

px.bar(digimon_attribute, x='Attribute', y='value')