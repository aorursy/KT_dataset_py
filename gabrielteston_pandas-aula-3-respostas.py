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
pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv')
fires_in_brazil = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin-1')

fires_in_brazil.head()
fires_in_brazil.describe()
int(fires_in_brazil[(fires_in_brazil["state"] == "Alagoas") & (fires_in_brazil["month"] == "Janeiro")]["number"].mean())
fires_in_brazil["state"].unique()
fires_brazil_by_state = fires_in_brazil.groupby(by = ['state']).sum().reset_index()

fires_brazil_by_state
fire_states = pd.Series(fires_in_brazil["state"].unique()).map(str.upper)

fire_states
fires_brazil_by_state["number"].map(lambda column: column/fires_brazil_by_state["number"].sum())
fires_brazil_by_state.apply(lambda column: column/column.sum() if column.name == "number" else column, axis="rows")
brazil = pd.read_csv('/kaggle/input/brazil/brazil.csv', encoding='utf-8')

brazil
states = brazil["Federative unit"].map(str.upper)

states
fire_states[fire_states.isin(states)==False]
def normilize_states(state):

    import re

    d = {r"[ÁÃ]":"A",

    r"É":"E",

    r"Í":"I",

    r"[ÓÔ]":"O",

    r"Ú":"U",

    r"^RIO$": "RIO DE JANEIRO",

    r"^PIAU$": "PIAUI"}

    for k,v in d.items():

        state = re.sub(k,v,state)

    return state
normalized_states = states.map(normilize_states)

normalized_fire_states = fire_states.map(normilize_states)
normalized_states[normalized_states.isin(normalized_fire_states)==False]