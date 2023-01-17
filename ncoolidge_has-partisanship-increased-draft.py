# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #MPL for plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/states_party_strength_cleaned.csv')
congressDel = df["congress house"].str.lstrip().str.split(",", expand=True)

lowerHouse = df["legislature house"].str.lstrip().str.split(",", expand=True)

upperHouse = df["legislature sen"].str.lstrip().str.split(",", expand=True)

senateDel = df["congress sen"].str.lstrip().str.split(",", expand=True)
congressDel.iloc[0]
# IMPORTANT: From this point on, this notebook is very much in DRAFT form.



def partyNumbs(df):

    legis = {}

    for item in df:

        if item != None:

            leg = {}

            leg['members'] = int(item[:-1])

            leg['party'] = item[-1:]

            legis[item.index] = leg

        else:

            pass

    return legis

cong_dict = congressDel.apply(partyNumbs, axis=1)
cong_dict