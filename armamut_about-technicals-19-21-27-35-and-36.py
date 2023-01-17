# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# The usual stuff here.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 120)
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
df.head()
a = df[df['id'] == 30]

ac = a['y'].cumsum()

ad = a['technical_19'].cumsum() / 350.0

ae = a['technical_21'].cumsum() / 200.0

af = a['technical_27'].cumsum() / 450.0

ag = a['technical_35'].cumsum() / 500.0

ah = a['technical_36'].cumsum() / 500.0



ax = ac.plot()

ad.plot(ax=ax)

ae.plot(ax=ax)

af.plot(ax=ax)

ag.plot(ax=ax)

ah.plot(ax=ax)
a = df[df['id'] == 11]

ac = a['y'].cumsum()

ad = a['technical_19'].cumsum() / 350.0

ae = a['technical_21'].cumsum() / 200.0

af = a['technical_27'].cumsum() / 450.0

ag = a['technical_35'].cumsum() / 500.0

ah = a['technical_36'].cumsum() / 500.0



ax = ac.plot()

ad.plot(ax=ax)

ae.plot(ax=ax)

af.plot(ax=ax)

ag.plot(ax=ax)

ah.plot(ax=ax)
a = df[df['id'] == 501]

ac = a['y'].cumsum()

ad = a['technical_19'].cumsum() / 350.0

ae = a['technical_21'].cumsum() / 200.0

af = a['technical_27'].cumsum() / 450.0

ag = a['technical_35'].cumsum() / 500.0

ah = a['technical_36'].cumsum() / 500.0



ax = ac.plot()

ad.plot(ax=ax)

ae.plot(ax=ax)

af.plot(ax=ax)

ag.plot(ax=ax)

ah.plot(ax=ax)