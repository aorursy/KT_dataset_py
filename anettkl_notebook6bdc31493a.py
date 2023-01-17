# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/chicago_taxi_trips_2016_01.csv")
df
df.info()
af = df[df.trip_miles!= 0]

bf = df[df.trip_seconds!= 0]
print("Sõite kokku:", len(df))

print("Kõige pikem sõit kestis ligikaudu:", (df["trip_seconds"].max()//3600), "tundi" )

print("Kõige pikem sõit kestis ligikaudu:", (bf["trip_seconds"].min()//60), "minutit" )

print("Kõige lühema sõidu distants:", (af["trip_miles"].min()*1.6), "kilomeetrit")

print("Kõige pikema sõidu distants:", (af["trip_miles"].max()*1.6), "kilomeetrit")
df["taxi_id"].value_counts()
df.groupby("taxi_id").aggregate({"trip_total":["sum","mean"],

                                "tips":["sum","mean"]})
movie_yearly_count = df['payment_type'].value_counts().sort_index().plot(kind='bar', color='r', alpha=0.5, grid=False, rot=45)

movie_yearly_count.set_xlabel('Makseviis')

movie_yearly_count.set_ylabel('Maksete arv')

movie_yearly_count.set_title('Taksosõidu eest tasumine')
af.plot.scatter("fare", "trip_miles", alpha=0.2);