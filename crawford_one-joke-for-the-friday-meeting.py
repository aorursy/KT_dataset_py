import csv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

file = '../input/jester_items.tsv'

df = pd.read_csv(file, sep=":\t", header=None, engine='python').rename(columns={0: "joke_id", 1: "joke"})

df.head()



# Random joke for the friday meeting:

d2 = df.sample()



for joke in d2.joke:

    print(joke)