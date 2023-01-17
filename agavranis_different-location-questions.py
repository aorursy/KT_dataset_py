import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/quora-question-pairs"]).decode("utf8"))

print(check_output(["ls", "../input/movehub-city-rankings"]).decode("utf8"))
dataset = "test" # Obviously you want to run this on the test set as well



df = pd.read_csv("../input/quora-question-pairs/{}.csv".format(dataset))

locations = pd.read_csv("../input/movehub-city-rankings/cities.csv")

locations.to_csv('cities.csv',index=False)