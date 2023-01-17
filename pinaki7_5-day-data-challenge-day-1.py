# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import pandas as pd

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



dataset=pd.read_csv("../input/scrubbed.csv")

dataset.describe()