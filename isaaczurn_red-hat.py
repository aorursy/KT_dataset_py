import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
people = pd.read_csv("../input/people.csv")
#people.head()
act = pd.read_csv("../input/act_train.csv")
people["char_1"].value_counts()
act["outcome"].value_counts()
act["activity_category"].value_counts()
