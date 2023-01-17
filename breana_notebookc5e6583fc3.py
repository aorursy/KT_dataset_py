import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



train_animals=pd.read_csv("../input/train.csv")

sample=pd.read_csv("../input/sample_submission.csv")

test_animals=pd.read_csv("../input/test.csv")

print(train_animals.head())
type_count=train_animals["AnimalType"].value_counts()

breed_count=train_animals["Breed"].value_counts()

color_count=train_animals["Color"].value_counts()

type_count.plot(kind="bar")

print(color_count)


pure_mutt=[]

breed=np.array(train_animals["Breed"])

for i in breed:

    if ("Mix" in i) | ('/' in i):

        pure_mutt.append("Mutt")

    else:

        pure_mutt.append("Pure")



train_animals["Pure/Mutt"]=pure_mutt        

print(train_animals[["Pure/Mutt", "Breed"]].head(200))