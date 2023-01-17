%matplotlib inline
import pandas as pd 
import matplotlib.pyplot as plt
#Above, just importing the libraries I'm sure I'll need.
agedemo=pd.read_csv("../input/age_gender_bkts.csv") #Let's import the data
usdemo=[agedemo[agedemo.country_destination.isin(["US"])]] #I'm specifically interested in the US data.
usdemo
#I'm not really sure how helpful this data is. I don't use it in any of my models currently.
#But we can explore it a bit more.
#Let's import the age data from the training set.
train=pd.read_csv("../input/train_users.csv")
trainages=train[train.age > 0]
trainages.age.value_counts()
#I'm extremely skeptical that anyone over 100 and under 16 is using airbnb.
#There must be some mistake with the data. Let's remove the outliers.
trainages=trainages[(trainages.age > 15) & (trainages.age < 100)]
trainages.age.value_counts()
#I'll be continuing this later. For now, has anyone found anything interesting in this data-set? Dicuss it down there.