import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd

import seaborn as sns



%matplotlib inline
# We read from the csv files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# and view how the first lines are composed

train.head()
# Get the total number of passengers we have in our dataset

n_passengers = train.shape[0]



# Now we discard some of the columns that would render unique when counting their values

train_to_print = train.drop(["PassengerId", "Name", "Age", "Ticket", "Cabin", "Fare"], axis=1)





for item in train_to_print:

    print(train_to_print[item].value_counts())

    print("{0}: Missing data from {1} passengers".format(item, n_passengers - train[item].count()))

    print("\n")

    
train.count()
train[["Survived", "Sex"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Sex")
children = train.loc[(train.Age > 0) & (train.Age <= 10)]

mean_child = children[["Survived", "Age"]].groupby(["Age"], as_index=False).mean()["Survived"].mean()

print("The average child between 0 and 10 had a {:.2f}% chance of survival".format(mean_child*100))



others = train.loc[(train.Age > 10)]

mean_others = others[["Survived", "Age"]].groupby(["Age"], as_index=False).mean()["Survived"].mean()

print("The mean chances of survival for the rest were: {:.2f}%".format(mean_others*100))
train[["Survived", "Pclass"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Pclass")
# Create the array with the x locations for the groups

x_locations = np.arange(1)  

# Set the bar's width

bar_width = 0.2      



# Set each pair of values we'll use

survived = [0, 1]

gender = ["female", "male"]

colors = ['c', 'b']

spacing = [x_locations, x_locations + 0.3] 



# Declare the figure of our plot and its size

fig, axes = plt.subplots(1, 2, figsize=(8,3.8))



# Create both plots 

for ax, surv in zip(axes, survived):

    for sex, sp, c in zip(gender, spacing, colors):

        gender_surv_df = train.loc[(train.Sex == sex) & (train.Survived == surv)]

        count = gender_surv_df.PassengerId.count()



        ax.bar(sp, count, bar_width, color=c)

        ax.set_title('Survived=' + str(surv))

        ax.set_xticks([0, 0.3])

        ax.set_xticklabels(('Female', 'Male'))



# First fill the array with the columns we want to drop

to_drop = ["PassengerId", "Fare", "Cabin"]



# and we drop them. The parameter axis=1 means that these values

# are contained in the columns.

new_train = train.drop(to_drop, axis=1)



new_train.head(3)