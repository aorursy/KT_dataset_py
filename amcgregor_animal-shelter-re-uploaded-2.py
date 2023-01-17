#invite everyone to the Kaggle partay

import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from collections import Counter
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head()
train.describe()
def missing_values(data): 
    number_of_missing_values = data.isnull().sum()
    percentage_missing = (data.isnull().sum())/data.isnull().count()
    missing_values_table = pd.concat([percentage_missing, number_of_missing_values], axis=1, keys=["Percentage missing", "Number of missing values"])
    return missing_values_table.sort_values("Percentage missing", ascending=False)
    
missing_values(train)
#Drop missing values for age and sex
train.drop(train[pd.isnull(train["SexuponOutcome"])].index, inplace=True)
train.drop(train[pd.isnull(train["AgeuponOutcome"])].index, inplace=True)
#Drop outcome subtype entirely
train.drop(["OutcomeSubtype"], axis="columns", inplace=True)
#Fill missing name values
train["Name"].fillna("nameless", inplace=True)
#Create a new column corresponding to whether the animal had a name or not
train["HadAName"] = "yes"
indexes_of_nameless = train.loc[train["Name"]=="nameless"].index
train.at[indexes_of_nameless.values, "HadAName"] = "no"
#Confirm that there are no missing values left
missing_values(train)
#Check everything looks fine and dandy
train.head()

#make a copy of this data to be used later if I need to start from almost-scratch again
train_raw = train.copy()
train.dtypes
baseline_model_features_names = ["AnimalType", "SexuponOutcome", "HadAName"]
baseline_model_X = train[baseline_model_features_names].copy()
baseline_model_X = pd.get_dummies(baseline_model_X, drop_first=True) #drop one column since it would be linearly dependent 
                                                        #and just makes unneeded extra columns to deal with
baseline_model_y = train["OutcomeType"].copy()
baseline_model_X.head() 
#Need to transform the categorical labels into numeric data
baseline_model_y.unique()
outcomes_dict = {"Return_to_owner" : 0,
                "Euthanasia" : 1,
                "Adoption" : 2,
                "Transfer" : 3,
                "Died" : 4}
baseline_model_y = baseline_model_y.map(outcomes_dict)
baseline_model_y.head(5)
#Create a training and validation set
X_train, X_test, y_train, y_test = train_test_split(baseline_model_X, baseline_model_y, random_state=36)
#I will choose to use a random forest model since they are ususally decent at predicting different categories. This kernel will be more about data
#exploration and feature selection and engineering as opposed to finding a specific model/parameters of the model. This is because the way the data is
#presented can often impact the accuracy much more than the model selection, so I want to refine this skill. 

#The metrics used will be log loss (since this is what Kaggle judges this competition in) and accuracy (since this is easier to interpret). 

def RF_scores(X, y):
    RF_baseline = RandomForestClassifier(random_state=36, n_estimators=100)
    y_predictions_proba = cross_val_predict(RF_baseline, X, y, method="predict_proba", cv=5)
    log_loss_score = log_loss(y, y_predictions_proba)
    accuracy = cross_val_score(RF_baseline, X, y, cv=5)
    return "The log loss is {0} and the accuracy is {1}".format(log_loss_score, np.mean(accuracy))
RF_scores(baseline_model_X, baseline_model_y)
#Refreshing myself
train = train_raw.copy()
train.head()
#Drop animal ID as I don't think it will have any relevance and it's a mess to sort out. Will bear this in mind as a possibility to come back to later
#I will also drop the 'name' columns as for now I will make do with having a name or not since there are so many names to sort through otherwise
train.drop("AnimalID", axis="columns", inplace=True)
train.drop("Name", axis="columns", inplace=True)
train.head()
Counter(train["SexuponOutcome"])
train.groupby("AnimalType")["Breed"].nunique()
dog_indexes = train.loc[train["AnimalType"] == "Dog"].index
cats_data = train.copy()
cats_data = cats_data.groupby("AnimalType").get_group(("Cat"))
cats_data.head()
#Calculate percentage of cats which fall into the most popular breeds 
def percentage_in_most_popular(data, number_of_breeds_included):
    breeds_counted = Counter(data["Breed"])
    list_of_n_most_common_breeds = breeds_counted.most_common()[:number_of_breeds_included:1]
    number_of_animals_in_breeds = sum(x[1] for x in list_of_n_most_common_breeds)
    total_no_breeds = cats_data["Breed"].count()
    percentage_covered = number_of_animals_in_breeds/total_no_breeds
    return percentage_covered

percentage_covered_by_number_of_breeds_array = []
for breeds in np.arange(train.groupby("AnimalType")["Breed"].nunique()["Cat"]+1):    
    percentage_covered_by_number_of_breeds_array.append(percentage_in_most_popular(cats_data, breeds))
plt.plot(np.arange(train.groupby("AnimalType")["Breed"].nunique()["Cat"]+1), percentage_covered_by_number_of_breeds_array)
plt.title("Line plot showing percentage of cats covered by x amount of breeds")
plt.xlabel("Number of breeds")
plt.ylabel("Percentage of animals covered")
plt.show()
plt.plot(np.arange(train.groupby("AnimalType")["Breed"].nunique()["Cat"]+1), percentage_covered_by_number_of_breeds_array)
plt.title("Line plot showing percentage of cats covered by x amount of breeds - Zoomed in")
plt.xlabel("Number of breeds")
plt.ylabel("Percentage of animals covered")
plt.xlim(0,8)
plt.show()
percentage_covered_by_number_of_breeds_array[4]
indexes_of_breed = [3,3,4,5]
for a in np.arange(1):

    indexes_of_next_breed = np.array(cats_data.loc[cats_data["Breed"] == [x[0] for x in Counter(cats_data["Breed"]).most_common()[a:a+1:1]][0]].index)
   # indexes_of_breed + indexes_of_next_breed
    #print(indexes_of_next_breed)
    
#new = indexes_of_breed + indexes_of_next_breed
indexes_of_next_breed
#if breed name is not in top 4 common
#find all its indexes 
#change breed value to 'other'

cat_breeds = train.groupby("AnimalType")["Breed"].unique()["Cat"]
number_of_cat_breeds = train.groupby("AnimalType")["Breed"].nunique()["Cat"]

for breed in cat_breeds:
    if breed in [x[0] for x in Counter(cats_data["Breed"]).most_common()[:-(number_of_cat_breeds-4):-1]]:
        breed_indexes = cats_data.loc[cats_data["Breed"] == breed].index
        cats_data.at[breed_indexes.values, "Breed"] = "other"
cats_data.head(10)
outcomes_dict = {"Return_to_owner" : 0,
                "Euthanasia" : 1,
                "Adoption" : 2,
                "Transfer" : 3,
                "Died" : 4}
y = cats_data["OutcomeType"].map(outcomes_dict)

news = cats_data.groupby(["Breed", "OutcomeType"]).size()
cats_outcomes_breeds = pd.DataFrame(news.reset_index())

cats_outcomes_breeds
cats_outcomes_breeds.groupby("Breed").get_group("other")[0]
#Plot the proportion of each breed that had each outcome

fig, ax = plt.subplots(2, 2, figsize=(10,10))

domestic_longhair = cats_outcomes_breeds.groupby("Breed").get_group("Domestic Longhair Mix")[0]
domestic_mediumhair = cats_outcomes_breeds.groupby("Breed").get_group("Domestic Medium Hair Mix")[0]
domestic_shorthair = cats_outcomes_breeds.groupby("Breed").get_group("Domestic Shorthair Mix")[0]
other = cats_outcomes_breeds.groupby("Breed").get_group("other")[0]

wedges, texts, autotexts = ax[0,0].pie(domestic_longhair, autopct="%.0f%%")
ax[0,0].set_title("Domestic Longhair")
wedges, texts, autotexts = ax[0,1].pie(domestic_mediumhair, autopct="%.0f%%")
ax[0,1].set_title("Domestic Mediumhair")
wedges, texts, autotexts = ax[1,0].pie(domestic_shorthair, autopct="%.0f%%")
ax[1,0].set_title("Domestic Shorthair")
wedges, texts, autotexts = ax[1,1].pie(other, autopct="%.0f%%")
ax[1,1].set_title("Other")

plt.suptitle("Pi charts showing the percentage of each outcome of the top 4 breeds of cat")
labels = ["Adoption", "Died", "Euthanasia", "Return to owner", "Transfer"]
plt.legend(wedges, labels, loc="upper right", bbox_to_anchor=(1.1, 0.3, 0.5, 1), fontsize="large")
plt.show()
cats_data.head()
