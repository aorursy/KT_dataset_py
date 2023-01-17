import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import csv

from subprocess import check_output

outcome_results = {'Adoption': [], 'Died': [], 'Euthanasia': [], 'Return_to_owner': [], 'Transfer': []}

# Removes the leading 'A' from each ID and converts to integer
def preprocess_ids(id_list):
    new_list = []

    for animal_id in id_list:
        new_list.append(int(animal_id[1:]))

    return new_list

animals = pd.read_csv("../input/train.csv")

animal_ids = animals['AnimalID'].tolist()
outcomes = animals['OutcomeType'].tolist()

animal_ids = preprocess_ids(animal_ids)

i = 0

for outcome in outcomes:
    outcome_results[outcome].append(animal_ids[i])
    i += 1

data = []
labels = []
for key in outcome_results:
    labels.append(key)
    data.append(outcome_results[key])


fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_ylabel('ID')
ax.set_xticklabels(labels)

plt.show()



potential_outcomes = ('Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer')
outcome_results = {}

for out in potential_outcomes:
    outcome_results[out] = [0, 0]
    
animal_names = animals['Name'].tolist()

i = 0
for name in animal_names:
    if not type(name) == float:
        outcome_results[outcomes[i]][0] += 1
    outcome_results[outcomes[i]][1] += 1
    i += 1

name_to_death_ratio = []

for key in outcome_results:
    labels.append(key)
    name_to_death_ratio.append(100 * outcome_results[key][0] / outcome_results[key][1])

fig, ax = plt.subplots()
ind = np.arange(5)
ax.bar(ind, name_to_death_ratio, 0.5, color='r')
ax.set_ylabel('Percentage with a name (%)')
ax.set_xticklabels(labels)
ax.set_xticks(ind + 0.3)

plt.show()