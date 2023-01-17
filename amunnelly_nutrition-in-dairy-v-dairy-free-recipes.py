import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/epi_r.csv")

data.describe()
%matplotlib inline

nutrition = [['calories', 'fat'], ['sodium', 'protein']]

fig, ax = plt.subplots(2, 2)

for firstN in range(2):

    for secondN in range(2):

        temp = data[nutrition[firstN][secondN]]

        temp = temp.dropna()

        ax[firstN, secondN ].boxplot(temp)

        ax[firstN, secondN].set_title(nutrition[firstN][secondN].title())

plt.subplots_adjust(wspace = .66)
data[['calories', 'fat', 'sodium', 'protein']].describe()
nutrition = ['calories', 'fat', 'sodium', 'protein']

fig, ax = plt.subplots(1,4, sharey=True, figsize = (8, 5))

for i in range(4):

    temp = data[nutrition[i]]

    temp = temp.dropna()

    temp = temp[temp<=temp.quantile(0.75)*1.5]

    ax[i].boxplot(temp)

    ax[i].set_title(nutrition[i].title())
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize = (8, 5))

plt.suptitle('Nutrient v Nutrient', fontweight = 'bold', fontsize = 14)

nutrient_v_nutrient = [['fat', 'protein'],

                       ['fat', 'sodium'],

                       ['fat', 'calories'],

                       ['protein', 'sodium'],

                       ['protein', 'calories'],

                       ['calories', 'sodium']]

counter_x = 0

counter_y = 0



for pairing in nutrient_v_nutrient:

    temp = data[pairing]

    ax[counter_x, counter_y].scatter(temp[pairing[0]].values, temp[pairing[1]].values)

    ax[counter_x, counter_y].set_xlabel(pairing[0].title())

    ax[counter_x, counter_y].set_ylabel(pairing[1].title())

    current_title = " v ".join([pairing[0].title(), pairing[1].title()])

    ax[counter_x, counter_y].set_title(current_title)



    counter_y += 1

    if counter_y > 2:

        counter_y = 0

        counter_x += 1
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

counter_x = 0

counter_y = 0



fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize = (8, 5))

plt.suptitle('Nutrient v Nutrient (Data Scaled to between 0 and 1)', fontweight = 'bold', fontsize = 14)



for pairing in nutrient_v_nutrient:

    temp = data[pairing].dropna()

    x_scaled = scaler.fit_transform(temp[pairing[0]].values.reshape(-1, 1))

    y_scaled = scaler.fit_transform(temp[pairing[1]].values.reshape(-1, 1))

#     The latest version of MinMaxScaler() demains the data be reshaped



    ax[counter_x, counter_y].scatter(x_scaled,

                                     y_scaled,

                                     color = 'red',

                                     s = 75,

                                    alpha = 0.05)

    ax[counter_x, counter_y].set_xlabel(pairing[0].title())

    ax[counter_x, counter_y].set_ylabel(pairing[1].title())

    current_title = " v ".join([pairing[0].title(), pairing[1].title()])

    ax[counter_x, counter_y].set_title(current_title)



    counter_y += 1

    if counter_y > 2:

        counter_y = 0

        counter_x += 1
from collections import defaultdict

ingredients = data.columns

counter = defaultdict(int)

for i in ingredients:

    try: # to escape errors where the values aren't of type(int)

        temp = data[data[i] > 0].copy() # to avoid the a changed-values-of-original error

        counter[i] = len(temp)

    except:

        pass

    

counterCounter = list(counter.values())



counterCounter.sort(reverse=True)

for c in counterCounter[:30]:

    for key, value in counter.items():

        if value == c:

            print ("{:<19}{:,}".format(key, value))

fig, ax = plt.subplots(6, 2, sharex=True, sharey=True, figsize = (8, 24))

plt.suptitle('Dairy v Dairy-Free', fontsize = 25, fontweight='bold', y = 0.95)



counter_x = 0

counter_y = 0



for pairing in nutrient_v_nutrient:

    temp = data[data['dairy'] == 1].copy()

    temp = temp[pairing]

    temp = temp.dropna()

    x_scaled = scaler.fit_transform(temp[pairing[0]].values.reshape(-1, 1))

    y_scaled = scaler.fit_transform(temp[pairing[1]].values.reshape(-1, 1))



    ax[counter_x, counter_y].scatter(x_scaled,

                                     y_scaled,

                                     color = 'seagreen',

                                     s = 50,

                                    alpha = 0.25,

                                    label = 'dairy')

    ax[counter_x, counter_y].set_xlabel(pairing[0].title())

    ax[counter_x, counter_y].set_ylabel(pairing[1].title())

    current_title = " v ".join([pairing[0].title(), pairing[1].title()])

    ax[counter_x, counter_y].set_title(current_title)

    ax[counter_x, counter_y].legend()



    counter_x += 1

    

counter_y = 1

counter_x = 0







for pairing in nutrient_v_nutrient:

    temp = data[data['dairy free'] == 1].copy()

    temp = temp.dropna()

    temp = temp[pairing]

    x_scaled = scaler.fit_transform(temp[pairing[0]].values.reshape(-1, 1))

    y_scaled = scaler.fit_transform(temp[pairing[1]].values.reshape(-1, 1))



    ax[counter_x, counter_y].scatter(x_scaled,

                                     y_scaled,

                                     color = 'slateblue',

                                     s = 50,

                                    alpha = 0.25,

                                    label = 'dairy-free')

    ax[counter_x, counter_y].set_xlabel(pairing[0].title())

    ax[counter_x, counter_y].set_ylabel(pairing[1].title())

    current_title = " v ".join([pairing[0].title(), pairing[1].title()])

    ax[counter_x, counter_y].set_title(current_title)

    ax[counter_x, counter_y].legend()

    counter_x += 1

    