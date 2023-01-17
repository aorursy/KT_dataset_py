import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



path = '/kaggle/input/google-play-store-apps/googleplaystore.csv'



data = pd.read_csv(path)

data
arcades = data[(data["Category"] == "GAME") & (data["Genres"] == "Arcade")]

arcades = arcades.assign(Installs_Int = lambda el: el.Installs.apply(lambda el: int(el.replace(',', '').replace('+', ''))))



arcades = arcades.sort_values(by = ['Installs_Int'])



installs = arcades["Installs_Int"]

prices = arcades.Price.apply(lambda el: float(el.replace('$', '')))

size = arcades.Size.apply(lambda el: float(el.replace('Varies with device', '0').replace('M', '')))

rating = arcades.Rating



free_projects = len(arcades[arcades["Type"] == "Free"])
def find_max_notnan(array):

    sorted_array = sorted(array)[:: -1]

    for i in sorted_array:

        if (not np.isnan(i)):

            return i



max_val = 50



installs_graph = (np.array(installs) * max_val) / find_max_notnan(np.array(installs))

prices_graph = (np.array(prices) * max_val) / find_max_notnan(np.array(prices))

size_graph = (np.array(size) * max_val) / find_max_notnan(np.array(size))

rating_graph = (np.array(rating) * max_val) / find_max_notnan(np.array(rating))
figure = plt.figure()

ax = figure.add_subplot()



figure.set_facecolor('white')



ax.plot(np.arange(len(arcades)), installs_graph, color = "red", label = "Installs")

ax.plot(np.arange(len(arcades)), prices_graph, color = "blue", label = "Prices")

ax.plot(np.arange(len(arcades)), size_graph, color = "black", label = "Size")

ax.plot(np.arange(len(arcades)), rating_graph, color = "green", label = "Rating")



ax.set_facecolor('white')



figure.set_figwidth(30)

figure.set_figheight(7)



plt.legend()

plt.show()