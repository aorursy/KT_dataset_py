import numpy as np
print(5 + np.pi)
cvalues = [0.5, 10.8, 21.9, 32.5, 42.7, 51.8, 61.3, 70.9, 86.9, 99.7]

print (cvalues)

# print(cvalues * 9 / 5 + 32)
# this would produce an error, because cvalues is still a list and a list can't be multiplied by an integer that easy
# when using a huge amount of data: better use numpy arrays

C = np.array(cvalues)

print(C * 9 / 5 + 32)
# example
import pandas as pd

#define Data and structure
example_columns = ["OEM", "Modell", "Price", "TFlops"]

example_data = [
    ["Apple", "Macbook Pro", 2200, 1],
    ["Apple", "Macbook", 1400, 0.8],
    ["Dell", "XPS 15", 1900, 1.1],
    ["Lenovo", "Yoga", 1100, 0.6],
    ["HP", "Elitebook", 1600, 0.9],
    ["HP", "X360", 1900, 1.1],
    ["Asus", "Zephyrus", 2900, 1.5],
    ["Microsoft", "Surface Pro", 1800, 0.9]
]

# build dataframe
dataframe = pd.DataFrame(columns = example_columns, data = example_data)
# the pd.DataFrame is saved by the name: dataframe

# display dataframe
dataframe.head(10)
# only show OEMs
dataframe.OEM
# count how often every OEM is included
dataframe.OEM.value_counts()
# All Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # file system access

# read out data of file
data = pd.read_csv("../input/radfahren_muenchen.csv")
air_bnb = pd.read_csv("../input/AirBnB_NY.csv")
wine_quality =pd.read_csv("../input/redwine_quality.csv")
data.head(3)
# first example
import matplotlib.pyplot as plt
import numpy as np

# first graph
plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(dataframe.Modell))
error = np.random.rand(len(dataframe.Modell))

ax.barh(y_pos, dataframe.TFlops)
ax.set_yticks(y_pos)
ax.set_yticklabels(dataframe.OEM + " " + dataframe.Modell)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Geschwindigkeit')
ax.set_title('Geschwindigkeit der Computer')

plt.show()


# second graph
plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(dataframe.Modell))
error = np.random.rand(len(dataframe.Modell))

ax.barh(y_pos, dataframe.Price)
ax.set_yticks(y_pos)
ax.set_yticklabels(dataframe.OEM + " " + dataframe.Modell)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Preis')
ax.set_title('Preis der Computer')

plt.show()
# second example
# import statements
import matplotlib.pyplot as plt # data visualization
from mpl_toolkits.mplot3d import Axes3D # 3D visualization

# define data to plot
y = dataframe.Price
x = dataframe.TFlops

# create canvas
fig, ax = plt.subplots()

# set some visualizations
ax.set_ylabel("Preis", fontsize=10)
ax.set_xlabel("Geschwindigkeit", fontsize=10)
ax.set_title('Rechengeschwindigkeit der Computer in Relation zum Preis', fontsize = 13)

# plot data
ax.scatter(x, y)

# plot figure to console
plt.show()
Erhardtdata = data[data["zaehlstelle"] == "Erhardt"]
Arnulfdata = data[data["zaehlstelle"] == "Arnulf"]
Olympiadata = data[data["zaehlstelle"] == "Olympia"]

small_data = Erhardtdata.append(Arnulfdata, ignore_index = True)
small_data = small_data.append(Olympiadata, ignore_index = True)

# create canvas
fig, ax = plt.subplots(figsize=(10, 10))

# plot data
ax.scatter(small_data.niederschlag, small_data.gesamt)

# set some visualizations
ax.set_xlabel("Niederschlag", fontsize=15)
ax.set_ylabel("Anzahl Radfahrer", fontsize=15)
ax.set_title('Radfahrer in München', fontsize=20)

# plot figure to console
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

# plot data
ax.scatter(small_data.niederschlag, small_data.bewoelkung, c = "lightgreen")

# set some visualizations
ax.set_xlabel("Niederschlag", fontsize=15)
ax.set_ylabel("Bewölkung", fontsize=15)
ax.set_title('Radfahrer in München', fontsize=20)

# plot figure to console
plt.show()
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="ticks")

fig, ax = plt.subplots(figsize=(12, 12))
ax = sns.scatterplot(x="max-temp", y="gesamt",hue = "zaehlstelle",size = "sonnenstunden", sizes = (10, 400), data=small_data)

ax.set_xlabel("Maximale Temperatur", fontsize=15)
ax.set_ylabel("Anzahl Radfahrer gesamt", fontsize=15)
ax.set_title("Radfahrer in München", fontsize=20)

plt.show()
# create 3D canvas
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


# plot data
ax.scatter(Arnulfdata.sonnenstunden, Arnulfdata.gesamt, Arnulfdata["max-temp"], c="black")
ax.scatter(Olympiadata.sonnenstunden, Olympiadata.gesamt, Olympiadata["max-temp"], c = "orange")
ax.scatter(Erhardtdata.sonnenstunden, Erhardtdata.gesamt, Erhardtdata["max-temp"], c = "purple")

# set some visualizations
ax.set_xlabel("Sonnenstunden", fontsize=15)
ax.set_ylabel("Anzahl", fontsize=15)
ax.set_zlabel("Temperatur", fontsize=15)
ax.set_title('3D Plot Radfahrer in München', fontsize=20)

# plot figure to console
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

# set some visualizations
ax.set_title("AirBnb Preise nach New Yorker Stadtteilen", fontsize=20)
ax.set_ylabel("Preis", fontsize=15)

# plot data
ax.scatter(air_bnb.neighbourhood_group, air_bnb.price, c = "darkred")
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="ticks")

fig = sns.jointplot(air_bnb.price, air_bnb.number_of_reviews , kind="hex", color="yellow")
fig.set_axis_labels('Preis', 'Anzahl Reviews', fontsize=15)

fig, ax = plt.subplots(figsize=(6, 6))

# plot data
ax.scatter(air_bnb.price, air_bnb.number_of_reviews , c = "darkred")

# set some visualizations
ax.set_xlabel("Preis", fontsize=15)
ax.set_ylabel("Anzahl Reviews", fontsize=15)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="ticks")

fig = sns.jointplot(wine_quality.alcohol, wine_quality.quality , kind="hex", color="darkred")
fig.set_axis_labels('Alkoholgehalt', 'Qualität', fontsize=15)


fig, ax = plt.subplots(figsize=(6, 6))

# plot data
ax.scatter(wine_quality.alcohol, wine_quality.quality, c = "darkred")

ax.set_xlabel("Alkoholgehalt", fontsize=15)
ax.set_ylabel("Qualität", fontsize=15)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="ticks")

fig = sns.jointplot(wine_quality.alcohol, wine_quality.density , kind="hex", color="darkred")
fig.set_axis_labels('Alkoholgehalt', 'Dichte', fontsize=15)
ucc = pd.read_csv("../input/used_cars_clean.csv")
ucc.head(5)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="ticks")

fig = sns.jointplot(x="Power", y = "Price" , kind="hex", color="darkred", data=ucc, xlim=(0,200), ylim=(0,40))
# An alternative way to define the x and y axis is shown.
# This way is useful if there are blanks in the header of the column (e.g. "Owner Type" or "Kilometers Driven")

fig.set_axis_labels('Leistung', 'Preis', fontsize=15)
# 0 zu ersetzen:
a1 = [1, 1]
a2 = [2, 1]
a3 = [3, 1]
a4 = [4, 1]
a5 = [5, 1]
a6 = [6, 1]
antworten = [a1,a2,a3,a4,a5,a6]
meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Answer"])
meine_antworten.to_csv("meine_loesung.csv", index = False)
