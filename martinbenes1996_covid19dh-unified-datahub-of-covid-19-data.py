!pip install covid19dh --upgrade > nul
from covid19dh import covid19

x,src = covid19(verbose = False)

x
src
import pandas as pd # data lib



import matplotlib # visualization lib

import matplotlib.pyplot as plt

matplotlib.rc('figure', figsize=(12, 5))
usa,usa_src = covid19("USA") # read US data
usa
# plot confirmed

plt.plot(usa["date"], usa["confirmed"], "r")

# format plot

plt.gcf().autofmt_xdate()

plt.suptitle("Confirmed cases of Covid-19 in USA", size=16)

plt.box(on=None)

plt.show()
# plots

plt.plot(usa["date"], usa["deaths"], "k")

plt.plot(usa["date"], usa["recovered"], "y")

# format plot

plt.gcf().autofmt_xdate()

plt.suptitle("Recovered and deaths of Covid-19 in USA", size=16)

plt.box(on=None)

plt.show()
usa2,usa2_src = covid19("USA", level = 2, verbose = False) # fetch US states data
usa2
fig = plt.figure()

ax = fig.add_subplot(111)

# plot each state separately

for state, data in usa2.groupby("administrative_area_level_2"):

    data = usa2[usa2["administrative_area_level_2"] == state]

    ax.plot(data["date"], data["confirmed"], label = state)

    ax.text(data["date"].iloc[-1], data["confirmed"].iloc[-1], state, fontsize=7)

# format plot

fig.suptitle("Confirmed cases of Covid-19 in US States", size = 16)

#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox = False, ncol = 8, prop={'size': 9})

plt.gcf().autofmt_xdate()

plt.box(on=None)

fig.show()
# get maximal record

usa2[ usa2["confirmed"] == max(usa2["confirmed"]) ]
x,src = covid19(["ES","italy"], start="2020-03-01", end="2020-04-30", verbose=False)
x
# plot each state separately

for country in x.groupby("administrative_area_level_1").groups:

    data = x[x["administrative_area_level_1"] == country]

    plt.plot(data["date"], data["confirmed"], label = country)

# format plot

plt.gcf().autofmt_xdate()

plt.suptitle("Confirmed cases of Covid-19 in Italy and Spain", size = 16)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox = False, ncol = 8, prop={'size': 9})

plt.box(on=None)

plt.show()
x, src = covid19("USA", end = "2020-04-22", vintage = True, verbose = False)
x, src = covid19("USA", vintage = True) # too early to get today's vintage

print(x, src)