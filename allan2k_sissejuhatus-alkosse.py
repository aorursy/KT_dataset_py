import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/student-mat.csv")



import matplotlib.pyplot as plt



%matplotlib inline

pd.set_option('display.max_rows', 20)



import matplotlib.ticker as plticker



loc = plticker.MultipleLocator(base=1.0)
m = df.health[df["sex"] == "M"]

n = df.health[df["sex"] == "F"]



s = m.plot.hist(bins=5, grid=False, rwidth=0.95, color='black'); 

v = n.plot.hist(bins=5, grid=False, rwidth=0.7, alpha=0.8, color='red'); 

v.set_xlabel("1 - väga halb, 5 - väga hea")

v.set_ylabel("Õpilaste arv")

v.xaxis.set_major_locator(loc)

plt.title('ÕPILASTE TERVIS', color='black')

v.legend(["Mehed", "Naised"]);
m = df.Dalc[df["sex"] == "M"]

n = df.Dalc[df["sex"] == "F"]



s = m.plot.hist(bins=5, grid=False, rwidth=0.95, color='black'); 

v = n.plot.hist(bins=5, grid=False, rwidth=0.7, alpha=0.8, color='red'); 

v.set_xlabel("1 - väga vähe, 5 - väga palju")

v.set_ylabel("Õpilaste arv")

v.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE NÄDALA SEES', color='black')

v.legend(["Mehed", "Naised"]);
m = df.Walc[df["sex"] == "M"]

n = df.Walc[df["sex"] == "F"]



s = m.plot.hist(bins=5, grid=False, rwidth=0.95, color='black'); 

v = n.plot.hist(bins=5, grid=False, rwidth=0.7, alpha=0.8, color='red'); 

v.set_xlabel("1 - väga vähe, 5 - väga palju")

v.set_ylabel("Õpilaste arv")

v.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE NÄDALAVAHETUSEL', color='black')

v.legend(["Mehed", "Naised"]);


l = df.studytime[df["Walc"] == 1]

m = df.studytime[df["Walc"] == 2]

n = df.studytime[df["Walc"] == 3]

o = df.studytime[df["Walc"] == 4]

p = df.studytime[df["Walc"] == 5]



q = l.plot.hist(bins=4, grid=False, rwidth=0.95, color='black');

r = m.plot.hist(bins=4, grid=False, rwidth=0.8, alpha=0.8, color='red');

s = n.plot.hist(bins=4, grid=False, rwidth=0.65, alpha=0.8, color='yellow');

t = o.plot.hist(bins=4, grid=False, rwidth=0.5, alpha=0.8, color='brown');

u = p.plot.hist(bins=4, grid=False, rwidth=0.35, alpha=0.8, color='green');

q.set_xlabel("1 = <2 tundi, 2 = 2 kuni 5 tundi, 3 = 5 kuni 10 hours,  4 = >10 tundi")

q.set_ylabel("Õpilaste arv")

s.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE VS ÕPPIMISELE KULUTATUD AEG NÄDALAS', color='black')

q.legend(["Väga vähe", "Natsa", "Keskmiselt", "Sai natsa palju", "Lukku"]);
l = df.health[df["Walc"] == 1]

m = df.health[df["Walc"] == 2]

n = df.health[df["Walc"] == 3]

o = df.health[df["Walc"] == 4]

p = df.health[df["Walc"] == 5]



q = l.plot.hist(bins=5, grid=False, rwidth=0.95, color='black');

r = m.plot.hist(bins=5, grid=False, rwidth=0.8, alpha=0.8, color='red');

s = n.plot.hist(bins=5, grid=False, rwidth=0.65, alpha=0.8, color='yellow');

t = o.plot.hist(bins=5, grid=False, rwidth=0.5, alpha=0.8, color='brown');

u = p.plot.hist(bins=5, grid=False, rwidth=0.35, alpha=0.8, color='green');

q.set_xlabel("1 - väga halb, 5 - väga hea")

q.set_ylabel("Õpilaste arv")

s.xaxis.set_major_locator(loc)

plt.title('ALKOHOLI TARBIMINE VS TERVIS', color='black')

q.legend(["Väga vähe", "Natsa", "Keskmiselt", "Sai natsa palju", "Lukku"]);