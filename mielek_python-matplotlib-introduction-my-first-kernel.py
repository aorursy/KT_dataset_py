import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

dunya = pd.read_csv("../input/dunya.csv")
x=["1.USA","2.CHINA","3.JAPAN","4.GERMANY","5.ENGLAND","6.FRANCE"]

y=[20410,14090,5170,4210,2940,2930]

z=[63039,9957,40649,51158,44162,44915]
plt.plot(x,y,"r")

plt.xlabel("COUNTRIES")

plt.ylabel("NATIONAL INCOME")

plt.title("NATIONAL INCOME BY COUNTRIES")

plt.show()
plt.subplot(1,2,1)

plt.plot(x,y,"r--")

plt.subplot(1,2,2)

plt.plot(x,z,"y*-")

plt.tight_layout()
f=plt.figure()

axes=f.add_axes([0.1,0.1,0.9,0.9])

axes.plot(x,y)

axes.set_xlabel("COUNTRIES")

axes.set_ylabel("NATIONAL INCOME")

axes.set_title("NATIONAL INCOME BY COUNTRIES")
f=plt.figure()

axes1=f.add_axes([0.1,0.1,0.9,0.9])

axes2=f.add_axes([0.5,0.5,0.4,0.4])

axes1.plot(x,y,"y")

axes2.plot(x,z,"r")

fig,axes=plt.subplots(nrows=1, ncols=2)



axes[0].plot(x,y,"b")

axes[0].set_title("NATIONAL INCOME")



axes[1].plot(x,z,"y")

axes[1].set_title("PER PERSON")



plt.tight_layout()
f=plt.figure()

axes=f.add_axes([0.1,0.1,0.9,0.9])

axes.plot(x,y,color="red",linewidth=1,linestyle="-",marker="o",markersize=20,markerfacecolor="yellow",markeredgewidth=4,markeredgecolor="blue",alpha=0.9)

axes.set_xlabel("COUNTRIES")

axes.set_ylabel("NATIONAL INCOME")

axes.set_title("NATIONAL INCOME BY COUNTRIES")
f=plt.figure()

axes=f.add_axes([0.1,0.1,0.9,0.9])

axes.plot(x,y,color="red",linewidth=1,linestyle="-")

axes.set_xlabel("COUNTRIES")

axes.set_ylabel("NATIONAL INCOME")

axes.set_title("NATIONAL INCOME BY COUNTRIES")



axes.set_xlim([2,6])

axes.set_ylim([2000,7500])
f=plt.figure(figsize=(10,6))

axes=f.add_axes([0,0,1,1])

axes.plot(x,y)

axes.set_xlabel("COUNTRIES")

axes.set_ylabel("NATIONAL INCOME")

axes.set_title("NATIONAL INCOME BY COUNTRIES")
fig,axes=plt.subplots(nrows=1, ncols=2,figsize=(13,5))



axes[0].plot(x,y,"b")

axes[0].set_title("NATIONAL INCOME")



axes[1].plot(x,z,"y")

axes[1].set_title("PER PERSON")



plt.tight_layout()
fig.savefig("picture.png")
fig=plt.figure(figsize=(10,5))



ax=fig.add_axes([0,0,1,1])



ax.plot(x,y,"r",label="national income")

ax.plot(x,z,"y",label="per person")



ax.legend(loc=5)
c=pd.read_csv("../input/dunya.csv")
plt.scatter(c["Country"],c["Per person"])
plt.hist(c["Per person"],bins=10)
plt.step(x,y)
labels = 'TURKEY', 'USA', 'CHINA', 'GERMANY'

sizes = [15, 35, 35, 15]

explode = (0.2, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.