import pandas as pd
data = pd.read_csv("../input/cereal.csv")
print(data)
print(data.columns)
# data.describe()

print(data.describe(include="all"))
import matplotlib.pyplot as plt
x = data['calories']

print(x)
plt.hist(x)

plt.title("Histogram of " + x.name)

plt.xlabel(x.name)

plt.ylabel("count")

plt.grid()

plt.show()
from scipy.stats import ttest_ind

from scipy.stats import probplot

import pylab
probplot(data['sodium'], dist='norm', plot=pylab)
# calory of hot cereal

sodiumOfHotCereal = data['sodium'][data['type']=='H']

print(sodiumOfHotCereal.describe())

print("==========================="+"\n")



# calory of cold cereal

sodiumOfColdCereal = data['sodium'][data['type']=='C']

print(sodiumOfColdCereal.describe())

# print('Calories of Cold:')

# print(caloriesOfCold)
testResult=ttest_ind(sodiumOfHotCereal, sodiumOfColdCereal, equal_var=False)

print("the p-value of t-test is {0}".format(testResult.pvalue))
fig = plt.figure()

fig.add_axes()

ax1 = fig.add_subplot(1,2,1)

ax1.hist(sodiumOfColdCereal)

ax1.set_title("sodium of cold cereal")

ax1.set_xlabel("sodium of cold cereal")

ax1.set_ylabel("count")

ax1.grid()

# print(ax1.get_ylim())

# print(ax1.get_xlim())



ax2 = fig.add_subplot(1,2,2)

ax2.hist(sodiumOfHotCereal, range=(ax1.get_xlim()))

ax2.set_xticks(ax1.get_xticks())

ax2.set_ylim(ax1.get_ylim())

ax2.grid()

ax2.set_title("sodium of hot cereal")

ax2.set_xlabel("sodium of hot cereal")

# ax2.set_ylabel("count")

plt.show(block=False)


plt.hist(sodiumOfColdCereal, alpha=0.5, label='cold')

plt.hist(sodiumOfHotCereal,label='hot')

plt.legend(loc='upper right')

plt.title("sodium content of cereals")

plt.show()
import pandas as pd

import matplotlib.pyplot as plt
data.head()





# x = data['type']

# print(x)
x.head(10)