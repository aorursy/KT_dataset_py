#Imports

import numpy as np

import pandas as pd

import statistics as sts

import scipy



#Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
# Reading Data from file

DataFrame = pd.read_csv("../input/irisdataset/iris.csv")
#describing data



DataFrame.describe()
slen = DataFrame["sepal length"]

swid = DataFrame["sepal width"]

plen = DataFrame["petal length"]

pwid = DataFrame["petal width"]

print("Mean:")

print("Sepal Length", sts.mean(slen))

print("Sepal Width", sts.mean(swid))

print("Petal Length", sts.mean(plen))

print("Petal Width", sts.mean(pwid))



print("")

print("Median:")

print("Sepal Length", sts.median(slen))

print("Sepal Width", sts.median(swid))

print("Petal Length", sts.median(plen))

print("Petal Width", sts.median(pwid))



print("")

print("Standard Deviation:")

print("Sepal Length", sts.stdev(slen))

print("Sepal Width", sts.stdev(swid))

print("Petal Length", sts.stdev(plen))

print("Petal Width", sts.stdev(pwid))



print("")

print("Range")

print("Sepal Length", max(slen) - min(slen) )

print("Sepal Width", max(swid) - min(swid) )

print("Petal Length", max(plen) - min(plen) )

print("Petal Width", max(pwid) - min(pwid) )



print("")

print("Percentiles 25%")

print("Sepal Length", np.percentile(slen, 25) )

print("Sepal Width", np.percentile(swid, 25) )

print("Petal Length", np.percentile(plen, 25) )

print("Petal Width", np.percentile(pwid, 25) )



print("")

print("Percentiles 50%")

print("Sepal Length", np.percentile(slen, 50) )

print("Sepal Width", np.percentile(swid, 50) )

print("Petal Length", np.percentile(plen, 50) )

print("Petal Width", np.percentile(pwid, 50) )



print("")

print("Percentiles 75%")

print("Sepal Length", np.percentile(slen, 75) )

print("Sepal Width", np.percentile(swid, 75) )

print("Petal Length", np.percentile(plen, 75) )

print("Petal Width", np.percentile(pwid, 75) )
print("Box Plot for Sepal Length")

plt.boxplot(slen, widths = 0.3)

plt.title("Sepal Length")

# plt.savefig('BB_SL.png')

plt.show()





print("Box Plot for Sepal Width")

plt.boxplot(swid, widths = 0.3)

plt.title("Sepal Width")

# plt.savefig('BB_SW.png')

plt.show()





print("Box Plot for Petal Length")

plt.boxplot(plen, widths = 0.3)

plt.title("Petal Length")

# plt.savefig('BB_PL.png')

plt.show()





print("Box Plot for Petal Width")

plt.boxplot(pwid, widths = 0.3)

plt.title("Petal Width")

# plt.savefig('BB_PW.png')

plt.show()

print("Sepal Width - Historgam")

plt.figure(figsize=(8, 8))

plt.hist(swid, bins = 8, color = "pink") 

plt.title("Sepal Width") 

plt.xlabel("Sepal Width") 

plt.ylabel("Count") 

# plt.savefig('Histogram_SW.png')

plt.show(block = False)







print("Petal Length - Historgam")

plt.figure(figsize=(8, 8))

plt.hist(plen, bins = 8, color = "lightblue") 

plt.title("Petal Length") 

plt.xlabel("Petal Length") 

plt.ylabel("Count") 

# plt.savefig('Histogram_PL.png')

plt.show(block = False)

plt.figure(figsize=(8, 8))

sns.pairplot(DataFrame, hue='class', diag_kind = 'kde')

# plt.savefig('Scatter_Matrix.png')

plt.show()

def scatter_plot(x_label, y_label, z_label, clas, c, m, label):

    x = DataFrame[DataFrame["class"] == clas] [x_label]

    y = DataFrame[DataFrame["class"] == clas] [y_label]

    z = DataFrame[DataFrame["class"] == clas] [z_label]

    

    ax.scatter(x,y,z, color = c, edgecolors='k',s=90, alpha = 0.9, marker=m, label = label)

    

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    ax.set_zlabel(z_label)

    return





ax = plt.figure(figsize=(15,10)).gca(projection='3d')

scatter_plot('sepal length','sepal width','petal width', 'Iris-setosa','r','o','Iris-setosa')

scatter_plot('sepal length','sepal width','petal width', 'Iris-versicolor','g','o','Iris-versicolor')

scatter_plot('sepal length','sepal width','petal width', 'Iris-virginica','b','o','Iris-virginica')



plt.legend()

# plt.savefig("3d scatter")

plt.show()

    
   

print("Quantile Quantile plot for Sepal Length and Normal Distribution")

fig = plt.figure()

ax = fig.add_subplot(111)

scipy.stats.probplot(slen, dist="norm", plot=ax);

ax.set_title("Sepal Length")

# plt.savefig("QQ_SL")

plt.show()





print("Quantile Quantile plot for Petal Length and Normal Distribution")

fig = plt.figure()

ax = fig.add_subplot(111)

scipy.stats.probplot(plen, dist="norm", plot=ax);

ax.set_title("Petal Length")

# plt.savefig("QQ_PL")

plt.show()