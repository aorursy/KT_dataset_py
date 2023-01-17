import numpy
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams["figure.figsize"] = (18,9)
from pandas import read_csv
dataframe = read_csv("../input/cov19-scenario-of-bangladesh-for-first-102-days/COV19_Bangladesh.csv", usecols=[1], engine='python')
dataframe.columns
plt.plot(dataframe,label="Affected daly case")
plt.xlabel ('Days',fontsize=20)
plt.ylabel ('Affected Numbers',fontsize=20)
plt.legend(fontsize=20)
plt.show()