#import pandas
import pandas as pd
data = pd.read_csv("../input/cereal.csv")
data.head()
data.describe()
from scipy.stats import probplot
import pylab
probplot(data["sodium"], dist="norm", plot=pylab)
from scipy import stats
sodium_hot = data["sodium"][data["type"] == "H"]
sodium_cold = data["sodium"][data["type"] == "C"]
stats.ttest_ind(sodium_hot, sugar_cold, equal_var = False)
import matplotlib.pyplot as plt
# plot the cold cereals
plt.hist(sodium_cold, alpha=0.5, label='cold')
# and the hot cereals
plt.hist(sodium_hot, label='hot')
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title("Sodium(mg) content of cereals by type")
