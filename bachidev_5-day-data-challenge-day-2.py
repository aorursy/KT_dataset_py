#import pandas
import pandas as pd
pd.read_csv("../input/cereal.csv")
data = pd.read_csv("../input/cereal.csv")
data.describe()
import matplotlib.pyplot as plt
plt.title('Sodium Histogram')
plt.hist(data["sodium"])
