import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/merc.csv")

np.std(data["price"])
np.random.seed(11)

sample_500 = data.sample(n = 500)["price"]

x_bar = np.mean(sample_500)

x_bar
np.random.seed(11)

sample_15 = data.sample(n = 15)["mileage"]

x_bar = np.mean(sample_15)

print("x_bar is:", str(x_bar))

print("s is:", np.std(sample_15))
from scipy import stats



no_of_samples = 15



print(stats.t.ppf(1-0.025, no_of_samples - 1))