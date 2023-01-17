import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
filepath = "../input/real-estate-dataset/data.csv"

my_data = pd.read_csv(filepath)

my_data.head()
my_data.shape
# LINE CHART SHOWING AGE DISTRIBUTION 



plt.figure(figsize = (14,7))

sns.lineplot(data=my_data['AGE'])

plt.title("DITRIBUTION OF AGE")

plt.ylabel('Age')