import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
my_filepath = "../input/arctic-sea-ice-19792015/SeaIce.csv"

my_data = pd.read_csv(my_filepath)
my_data.head()
sns.scatterplot(x=my_data['Year'], y=my_data['Extent'], hue=my_data['Area'])