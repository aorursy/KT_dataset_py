import pandas as pd

import seaborn as sns   # Why sns?  It's a reference to The West Wing

import matplotlib.pyplot as plt  # seaborn is based on matplotlib

sns.set(color_codes=True) # adds a nice background to the graphs

%matplotlib inline 

# tells python to actually display the graphs
auto = pd.read_csv('../input/automobiles/Automobile.csv')
auto.head()
sns.distplot(auto['highway_mpg']);
# we can turn the kde off and put a tic mark along the x-axis for every data point with rug

sns.distplot(auto['city_mpg'], kde=False, rug=True);
sns.jointplot(auto['engine_size'], auto['horsepower']);
sns.jointplot(auto['engine_size'], auto['horsepower'], kind="hex");
sns.jointplot(auto['engine_size'], auto['horsepower'], kind="kde");
sns.pairplot(auto[['normalized_losses', 'engine_size', 'horsepower']]);
sns.stripplot(auto['fuel_type'], auto['horsepower'], jitter=True);
sns.swarmplot(auto['fuel_type'], auto['horsepower']);
sns.boxplot(auto['number_of_doors'], auto['horsepower']);
sns.boxplot(auto['number_of_doors'], auto['horsepower'], hue=auto['fuel_type']);
sns.barplot(auto['body_style'], auto['horsepower'], hue=auto['fuel_type']);
sns.countplot(auto['body_style'],hue=auto['fuel_type']);
sns.pointplot(auto['body_style'], auto['horsepower'], hue=auto['number_of_doors']);
sns.catplot(x="fuel_type",

               y = "horsepower",

               hue="number_of_doors", 

               col="drive_wheels", 

               data=auto, 

               kind="box");

sns.lmplot(y="horsepower", x="engine_size", data=auto);
sns.lmplot(y="horsepower", x="engine_size",hue="fuel_type", data=auto);