import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

# to avoid typing plt.show() after every plot:

%matplotlib inline
auto = pd.read_csv('../input/automobile/Automobile.csv')
auto.head()
sns.distplot(auto["normalized_losses"]);
sns.distplot(auto['city_mpg'], kde=False, rug=True);
sns.jointplot(auto['engine_size'], auto['horsepower']);
sns.jointplot(auto['engine_size'], auto['horsepower'], kind='hex');
sns.jointplot(auto['engine_size'], auto['horsepower'], kind='kde');
sns.jointplot(auto['engine_size'], auto['horsepower'], kind='reg');
sns.jointplot(auto['engine_size'], auto['horsepower'], kind='resid');
sns.pairplot(auto[['normalized_losses', 'engine_size', 'horsepower']]);
sns.stripplot(auto['fuel_type'], auto['horsepower']);
sns.stripplot(auto['fuel_type'], auto['horsepower'], jitter=True);
sns.swarmplot(auto['fuel_type'], auto['horsepower']);
sns.boxplot(auto['number_of_doors'], auto['horsepower']);
sns.boxplot(auto['number_of_doors'], auto['horsepower'], hue=auto['fuel_type']);
sns.barplot(auto['body_style'], auto['horsepower'], hue=auto['engine_location'],);

plt.legend(loc= 'upper center');
sns.countplot(auto['body_style']);
sns.pointplot(auto['fuel_system'], auto['horsepower'], hue=auto['number_of_doors']);
sns.catplot(x="fuel_type", 

               y="horsepower", 

               hue="number_of_doors", 

               col="engine_location", 

               data=auto);
sns.catplot(x="fuel_type", 

               y="horsepower", 

               hue="number_of_doors", 

               col="engine_location", 

               data=auto, 

               kind='swarm');
sns.catplot(x="fuel_type", 

            #y="horsepower", for count plot, y needs to be removed

            hue="number_of_doors", 

            col="engine_location", 

            data=auto, 

            kind='count');
sns.catplot(x="fuel_type", 

               y="horsepower", 

               hue="number_of_doors", 

               col="engine_location", 

               data=auto, 

               kind='box');
sns.catplot(x="fuel_type", 

               y="horsepower", 

               hue="number_of_doors", 

               col="engine_location", 

               data=auto, 

               kind='violin');
sns.lmplot(x="horsepower", y="peak_rpm", data=auto);
sns.lmplot(x="horsepower", y="peak_rpm",hue="fuel_type", data=auto);