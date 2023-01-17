# Let's import the libraries we'll need
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv("../input/StudentsPerformance.csv")
# Let's check the 5 first rows
dataset.head()
dataset.info()
# Let's describe the dataset
dataset.describe()
dataset.groupby("gender").describe()
# Let's group by race/ethnicity
dataset.groupby("race/ethnicity").describe()
dataset.groupby("parental level of education").describe()
dataset.groupby(["race/ethnicity", "parental level of education"]).describe()["math score"]
# I choose to grab the "math score" just to reduce the size width of the output
# Let's fix the figure size
plt.rcParams["figure.figsize"] = [10,6]
# Seaborn uses to send unimportant warnings, I'll hide them with the warnings module. 
# It is not recommanded though to do so
import warnings
warnings.filterwarnings("ignore")
sns.pairplot(dataset, hue = "gender", palette= 'viridis', plot_kws= {'alpha': 0.6})
# The plot_kws= {'alpha': 0.5} gives us the possibility provide additional arguments in pairplots
dataset.head(3)
sns.countplot("parental level of education", data=dataset)
sns.distplot(dataset["math score"], kde = False, bins = 50)
sns.countplot("test preparation course", data = dataset)
sns.boxplot(x = "race/ethnicity", y = "math score", data = dataset, 
            hue = "gender", palette = "viridis")
sns.boxplot("race/ethnicity", "reading score", data = dataset,
           hue = "gender", palette = 'viridis')
sns.boxplot(x = "test preparation course", y = "reading score", data = dataset)
# Let's see correlation between scores
dataset.corr()
sns.heatmap(dataset.corr(), cmap = 'viridis_r')