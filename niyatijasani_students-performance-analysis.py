import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.features.rankd import Rank1D, Rank2D 
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
import numpy as np


sns.set(style="ticks", color_codes=True)
%matplotlib inline
datamat = pd.read_csv('../input/student-mat.csv', sep=';')
datapor = pd.read_csv('../input/student-mat.csv', sep=';')
datamat.head(3)
datapor.head(5)
datamat.head(5)
data = [datamat,datapor]
data=pd.concat(data)
data = data.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
data.info()
data.describe(include=['int64', 'float64'])
# Classyfing pandas categorical columns
data.describe(include=['O'])
#A Heat Map is a graphical representation of the correlation between all the numerical variables in a dataset. The input provided is a correlation matrix generated using pandas.
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),cmap="YlGnBu", annot = True,linecolor='white', cbar=True,linewidths=1)
from numpy.random import normal, uniform
def overlaid_histogram(G3, G3_name, G3_color, G2, G2_name, G2_color, G1,G1_name, G1_color, x_label, y_label, title):
    # Set the bounds for the bins so that the two distributions are
    # fairly compared
    max_nbins = 10
    data_range = [min(min(G3), min(G2)), max(max(G3), max(G2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins
    bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(G3, bins = bins, color = G3_color, alpha = 1, label = G3_name)
    ax.hist(G2, bins = bins, color = G2_color, alpha = 0.5, label = G2_name)
    ax.hist(G1, bins = bins, color = G1_color, alpha = 0.5, label = G1_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')

# Call the function to create plot
overlaid_histogram(G3 = data['G3']
                   , G3_name = 'Grade 3'
                   , G3_color = '#539caf'
                   , G2 = data['G2']
                   , G2_name = 'Grade 2'
                   , G2_color = '#7663b0'
                   , G1 = data['G1']
                   , G1_name = 'Grade 1'
                   , G1_color = '#D8BFD8'
                   , x_label = 'Grade 1, Grade 2, Grade 3'
                   , y_label = 'Score'
                   ,title = 'Distribution of Grades')
#Weekend Alcohol Consumption
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data['Walc'])
ax = ax.set(ylabel="Students", xlabel="Weekend Alcohol Consumption")
#Weekday Alcohol Consumption
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data['Dalc'])
ax = ax.set(ylabel="Students", xlabel="Daily Alcohol Consumption")
# Compute average alcohol consumption
data['AvgAlcohol'] = data[['Walc', 'Dalc']].mean(axis=1)

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data['AvgAlcohol'])
ax = ax.set(ylabel="Number of Students",
            xlabel="Average Alcohol Consumption among students")
#Workday Alcohol Consumption and Health
sns.set(style="whitegrid")

plot1 = sns.factorplot(x="Dalc", y="health", hue="sex", data=data)
plot1.set(ylabel="Health", xlabel="Workday Alcohol Consumption")

#Computing Realationship of Average alcohol consumption(Weekly+Daily) and Health of students
plot2 = sns.factorplot(x="AvgAlcohol", y="health", hue="sex", data=data)
plot2.set(ylabel="Health", xlabel="Average Alcohol Consumption")
#Grade1 performance in comparision to Average alcohol consumption
sns.boxplot(x="AvgAlcohol", y="G1", data=data)
#Grade2 performance in comparision to Average alcohol consumption
sns.boxplot(x="AvgAlcohol", y="G2", data=data)
#Average Alcohol cosumption and final grade performance amongst the students
sns.boxplot(x="AvgAlcohol", y="G3", data=data, medianprops={"zorder":3}, showcaps=False)
sns.swarmplot(x="AvgAlcohol", y="G3", data=data, alpha=0.7) 
data.drop(['AvgAlcohol'], axis=1)

data_mod = pd.get_dummies(data)
data_mod.head()
mod_df = data_mod
y = mod_df['G3'].values
df1 = mod_df.drop(['G3'],axis=1)
X = df1.values
mod_df.shape
new_corr = data.corr()
new_corr['G3'].sort_values(ascending=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=30)
print("X_train shape {} and y_train shape {}"\
      .format(X_train.shape, y_train.shape))
print("X_test shape {} and y_test shape {}"\
      .format(X_test.shape, y_test.shape))
lr = LinearRegression()
model = lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
tree= DecisionTreeRegressor()
tree= tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
ls = Lasso()
ls.fit(X_train, y_train)
print(ls.score(X_test, y_test))  
