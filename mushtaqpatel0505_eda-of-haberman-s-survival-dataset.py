import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Load haberman.csv into a pandas dataFrame.
Haberman = pd.read_csv("../input/haberman.csv",header=None, names=['age', 'year', 'axillary', 'survived'])
# (Q) how many data-points and featrues are there?
print (Haberman.shape)
Haberman.head()     # print first 5 rows  
Haberman.tail()   # print last 5 rows
#(Q) What are the column names in our dataset?
print (Haberman.columns)
#(Q) How many data points for each class are present? 

Haberman["survived"].value_counts()
print(Haberman.info())
Haberman.describe()
Haberman["age"].value_counts()
Haberman["year"].value_counts()
Haberman['survived'] = Haberman['survived'].map({1: True, 2: False})
colors = {True: 'green', False: 'red'}
(Haberman["survived"].value_counts() / len(Haberman)).plot.bar(color = ['green', 'red'])
(Haberman["survived"].value_counts() / len(Haberman))
#Seaborn plot of petal_length's PDF.
sns.FacetGrid(Haberman,hue="survived", size=6) \
   .map(sns.kdeplot, "axillary") \
   .add_legend();
plt.show();
ax = sns.kdeplot(Haberman['age'], cumulative=True)
ax = sns.kdeplot(Haberman['axillary'], cumulative=True)
ax = sns.kdeplot(Haberman['year'], cumulative=True)
plt.show()
sns.boxplot(x='survived',y='axillary', data=Haberman)
plt.show()
sns.boxplot(x='survived',y='age', data=Haberman)
plt.show()
sns.boxplot(x='survived',y='year', data=Haberman)
plt.show()
plt.close();
sns.pairplot(Haberman,hue="survived", size=3, diag_kind="kde");
plt.show()