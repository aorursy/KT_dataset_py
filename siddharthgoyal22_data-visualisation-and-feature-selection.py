%matplotlib inline
from IPython.display import display 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import time
data = pd.read_csv('../input/data.csv')
data.head(10)
y = data.diagnosis                          
ax = sns.countplot(y,label="number of cases")     
B, M = y.value_counts()
x = data.drop(['Unnamed: 32','id','diagnosis','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'],axis=1)
plt.figure(figsize = (30,30))
ax = sns.heatmap(x.corr(), linewidths=.5, annot=True)
a = data.drop(['Unnamed: 32','id','diagnosis','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'],axis=1)
plt.figure(figsize = (30,30))
az = sns.heatmap(a.corr(), linewidths=.5, annot=True)
#a4_dims = (11.7, 8.27)
#ax = plt.subplots(figsize=a4_dims)
plt.figure(figsize = (10,10))
sns.distplot(data["perimeter_mean"])
sns.distplot(data["radius_mean"])
plt.figure(figsize = (10,10))
sns.distplot(data["compactness_mean"])
sns.distplot(data["concavity_mean"])
plt.figure(figsize = (10,10))
sns.distplot(data["concave points_mean"])
sns.distplot(data["symmetry_mean"])
#fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
plt.figure(figsize = (10,10))

sns.boxplot(data["radius_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.boxplot(data["texture_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.boxplot(data["texture_mean"],data["diagnosis"])
#sns.boxplot(data["perimeter_mean"],data["radius_mean"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["radius_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["texture_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["perimeter_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["area_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["smoothness_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["compactness_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["concavity_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["concave points_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["symmetry_mean"],data["diagnosis"])
plt.figure(figsize = (10,10))

sns.swarmplot(data["fractal_dimension_mean"],data["diagnosis"])
sns.jointplot(x.loc[:,'concavity_mean'], x.loc[:,'concave points_mean'], kind="regg", color="#ce1414")
sns.jointplot(x.loc[:,'radius_mean'], x.loc[:,'area_mean'], kind="regg", color="#ce1414")
sns.jointplot(x.loc[:,'radius_mean'], x.loc[:,'perimeter_mean'], kind="regg", color="#ce1414")
sns.jointplot(x.loc[:,'radius_mean'], x.loc[:,'compactness_mean'], kind="regg", color="#ce1414")
sns.jointplot(x.loc[:,'fractal_dimension_mean'], x.loc[:,'texture_mean'], kind="regg", color="#ce1414")
sns.jointplot(x.loc[:,'smoothness_mean'], x.loc[:,'symmetry_mean'], kind="regg", color="#ce1414")
sns.jointplot(x.loc[:,'smoothness_mean'], x.loc[:,'texture_mean'], kind="regg", color="#ce1414")
a = data.drop(['Unnamed: 32','id','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','radius_mean','texture_worst','perimeter_worst','perimeter_mean','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','concave points_mean','symmetry_worst','fractal_dimension_worst'],axis=1)
g = sns.PairGrid(a, hue='diagnosis')
g = g.map_diag(plt.hist)
g = g.map(plt.scatter)
g = g.add_legend()
a = data.drop(['Unnamed: 32','id','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','radius_mean','texture_worst','perimeter_worst','perimeter_mean','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','concave points_mean','symmetry_worst','symmetry_mean','fractal_dimension_worst','fractal_dimension_mean'],axis=1)
g = sns.PairGrid(a, hue='diagnosis')
g = g.map_diag(plt.hist)
g = g.map(plt.scatter)
g = g.add_legend()
