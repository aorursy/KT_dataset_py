import pandas as pd   

# for reading the dataset



import matplotlib.pyplot as plt 

# for drawing the graph

%matplotlib inline  

# used so that the graph is drawn in the confined boundaries



from sklearn.ensemble import RandomForestClassifier

# used to enable the use of random forest algorithm on the dataset
ds = pd.read_csv('../input/breastcancer/breastcancer_test.csv')

ds
ds.info()



# This function is used to get a concise summary of the dataframe.
ds.describe()



# The .describe() method is use to give a descriptive exploration on the dataset
ds.shape



# Returns the number of rows and columns
ds.dtypes



# Returns the datatype of the values in each column in the dataset
ds.isnull().sum()



# .isnull() is used to check for null values in the dataset. It returns result in true/false manner.



# .sum() used with isnull() gives the combined number of null values in the dataset if any.
ds.hist(grid=True, figsize=(20,10), color='c')
ds1 = ds.drop(['Class'], axis='columns')    # Independent Variable

ds2 = ds.Class        # Dependent Variable
import seaborn as sns



# importing seaborn library for interpretation of visual data
x, y = ds1['Cl.thickness'], ds1['Cell.size']

with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex", color="k")
sns.jointplot(x=ds1['Cl.thickness'], y=ds1['Cell.shape'], data=ds1, kind="kde")
sns.pairplot(ds1)
g = sns.PairGrid(ds1)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(ds1,ds2, test_size=0.3, random_state=0) 

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train, y_train)
model.score(x_test, y_test)
y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

cm
import seaborn as sn

#importing seaborn for graphical representation of the data

f,ax = plt.subplots(figsize=(9,6))

#plt.figure(figsize=(10,7))

sn.heatmap(cm, annot=True,fmt="d",linewidths=.5,ax=ax)

plt.xlabel('Predicted')

plt.ylabel('Actual')