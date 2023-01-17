import pandas as pd   

# for reading the dataset



import matplotlib.pyplot as plt 

# for drawing the graph

%matplotlib inline  

# used so that the graph is drawn in the confined boundaries



from sklearn import tree   

# used to import Decision Tree model
ds = pd.read_csv('../input/diabetics/PimaIndiansDiabetes.csv')

ds
ds1 = ds.drop(['Class'], axis='columns')

ds2 = ds.Class

ds.hist(grid=True, figsize=(10,5), color='olive')
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(ds1,ds2, test_size=0.3, random_state=0) 

#random state = 0 is used to fix the dataset part used for testing and training
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
y = model.predict(x_test)
model.score(x_test, y_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y)

cm
import seaborn as sn

sn.heatmap(cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Actual')