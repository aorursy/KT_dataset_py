import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression
ds = pd.read_csv('../input/mtcarscsv/mtcars.csv')

ds
ds.info()



# This function is used to get a concise summary of the dataframe.
ds.shape



# Returns the number of rows and columns
ds.dtypes



# Returns the datatype of the values in each column in the dataset
ds.isnull().sum()



# .isnull() is used to check for null values in the dataset. It returns result in true/false manner.



# .sum() used with isnull() gives the combined number of null values in the dataset if any.
ds.hist(grid=True, figsize=(20,10), color='purple')
from sklearn.model_selection import train_test_split

# This library is used to import the method to dvide the dataset into training and testing part.
ds1 = ds.drop(['mpg', 'fast', 'cars', 'carname'], axis='columns')  #independent variables

# The drop() function is used to drop all those columns that are mentioned in the parenthesis



ds2 = ds.fast  #dependent variable
X_train, X_test, Y_train, Y_test = train_test_split(ds1, ds2, train_size=0.7, random_state=0)
md = LogisticRegression()
md.fit(X_train, Y_train)
md.score(X_test, Y_test) # It is used to check the ccuracy of a model used
md.predict_proba(X_test)

# This function is used to predict the probability of occurance or not occuring of a scenario
y_predict = md.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_predict)

cm
import seaborn as sn

sn.heatmap(cm,annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')