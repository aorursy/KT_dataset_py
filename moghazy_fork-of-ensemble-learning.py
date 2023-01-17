import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# The following two lines determines the number of visible columns and 

#the number of visible rows for dataframes and that doesn't affect the code

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
train.head()
print("The number of traning examples(data points) = %i " % train.shape[0])

print("The number of features we have = %i " % train.shape[1])
train.describe()
train.drop(['Id'], axis = 1, inplace = True)

train.isnull().sum()
import seaborn as sns





import matplotlib.pyplot as plt





corr = train.corr()

f, ax = plt.subplots(figsize=(25, 25))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)
corr
import matplotlib.pyplot as plt

classes = np.array(list(train.Cover_Type.values))



def plotRelation(first_feature, sec_feature):

    

    plt.scatter(first_feature, sec_feature, c = classes, s=10)

    plt.xlabel(first_feature.name)

    plt.ylabel(sec_feature.name)

    

f = plt.figure(figsize=(25,20))

f.add_subplot(331)

plotRelation(train.Horizontal_Distance_To_Hydrology, train.Horizontal_Distance_To_Fire_Points)

f.add_subplot(332)

plotRelation(train.Horizontal_Distance_To_Hydrology, train.Horizontal_Distance_To_Roadways)

f.add_subplot(333)

plotRelation(train.Elevation, train.Vertical_Distance_To_Hydrology)

f.add_subplot(334)

plotRelation(train.Hillshade_9am, train.Hillshade_3pm)

f.add_subplot(335)

plotRelation(train.Horizontal_Distance_To_Fire_Points, train.Horizontal_Distance_To_Hydrology)

f.add_subplot(336)

plotRelation(train.Horizontal_Distance_To_Hydrology, train.Vertical_Distance_To_Hydrology)
train.head()
from sklearn.model_selection import train_test_split

x = train.drop(['Cover_Type'], axis = 1)

y = train['Cover_Type']

print( y.head() )



x_train, x_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.25, random_state=42 )
unique, count= np.unique(y_train, return_counts=True)

print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



from sklearn import decomposition



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
train.isna().sum()
###### from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

#uncomment the commented code and uncomment the commented to perform gridsearchCV

from xgboost import XGBClassifier



clf = DecisionTreeClassifier(random_state=0)



clf.fit(x_train, y_train)

print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(x_test, y_test) * 100))
test.head()



id = test['Id']

test.drop(['Id'] , inplace = True , axis = 1)



test = scaler.transform(test)
#Uncomment the commented code and comment the other line to run the grid search predict



# predictions = grid.best_estimator_.predict(test)

predictions = clf.predict(test)

out = pd.DataFrame()

out['Id'] = id

out['Cover_Type'] = predictions

out.to_csv('my_submission.csv', index=False)

out.head(5)