# Loading libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



import pandas_profiling as pp

import plotly.graph_objs as go

from plotly.offline import iplot



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler





# Import datasets

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')

# Shape of the data:

print("Train shape is: ", train.shape)

print("Test shape is: ", test.shape)
train.head()
print(train.isna().sum())
# check for missing values

print(train.isna().any().sum())

test.head()
test.isna().sum()
print(test.isna().any().sum())
train.columns
features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',

       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',

       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',

       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',

       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',

       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',

       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',

       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',

       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',

       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',

       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',

       'Soil_Type39', 'Soil_Type40']
target = ['Cover_Type']
X_train = train[features]
y_train = train[target]
#Look at a usual row from our features:



X_train.iloc[:20].head(10)
y_train.head(10)
# Train data has 56 columns and Test shape is 55. They must be equal ( in ML). Too the rows in X_train y_train and 

# tests data: X_test, y_test should be equal. 

#This will show us important later.



X_train = train.drop(['Cover_Type'], axis=1)

y_train = train['Cover_Type']

X_test = test.drop(['Id'], axis=1)
from sklearn.ensemble import RandomForestClassifier
#Split the Dataset into Training and Test Data 





X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.33, random_state=324)
print(X_train.shape)

print(y_train.shape)
print(X_test.shape)

print(y_test.shape)
m = RandomForestClassifier(n_estimators=500)

m.fit(X_train, y_train)
predict = m.predict(X_test)

print(accuracy_score(y_test, predict))
pred = m.predict(X_test)

pred
X_test.describe().T
y_test.describe().T
train.Wilderness_Area1.value_counts()
sns.set(font_scale=1.4)

train.Wilderness_Area1.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)

plt.xlabel("Wilderness Area1", labelpad=14)

plt.ylabel("Count of Wilderness Area1", labelpad=14)

plt.title("Count of Wilderness Area1 by Value", y=1.02);
train.Wilderness_Area2.value_counts()
sns.set(font_scale=1.4)

train.Wilderness_Area2.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)

plt.xlabel("Wilderness Area2", labelpad=14)

plt.ylabel("Count of Wilderness Area2", labelpad=14)

plt.title("Count of Wilderness Area2 by Value", y=1.02);
train.Wilderness_Area3.value_counts()
sns.set(font_scale=1.4)

train.Wilderness_Area3.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)

plt.xlabel("Wilderness Area3", labelpad=14)

plt.ylabel("Count of Wilderness Area3", labelpad=14)

plt.title("Count of Wilderness Area3 by Value", y=1.02);
train.Wilderness_Area4.value_counts()
sns.set(font_scale=1.4)

train.Wilderness_Area4.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)

plt.xlabel("Wilderness Area4", labelpad=14)

plt.ylabel("Count of Wilderness Area4", labelpad=14)

plt.title("Count of Wilderness Area4 by Value", y=1.02);
# sales by outlet size

Cover_To_Wilderness1 = train.groupby('Cover_Type').Wilderness_Area1.mean()



# sort by sales

Cover_To_Wilderness1.sort_values(inplace=True)

Cover_To_Wilderness1
x = Cover_To_Wilderness1.index.tolist()

y = Cover_To_Wilderness1.values.tolist()



# set axis labels

plt.xlabel('Cover_Type')

plt.ylabel('Wilderness_Area1')



# set title

plt.title('Mean Wilderness Area1 for Cover Type ')



# set xticks 

plt.xticks(labels=x, ticks=np.arange(len(x)))



plt.bar(x, y, color=['red', 'orange', 'magenta', 'green'])
fig, ax =plt.subplots(2,2, figsize=(11,10))

sns.countplot("Wilderness_Area1", data=train, ax=ax[0][0])

sns.countplot("Wilderness_Area2", data=train, ax=ax[0][1])

sns.countplot("Wilderness_Area3", data=train, ax=ax[1][0])

sns.countplot("Wilderness_Area4", data=train, ax=ax[1][1])



plt.show()
fig, ax =plt.subplots(2,2, figsize=(20,10))

sns.countplot(x="Cover_Type", hue="Wilderness_Area1", data=train, ax=ax[0][0])

sns.countplot(x="Cover_Type", hue="Wilderness_Area2", data=train, ax=ax[0][1])

sns.countplot(x="Cover_Type", hue="Wilderness_Area3", data=train, ax=ax[1][0])

sns.countplot(x="Cover_Type", hue="Wilderness_Area4", data=train, ax=ax[1][1])



plt.show()
fig, ax =plt.subplots(4,3, figsize=(22,20))

sns.boxplot("Soil_Type1", "Elevation", data=train, ax=ax[0][0])

sns.boxplot("Soil_Type2", "Aspect", data=train, ax=ax[0][1])

sns.boxplot("Soil_Type3", "Slope", data=train, ax=ax[0][2])

sns.boxplot("Soil_Type4", "Horizontal_Distance_To_Hydrology", data=train, ax=ax[1][0])

sns.boxplot("Soil_Type5", "Vertical_Distance_To_Hydrology", data=train, ax=ax[1][1])

sns.boxplot("Soil_Type6", "Horizontal_Distance_To_Roadways", data=train, ax=ax[1][2])

sns.boxplot("Soil_Type7", "Horizontal_Distance_To_Fire_Points", data=train, ax=ax[2][0])

sns.boxplot("Soil_Type8", "Hillshade_9am", data=train, ax=ax[2][1])

sns.boxplot("Soil_Type9", "Hillshade_Noon", data=train, ax=ax[2][2])

sns.boxplot("Soil_Type10", "Hillshade_3pm", data=train, ax=ax[3][0])

sns.boxplot("Soil_Type11", "Hillshade_9am", data=train, ax=ax[3][1])

sns.boxplot("Soil_Type12", "Hillshade_Noon", data=train, ax=ax[3][2])



fig.show()
train.plot(kind='box', y='Vertical_Distance_To_Hydrology', color='darkblue', figsize = (20,12)[:20])

plt.title('Vertical Distance To Hydrology', size=20)

#plt.xlabel("'Cover_Type")

plt.ylabel("Count")

plt.show()
train.plot(kind='box', y='Horizontal_Distance_To_Hydrology', color='darkblue', figsize = (20,12))

plt.title('Horizontal Distance To Hydrology', size=20)

#plt.xlabel("'Cover_Type")

plt.ylabel("Count")

plt.show()
var = train.groupby('Cover_Type').Elevation.sum()

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Cover Type')

ax1.set_ylabel('Sum of Elevation')

ax1.set_title("Cover Type by Sum of Elevations")

var.plot(kind='line')
#Stacked Column Chart

var = train.groupby(['Cover_Type','Wilderness_Area1']).Elevation.mean()[:10]#.transpose()

var.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], grid=False)
# Now we can try another approach



import pandas as pd 

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
m_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

m_classifier.fit(X_train, y_train)
predictions = m_classifier.predict(X_test)
predictions[:10]
y_test[:10]
# Measure Accuracy of the Classifier 

accuracy_score(y_true = y_test, y_pred = predictions)
# Helpful would be to find why is the predicted accuracy almost 84% in RandomForests and only 63% in DecisionTreeClassifier.

# But now it is enough. We will see this later.