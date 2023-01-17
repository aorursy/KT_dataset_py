import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Importing the dataset

data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

data.head()
pd.set_option('display.max_columns', None)

data.describe()
#Correlation Plot for numerical columns

cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 

        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',]

corr = data[cols].corr()

plt.figure(figsize=(10,6))

sns.heatmap(corr, annot = True, cmap = 'coolwarm')

plt.show()
#Highly correlated Variables

unstack = corr.unstack()

unstack = unstack.sort_values(kind="quicksort")

unstack[((unstack > 0.6) | (unstack < -0.6)) & (unstack != 1)]
#Visualizing Numerical Columns by Cover type

#Slope, Horizontal distance to hydrology, Vertical distance to hydrology, Horizontal distance to roadways, Horizontal distance to fire points have notable variance for the cover type

for i in range(1,10):

    sns.barplot(x = 'Cover_Type', y = cols[i], data = data,  estimator = np.average, palette='deep')

    plt.show()
#Converting the One-hot encoded wilderness area into a single column

cols = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']

wild = data[cols]

wild = pd.DataFrame(np.argmax(np.array(wild), axis = 1), columns=['Wilderness'])

wild = pd.DataFrame(wild['Wilderness'].map(lambda Label: Label+1))

wild.head()



#Converting the One-hot encoded soil type into a single column

cols = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3','Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8','Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12','Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16','Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20','Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24','Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28','Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32','Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36','Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

soil = data[cols]

soil = pd.DataFrame(np.argmax(np.array(soil), axis = 1), columns=['Soil_Type'])

soil = pd.DataFrame(soil['Soil_Type'].map(lambda Label: Label+1))



#Concatenating Wilderness and Soiltype table with the main table

cols = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']

data_ = data[cols]

data_ = pd.concat([data_,wild,soil], axis = 1)

data_.head()
plt.figure(figsize = (8,4))

sns.countplot(x = 'Wilderness', data = data_, hue = 'Cover_Type', palette='deep')

plt.legend(bbox_to_anchor = (1.05,1), loc = 2, borderaxespad = 0)

plt.show()
plt.figure(figsize = (25,8))

sns.countplot(x = 'Soil_Type', data = data_, hue = 'Cover_Type')

plt.show()
#Removing Soil type 9 and Soil type 15

data_ = data.drop(['Id','Soil_Type9','Soil_Type15','Cover_Type'],axis = 1)



#Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data_)



X = pd.DataFrame(scaled_data, columns = data_.columns)

y = data['Cover_Type']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



#Fitting the model

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1)

logit = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', max_iter = 500)

logit.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix



#Prediction

pred = logit.predict(X_test)



#Evaluation

accuracy_score(y_test, pred)
from sklearn.ensemble import RandomForestClassifier



#Fitting the model

rfc = RandomForestClassifier(n_estimators=500)

rfc.fit(X_train, y_train)
#Prediction

pred = rfc.predict(X_test)



#Evaluation

confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)