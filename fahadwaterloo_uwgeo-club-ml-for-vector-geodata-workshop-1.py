# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = np.array([['a','b','c'], [4,5,6], [True,False,None]])
data
df_data = pd.DataFrame(data)
df_data
import matplotlib.pyplot as plt
df_emissions = pd.DataFrame(data = [[25, 30], [45, 25], [30, 45]],

                            index = [2011, 2012, 2013],

                            columns = ['Canada', 'USA'])
df_emissions
plt.figure(figsize=(10,5))

plt.plot(df_emissions)

plt.xticks(df_emissions.index)

plt.legend(df_emissions.columns)

plt.xlabel('Time')

plt.ylabel('Emissions')

plt.title('Emissions From Cars')



plt.show()
import folium
data = (

    # Matrix of size 70 by 3, filled with numbers 

    # between -1 and 1 following a normal distribution

    np.random.normal(size=(70, 3)) *

    # Matrix of size 3 by 1, to scale values in the previous matrix

    np.array([[0.01, 0.01, 1]]) +

    # Matrix of size 3 by 1, with coordiates of University of Waterloo

    np.array([[43.471257, -80.543021, 1]])

).tolist()
# format of data: [latitude, longitude, weight]

data[0:3]
from folium.plugins import HeatMap



map_demo = folium.Map([43.471257, -80.543021],

                      #tiles='Stamen Toner',

                      zoom_start=14)

HeatMap(data).add_to(map_demo)



map_demo
# Importing OS is typically used for reading and writing from files

import os

# Print contents in the input folder

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Pandas for data importing

df_ksi = pd.read_csv('../input/KSI_CLEAN.csv')
# the info() function writes down descriptive information about the dataset such as:

#  - Number of rows

#  - Attribute names, here each attribute is represented as a column of data

#  - Data types of attributes

#  - Order of attributes



df_ksi.info()
# write down just the column names of dataset

print(df_ksi.columns)
# Shortlist of the numerical attributes

ksi_numerical = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTES', 'WEEKDAY', 

                 'LATITUDE', 'LONGITUDE', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 

                 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 

                 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'FATAL']
# Shortlist of the categorical attributes

ksi_categorical = ['Ward_Name', 'Ward_ID', 'Hood_Name', 'Hood_ID', 'Division', 

                   'District', 'STREET1', 'STREET2', 'OFFSET', 'ROAD_CLASS', 

                   'LOCCOORD', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 

                   'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 

                   'INJURY', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 

                   'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 

                   'CYCACT', 'CYCCOND']
# Shortlist of the geographic attributes

ksi_geographical = ['LATITUDE', 'LONGITUDE', 'Ward_Name', 'Hood_Name', 'District',

                   'Division', 'STREET1', 'STREET2']
# Shortlist of the Boolean attributes

ksi_boolean = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 

               'TRSN_CITY_VEH','EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 

               'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'FATAL']
df_ksi.head(3)
df_ksi.tail(3)
# Remove incidents which are listed as property damage

# in order to keep only Fatal and Non-Fatal injuries



print("KSI number of rows: ", len(df_ksi))



df_ksi = df_ksi[df_ksi['ACCLASS'] != 'Property Damage Only']



print("KSI number of rows: ", len(df_ksi))
df_ksi.sample(3)
print("Numerical Attributes: \n", str(ksi_numerical))
# Learning Exercise

# Try changing the attribute with another numerical attribute listed above:

attribute = 'YEAR'



plt.figure(figsize=(12,6))

data = df_ksi[attribute].value_counts().sort_index()

data.plot(kind='bar')

plt.show()
print("Categorical Attributes: \n", str(ksi_categorical))
# Learning Exercise

# Try changing the attribute with another categorical attribute listed above:

# INVAGE is "Age of Involved Party"

attribute = 'INVAGE'



plt.figure(figsize=(12,6))

data = df_ksi[attribute].value_counts()

data.plot(kind='bar')

plt.show()
print(ksi_boolean)
# Learning Exercise

# Change the attribute below to calculate what percentage of incidents

# reported th involvemnt of possible items listed above

attribute = 'AUTOMOBILE'



print("Percentage of incidents that involve", attribute, "=", df_ksi[attribute].sum() / df_ksi[attribute].count())
# Summary statistics, for simplicity

# we run it over all attributes,

# but they only make sense for numeric

df_ksi.describe()
# Summary statistics aggregated by month

df_ksi_monthly = df_ksi.groupby(by=['YEAR', 'MONTH'],as_index=False).sum()



# Show only the last 10 rows

df_ksi_monthly.tail(10)
import seaborn as sns
# Learning Exercise



# Try changing the independent (x) and dependent (y) 

# variables to visualize the trends in the data

# (optional: uncomment the Hue and see how that changes the plot)



plt.figure(figsize=(18,6))



sns.barplot(data = df_ksi_monthly,

            hue = 'MONTH',

            x = 'YEAR',

            # Number of events involving automobiles

            y = 'AUTOMOBILE',

            capsize = 0.2)



# Legend Placement

plt.legend(loc='best')

#plt.legend(loc='upper right')

#plt.legend(loc='upper left')



plt.show()
# Learning Exercise

# Try changing the independent (x) and dependent (y) 

# variables to visualize the trends in the data



sns.jointplot(data = df_ksi_monthly,

              kind = 'reg',

              x = 'AG_DRIV', # Aggressive and Distracted Driving Collision

              y = 'SPEEDING') # Speeding Related Collision

plt.show()
# More correlations between attributes accumulated by month



# PASSENGER Passenger Involved in Collision

# SPEEDING Speeding Related Collision

# AG_DRIV Aggressive and Distracted Driving Collision

# ALCOHOL Alcohol Related Collision

# FATAL Fatal event



sns.pairplot(df_ksi_monthly[['PASSENGER','SPEEDING','AG_DRIV','ALCOHOL','FATAL']])

plt.show()
data = df_ksi_monthly.pivot('MONTH','YEAR','FATAL')

data
plt.figure(figsize=(12,6))

sns.heatmap(data)

plt.show()
print("Geographic Attributes: \n", str(ksi_geographical))
# Learning Exercise

# Try changing the geographic attribute with another attribute listed above:

attribute = 'District'



plt.figure(figsize=(12,6))

data = df_ksi[attribute].value_counts().head(12)

data.plot(kind='bar')

plt.show()
df_ksi_geo = df_ksi[df_ksi['FATAL'] == 1]

df_ksi_geo = df_ksi_geo[['LATITUDE', 'LONGITUDE', 'FATAL']].sample(1000)

df_ksi_geo.sample(3)
lat_Toronto = df_ksi.describe().at['mean','LATITUDE']

lng_Toronto = df_ksi.describe().at['mean','LONGITUDE']
# Heatmap of fatal events

map_ksi_fatal = folium.Map(location = [lat_Toronto, lng_Toronto],

                           #tiles = 'Stamen Toner', # uncomment this code to change basemap

                           zoom_start = 11)



HeatMap(df_ksi_geo.values, min_opacity =0.4).add_to(map_ksi_fatal)

map_ksi_fatal
# Below we are visualizing the correlation of the attributes.

#  - If the correlation for two attributes is closer to positive one, 

#    they are positively correlated, meaning that if the value for one increases, 

#    the other is expected to increase as well.

#  - If the correlation for two attributes is close to negative one, 

#    they are negatively correlated, meaning that if the value for one increases, 

#    the other is expected to decrease as well.

#  - If the correlation is close to zero, there is likely no correlation.



plt.figure(figsize=(14,10))

sns.heatmap(df_ksi_monthly[ksi_boolean].corr())

plt.show()
from sklearn.model_selection import train_test_split
# the function below takes in the dataset, and a list of input columns names 

# as model input and a column name to predict for.

def split_train_test(data, X, y):

    X_all = data[X]

    y_all = data[y]

    

    X_train, X_test, y_train, y_test = train_test_split(X_all,

                                                        y_all,

                                                        test_size=0.4,

                                                        random_state=42)

    

    return X_train, X_test, y_train, y_test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
# Learning Exercise:

# Try changing the variables of the model and see how the model accuracy changes!

model_input = ['YEAR','MONTH','WEEKDAY','Hood_ID']

model_output = 'FATAL'



X_train, X_test, y_train, y_test = split_train_test(data = df_ksi,

                                                    X = model_input,

                                                    y = model_output)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
# The confusion matrix shows the proportion of incidents predicted correctly, and incorrectly



print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
accuracy_score(y_test, predictions)
from IPython.display import Image

from sklearn.externals.six import StringIO

from sklearn.tree import export_graphviz

import pydot



dot_data = StringIO()
def draw_decision_tree(tree):

    export_graphviz(tree,

                    out_file=dot_data,

                    feature_names = X_train.columns,

                    filled = True,

                    rounded = True)

    

    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    

    return Image(graph[0].create_png())
# NOTE: Depending on your device performance, you may want to

#       run the command below because it is resource intensive



# Running the command below will draw the entire descision tree!

# You can copy the image and page it in Microsoft Paint or another image viewer and see the decision paths.

draw_decision_tree(dtree)
from sklearn.ensemble import RandomForestClassifier
# Learning Exercise:

# Try changing the variables of the model and see how the model accuracy changes!

model_input = ['YEAR','MONTH','WEEKDAY','Hood_ID']

model_output = 'FATAL'



X_train, X_test, y_train, y_test = split_train_test(data = df_ksi,

                                                    X = model_input,

                                                    y = model_output)
model_input = ['YEAR','MONTH','WEEKDAY','LATITUDE','LONGITUDE']

model_output = 'FATAL'



X_train, X_test, y_train, y_test = split_train_test(data = df_ksi,

                                                    X = model_input,

                                                    y = model_output)
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
accuracy_score(y_test, predictions)
print("Number of Descision Trees  our Random Forest Model contains: ", len(rfc.estimators_))
# NOTE: Depending on your device performance, you may want to

#       run the command below because it is resource intensive



# Running the command below will draw the entire descision tree!



draw_decision_tree(rfc.estimators_[5])