#import gmplot
import numpy as np # linear algebra
import pyproj as pp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.plotly as py
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing Dataset into a DataFrame
dfcrime = pd.read_csv('..//input/crime.csv')

# Cleaning & Transforming the data
dfcrime['HOUR'].fillna(00, inplace = True)
dfcrime['NEIGHBOURHOOD'].fillna('N/A', inplace = True)
dfcrime['HUNDRED_BLOCK'].fillna('N/A', inplace = True)
del dfcrime['MINUTE']
dfcrime['NeighbourhoodID'] = dfcrime.groupby('NEIGHBOURHOOD').ngroup().add(1)
dfcrime['CrimeTypeID'] = dfcrime.groupby('TYPE').ngroup().add(1)
dfcrime['Incident'] = 1
dfcrime['Date'] = pd.to_datetime({'year':dfcrime['YEAR'], 'month':dfcrime['MONTH'], 'day':dfcrime['DAY']})
dfcrime['DayOfWeek'] = dfcrime['Date'].dt.weekday_name
dfcrime['DayOfWeekID'] = dfcrime['Date'].dt.weekday
dfpred = dfcrime[(dfcrime['YEAR'] >= 2017)]
dfcrime = dfcrime[(dfcrime['YEAR'] < 2017)]

# Calling a dataframe results
dfcrime.head()

%matplotlib inline
# Setting plot style for all plots
plt.style.use('seaborn')

# Count all crimes and group by year
dfCrimeYear = pd.pivot_table(dfcrime, values=["Incident"],index = ["YEAR"], aggfunc='count')

# Graph results of Year by Crimes
f, ax = plt.subplots(1,1, figsize = (12, 4), dpi=100)
xdata = dfCrimeYear.index
ydata = dfCrimeYear
ax.plot(xdata, ydata)
ax.set_ylim(ymin=0, ymax=60000)
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.title('Vancouver Crimes from 2003-2017')
plt.show()
%matplotlib inline
# Pivoting dataframe by Crime Type to calculate Number of Crimes
dfCrimeType = pd.pivot_table(dfcrime, values=["Incident"],index = ["TYPE"], aggfunc='count')

dfCrimeType = dfCrimeType.sort_values(['Incident'], ascending=True)

# Create bar graph for number of crimes by Type of Crime
crimeplot = dfCrimeType.plot(kind='barh',
               figsize = (6,8),
               title='Number of Crimes Committed by Type'
             )

plt.rcParams["figure.dpi"] = 100
plt.legend(loc='lower right')
plt.ylabel('Crime Type')
plt.xlabel('Number of Crimes')
plt.show(crimeplot)
%matplotlib inline
# Count of Incidents per Year By Type
dfPivYearType = pd.pivot_table(dfcrime, values=["Incident"],index = ["YEAR", "TYPE"], aggfunc='count')

dfCrimeByYear = dfPivYearType.reset_index().sort_values(['YEAR','Incident'], ascending=[1,0]).set_index(["YEAR", "TYPE"])

# Plot data on box whiskers plot
NoOfCrimes = dfCrimeByYear["Incident"]
plt.boxplot(NoOfCrimes)
plt.show()
%matplotlib inline
# Adding Days Lookup
days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']

# Grouping dataframe by Day of Week ID and plotting
dfcrime.groupby(dfcrime["DayOfWeekID"]).size().plot(kind='barh')

# Customizing Plot 
plt.ylabel('Days of the week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes')
plt.title('Number of Crimes by Day of the Week')
plt.show()
%matplotlib inline
# Create a pivot table with month and category. 
dfPivYear = dfcrime.pivot_table(values='Incident', index='TYPE', columns='YEAR', aggfunc=len)

fig, ax = plt.subplots(1, 1, figsize = (12, 6), dpi=300)
plt.title('Type of Crime By Year', fontsize=16)
plt.tick_params(labelsize=10)

sns.heatmap(
    dfPivYear.round(), 
    linecolor='lightgrey',
    linewidths=0.1,
    cmap='viridis', 
    annot=True, 
    fmt=".0f"
);

# Remove labels
ax.set_ylabel('Crime Type')    
ax.set_xlabel('Year')

plt.show()
'''
import gmplot 
# Clean the data of zero points of latitude amd longitude as we can not plot those coordinates
dfCoord = dfcrime[(dfcrime.Latitude != 0.0) & (dfcrime.Longitude != 0.0)]

# Assign datapoints in variables
latitude = dfCoord["Latitude"]
longitude = dfCoord["Longitude"]

# Creating the location we would like to initialize the focus on. 
# Parameters: Lattitude, Longitude, Zoom
gmap = gmplot.GoogleMapPlotter(49.262, -123.135, 11)

# Overlay our datapoints onto the map
gmap.heatmap(latitude, longitude)

# Generate the heatmap into an HTML file
gmap.draw("crime_heatmap.html")
'''
%matplotlib inline
# Crime count by Category per year
dfPivCrimeDate = dfcrime.pivot_table(values='Incident'
                                     ,aggfunc=np.size
                                     ,columns='TYPE'
                                     ,index='YEAR'
                                     ,fill_value=0)
plo = dfPivCrimeDate.plot(figsize=(15, 15), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
# New DataFrame to filter out columns needed
dfRandomF = dfcrime

# Split data for training and testing
#dfRandomF['train'] = np.random.uniform(0, 1, len(dfRandomF)) <= .70

X = dfRandomF[['YEAR', 'MONTH', 'DAY','HOUR', 'NeighbourhoodID']]

Y = dfRandomF[['TYPE']]

# To create a training and testing set, I am splitting the data
# by 70% training and 30% testing
X_train , X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 5)

print('Number of observations and columns in the training data:', X_train.shape, y_train.shape)
print('Number of observations and columns in the testing data:',X_test.shape, y_test.shape)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 5,
                               max_depth=5, min_samples_leaf=8)
clf_gini.fit(X_train, y_train)
# Adding prediction test
y_pred_gn = clf_gini.predict(X_test)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 5,
                                    max_depth=5, min_samples_leaf=8)
clf_entropy.fit(X_train, y_train)

# Adding prediction test
y_pred_en = clf_entropy.predict(X_test)
# Random values for prediction 
clf_gini.predict([[2017,1,5,15.0,12]])
# Using the same parameters for predition
dfpred[(dfpred['YEAR'] == 2017) & 
        (dfpred['MONTH'] == 1) & 
        (dfpred['DAY'] == 5) & 
        (dfpred['HOUR'] == 15.0) &
        (dfpred['NeighbourhoodID'] == 12)]
dfpred[(dfpred['YEAR'] == 2017) & 
        (dfpred['MONTH'] == 1) & 
        (dfpred['DAY'] == 5) & 
        (dfpred['NeighbourhoodID'] == 12)]
print ('Accuracy is', accuracy_score(y_test,y_pred_gn)*100, '%')
print ('Accuracy is', accuracy_score(y_test,y_pred_en)*100, '%')
'''
# Style Report
from IPython.core.display import HTML
css_file = 'style.css'
HTML(open(css_file, 'r').read())
'''