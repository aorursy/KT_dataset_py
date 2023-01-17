# loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from IPython.display import display
from sklearn.tree import export_graphviz
import random
# loading training data
df_clean = pd.read_csv('../input/ALL_LEAGUESx90.csv', encoding="ISO-8859-1", sep = ',')
df_clean['League'] = df_clean['League'].str.replace(" ","") #remove spaces
df_clean['Age now']= df_clean['Age now'] + 1 # Needed to uptdate the age since scraping took place in 2017
df_clean.iloc[0:5,:]
df_clean.shape
df_clean = df_clean[df_clean.Ratingf1 != 0]
df_clean = df_clean[df_clean.Mins > 1500]

fig, axes = plt.subplots(ncols=2, figsize = (12,5))
plt.subplot(1, 2, 1)
boxplot = df_clean.boxplot(column='Ratingf1');
plt.title('Boxplot of t+1 rating')
plt.subplot(1, 2, 2)
df_clean.Classf1.hist();
plt.title('Distribution of t+1 classes')
plt.show()
df_clean.Ratingf1.quantile([0.25,0.5,0.75,0.9])
#All of this to classify the players based on their rating (done in excel)
df_clean = df_clean[(df_clean.Position1 == "F")|(df_clean.Position2 == "F")|(df_clean.Position3 == "F")]
df_clean.shape
# This is the dimension of the sample of strikers I'll use
# create design matrix X and target vector y
X = np.array(df_clean.iloc[:,7:87])
y1 = np.array(df_clean['Classf1'])

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators= 10000)

rf.fit(X, y1)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 3), rf.feature_importances_),
                 list(df_clean.iloc[:,7:87])), reverse=True))
print("The ratio of the correctly predicted data points to all the predicted data points is:") 
print(round(rf.score(X,y1),3))
df = pd.read_csv('../input/ALL_LEAGUESx90.csv', encoding="ISO-8859-1", sep = ',')
df = df[(df.Position1 == "F")|(df.Position2 == "F")|(df.Position3 == "F")]
df17 = df[df.Year == 17]
df17 = df17[df17.Mins>1500]
df17['League'] = df17['League'].str.replace(" ","") #remove spaces
df17['Age now']= df17['Age now'] + 1 # Needed to update the age
X_17 = np.array(df17.iloc[:,7:87])
pred = rf.predict(X_17)
out_18 = pd.DataFrame({'Player':np.array(df17.iloc[:,0]),'Age':np.array(df17.iloc[:,1]),'Rating_17':np.array(df17.iloc[:,88]),'Predicted_performance':pred})
out_18 = out_18.sort_values(by=['Predicted_performance'])
fig, axes = plt.subplots(ncols=2, figsize = (12,5))
plt.subplot(1, 2, 1)
round(out_18.Predicted_performance).hist()
plt.title('Distribution of predicted classes in 2017-2018')
plt.subplot(1, 2, 2)
df_clean.Classf1.hist();
plt.title('Distribution of classes in my sample')
plt.show()
best18 = out_18[(out_18.Rating_17 < 715)&(out_18.Age < 30)]
best18 = best18[best18.Predicted_performance < best18.Predicted_performance.quantile(0.2)]
best18 = best18.iloc[:,0:3]
best18.iloc[:,:]
### Checking what happened in 2017-2018:

# I first import the dataset with 2017-2018 data and clean the data as before:
df_actual = pd.read_csv('../input/ALL_LEAGUESx90_18.csv', encoding="ISO-8859-1", sep = ',')
df_actual = df_actual[df_actual['Position1'].str.contains("F")]

# I now match the supposed talents with their real observation:
searchfor = best18['Player'].str[:11]
best18_actual = df_actual[df_actual['Player'].str.contains('|'.join(searchfor))]
best18_actual = best18_actual.iloc[:,[0,6,88]]
out = best18.merge(best18_actual, how='outer')
out = out.sort_values(by=['Player'])
out['Player']= np.where(out['Year'] != 18, out['Player'] + " (2017)", out['Player']) 
out.rename(columns={'Rating': 'Rating_18'}, inplace=True)
out.iloc[:,[0,1,2,4]]
print("My group have average Rating in 2018:",round(best18_actual["Rating"].mean()))
# Counterfactual: How good is my prediction actually?
# a) I keep only the players who are 'reasonable' candidates:
subsample = df17[(df17['Rating'] > 650)&(df17.Rating < 715)&(df17['Age now'] < 30)]
# b) Randomly getting 17 out of these gives us average Rating in 2018 of : 
test18 = subsample.sample(n=17)
searchfor = test18['Player'].str[:12]
test18_actual = df_actual[df_actual['Player'].str.contains('|'.join(searchfor))]
test18_actual.iloc[:,[0,6,88]]
print("The random sample has average Rating in 2018:",round(test18_actual["Rating"].mean()))
# Cleaning the data:
df_actual = df_actual[df_actual.Mins>1500];
df_actual['League'] = df_actual['League'].str.replace(" ","") #remove spaces
# Potential future talents, from least likely to most likely based on their 2017-2018 rating:
X_18 = np.array(df_actual.iloc[:,7:87])
pred = rf.predict(X_18)
out_19 = pd.DataFrame({'Player':np.array(df_actual.iloc[:,0]),'Age':np.array(df_actual.iloc[:,1]),'Rating_18':np.array(df_actual.iloc[:,88]),'Predicted_performance':pred})
out_19 = out_19.sort_values(by=['Rating_18'])
best19 = out_19[(out_19.Rating_18 < 715)&(out_19.Age < 30)]
best19 = best19[best19.Predicted_performance < best19.Predicted_performance.quantile(0.2)]

best19.iloc[:,[0,1,2]]
fig, axes = plt.subplots(ncols=2, figsize = (12,5))
plt.subplot(1, 2, 1)
round(out_19.Predicted_performance).hist();
plt.title('Distribution of predicted classes in 2018-2019')
plt.subplot(1, 2, 2)
df_clean.Classf1.hist();
plt.title('Distribution of classes in my sample')
plt.show()