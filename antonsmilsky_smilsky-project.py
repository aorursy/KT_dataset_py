# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#imports
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense, Dropout, Flatten
import keras
from keras.models import Sequential
crimes = pd.read_csv('../input/crime.csv')
crimes
# 1. Find out the basic information - how many different types of crimes and neighbourhoods in Vancouver.
print(str(len(crimes.TYPE.unique())) + " different types of crime in the table:")
print(crimes.TYPE.unique()) 
print(str(len(crimes.NEIGHBOURHOOD.unique())) + " neighbourhoods in Vancouver:")
print(crimes.NEIGHBOURHOOD.unique())
# 2. The most frequent crimes
ax = pd.value_counts(crimes['TYPE']).plot.bar(figsize=(20,10))
ax.set_title("Most Frequent Crimes in Vancouver", fontsize=18)
ax.set_xlabel("Crimes Types", fontsize=18);
ax.set_xticklabels(pd.value_counts(crimes['TYPE']).index,rotation=60, fontsize=11)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# 3. The most frequent crimes in Vancouver by neighbourhood.
g=pd.DataFrame(crimes.groupby(['NEIGHBOURHOOD', 'TYPE']).count().reset_index().sort_values('YEAR',ascending=False))
g=g[['NEIGHBOURHOOD','TYPE','YEAR']].sort_values(['NEIGHBOURHOOD', 'YEAR'],ascending=[True,False]).rename(columns={'YEAR':'NUMBER OF OCCURENCES'}).reset_index()
del g['index']
g
# 4. At what time most of the crimes occurred
# For this question, I decided to plot the number of crime occurences for each hour on a radial plot (so that it looks like a clock). 
# The way of doing this that I found on the Internet took an array, in which the numbers from 0 through 23 (for hours) repeated different numbers of times.
# Based on that array, tha bars were plotted for each hour, and their heights were equal to the number of occurences for each number. 
# I had to do some preparation to be able to plot the data from my data set in that way.
# Also, the original code had quite of few drawbacks, so I had to fix them.
# The link to where I found this method is the following: http://qingkaikong.blogspot.com/2016/04/plot-histogram-on-clock.html
# Preparation
byHour=pd.value_counts(crimes['HOUR']).sort_index(ascending=True)

flatByHour = []
for i in range(0,len(byHour)):
    flatByHour.append(np.repeat(i, byHour[i]))
    
arr=np.concatenate(flatByHour, axis=0 )    
# Plotting the radial plot

N = 24

# Creating theta for 24 hours with an offset, so that the bars for each hour would not be centered relative to the labels
theta = np.linspace(7.5*np.pi/180, 2 * np.pi + 7.5*np.pi/180, N, endpoint=False)

# Setting the properties for a polar plot
radii, tick = np.histogram(arr, bins = 24)
width = (2*np.pi) / N

# Plotting a polar plot
plt.figure(figsize = (15, 15))
ax = plt.subplot(111, projection='polar')
plt.title("Crimes per Hour")

# Setting the ticks
thetaticks = np.arange(0, 360, 15)
ax.set_thetagrids(thetaticks)

# Plotting the bars
bars = ax.bar(theta, radii, width=width)

# Set labels to go clockwise and start from the top
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Setting the labels
ticks = ['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
ax.set_xticklabels(ticks)


for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.025, p.get_height() * 1.005), ha='center')
    
#for i, t in enumerate(ax.patches()):
#    t.set_rotation(i*45)


plt.show()
# 5. Most frequent crimes for each hour
mode = lambda x: x.mode() if len(x) > 2 else np.array(x)
crimes.groupby('HOUR')['TYPE'].agg(mode).reset_index().rename(columns={'TYPE':'The Most Frequent Crime'})
# 6. Find out in which months most of the crimes occur.
# For this I decided to plot a heatmap with years on the y-axis and months on the x-axis.
chron=pd.DataFrame(crimes.groupby(['YEAR','MONTH']).count()).reset_index().rename(columns={'TYPE':'Number of Crimes'})[['YEAR','MONTH','Number of Crimes']]
chron=chron.pivot('MONTH','YEAR','Number of Crimes')
plt.figure(figsize=(15,12))
sns.heatmap(chron,annot=True,fmt='g', cmap='Reds')
# 7. Find out whether the crime rate has been decreasing in the past years.
ax = pd.value_counts(crimes[crimes.YEAR != 2017]['YEAR']).sort_index().plot(figsize=(15,12))
ax.set_title("Number of Crimes in Vancouver in 2003-2016", fontsize=20)
ticks=crimes[crimes.YEAR != 2017]['YEAR'].unique()
ax.set_xticks(ticks)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
for i,j in pd.value_counts(crimes[crimes.YEAR != 2017]['YEAR']).sort_index().items():
    ax.annotate(str(j), xy=(i, j))
# Now some machine learning. Will try to predict what the chances are that the crime is a theft from a vehicle (the most common crime type).
prep_data=crimes
prep_data.loc[:,'Theft from Vehicle'] = prep_data['TYPE']
prep_data.loc[crimes['Theft from Vehicle'] != 'Theft from Vehicle', 'Theft from Vehicle'] = 0
prep_data.loc[crimes['Theft from Vehicle'] == 'Theft from Vehicle', 'Theft from Vehicle'] = 1
prep_data['NEIGHBOURHOOD'].replace('\s+', '_',regex=True,inplace=True)
prep_data = prep_data.dropna(subset=['NEIGHBOURHOOD'])
prep_data
numerical_features = prep_data.select_dtypes(include=[np.number]).columns
categorical_features = prep_data.select_dtypes(include=[np.object]).columns
numerical_features = numerical_features.drop(['YEAR','Theft from Vehicle'])
categorical_features = categorical_features.drop(['TYPE','HUNDRED_BLOCK'])
target = 'Theft from Vehicle'
scaler = StandardScaler()
lb = LabelBinarizer()
numeric_scaled = scaler.fit_transform(prep_data[numerical_features])
features = 0
for i in categorical_features:
    if type(features) == type(0):
        features = lb.fit_transform(prep_data[i])
    else:
        features = np.c_[features,lb.fit_transform(prep_data[i])]
features = np.c_[features,numeric_scaled]
le = LabelEncoder()
labels = le.fit_transform(prep_data[target])
X_train, X_test, y_train,y_test = train_test_split(features,labels)
# 8. Decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
# quite low,probably because not many features were selected for the prediction.
# 9. Build logistic regression - maybe it is more effective than the decision tree
logReq=LogisticRegression()
logReq.fit(X_train,y_train)
logReq.score(X_test,y_test)
# the score is higher than the decision tree but still not sufficient enough.
# 10. Build a neuron network.
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=30,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
