# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))

        
Impact_data = pd.read_csv('/kaggle/input/asteroid-impacts/impacts.csv')

Orbit_data = pd.read_csv('/kaggle/input/asteroid-impacts/orbits.csv')

Impact_data.info()

Orbit_data.info()
dataIO = pd.concat([Impact_data, Orbit_data], axis=1, sort=False)

dataIO.columns = ['Object_Name', 'Period_Start', 'Period_End', 'Possible_Impacts',

       'Cumulative_Impact_Probability', 'Asteroid_Velocity',

       'Asteroid_Magnitude', 'Asteroid_Diameter(km)',

       'Cumulative_Palermo_Scale', 'Maximum_Palermo_Scale',

       'Maximum_Torino_Scale', 'Object_Name', 'Object_Classification',

       'Epoch(TDB)', 'Orbit_Axis(AU)', 'Orbit_Eccentricity',

       'Orbit_Inclination(deg)', 'Perihelion_Argument(deg)',

       'Node_Longitude(deg)', 'Mean_Anomoly(deg)',

       'Perihelion_Distance(AU)', 'Aphelion_Distance(AU)',

       'Orbital_Period(yr)', 'Minimum_Orbit_Intersection_Distance(AU)',

       'Orbital_Reference', 'Asteroid_Magnitude']

dataIO.columns
Impact_data.columns = ['Object_Name', 'Period_Start', 'Period_End', 'Possible_Impacts',

       'Cumulative_Impact_Probability', 'Asteroid_Velocity',

       'Asteroid_Magnitude', 'Asteroid_Diameter(km)',

       'Cumulative_Palermo_Scale', 'Maximum_Palermo_Scale',

       'Maximum_Torino_Scale']

Impact_data.head()



Impact_data.info()
Impact_data.Period_Start.value_counts()
Impact_data.Period_End.value_counts()
import matplotlib.pyplot as plt

import pandas as pd



Impact_data.plot(kind='scatter',x='Asteroid_Velocity',y='Possible_Impacts',color='red')

plt.show()
Impact_data.Object_Name.head(100).unique()
ListOfObj= list(Impact_data['Object_Name'].head(100).unique())



Period_Start_ratio = []

for i in ListOfObj:

    x = Impact_data[Impact_data['Object_Name']==i]

    Period_Start_rate = sum(x.Period_Start)/len(x)

    Period_Start_ratio.append(Period_Start_rate)

dataObj = pd.DataFrame({'object_list': ListOfObj,'Period_Start_ratio':Period_Start_ratio})





new_index = (dataObj['Period_Start_ratio'].sort_values(ascending=False)).index.values

sorted_data = dataObj.reindex(new_index)

                
# visualization

plt.figure(figsize=(40,20))

sns.pointplot(x=sorted_data['object_list'], y=sorted_data['Period_Start_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Object Names')

plt.ylabel('Period Start')

plt.title('Period Start for Object')



Orbit_data.columns = ['Object_Name', 'Object_Classification',

       'Epoch(TDB)', 'Orbit_Axis(AU)', 'Orbit_Eccentricity',

       'Orbit_Inclination(deg)', 'Perihelion_Argument(deg)',

       'Node_Longitude(deg)', 'Mean_Anomoly(deg)',

       'Perihelion_Distance(AU)', 'Aphelion_Distance(AU)',

       'Orbital_Period(yr)', 'Minimum_Orbit_Intersection_Distance(AU)',

       'Orbital_Reference', 'Asteroid_Magnitude']



Orbit_data.head()



Orbit_data.info()
Orbit_data.Object_Name.value_counts()
name_count = Counter(Orbit_data.Object_Name)

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)
plt.figure(figsize=(30,15))

ax= sns.stripplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Object Names')

plt.ylabel('Frequency')

plt.title('Most common 15 Object Name')

print(most_common_names)
Period_End_ratio = []

for i in ListOfObj:

    x = Impact_data[Impact_data['Object_Name']==i]

    Period_End_rate = sum(x.Period_End)/len(x)

    Period_End_ratio.append(Period_End_rate)

dataPer = pd.DataFrame({'object_list':ListOfObj ,'Period_End_ratio': Period_End_ratio})



new_index = (dataPer['Period_End_ratio'].sort_values(ascending=False)).index.values

sorted_data2 = dataPer.reindex(new_index)
plt.figure(figsize=(30,20))

sns.pointplot(x=sorted_data2['object_list'], y=sorted_data2['Period_End_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Object Names')

plt.ylabel('Period End')

plt.title('Period End for Object')
Cumulative_Impact_Probability = []

Asteroid_Velocity = []

Asteroid_Magnitude = []

Possible_Impacts = []

for i in ListOfObj:

    x = Impact_data[Impact_data['Object_Name']==i]

    Cumulative_Impact_Probability.append(sum(x.Cumulative_Impact_Probability)/len(x))

    Asteroid_Velocity.append(sum(x.Asteroid_Velocity) / len(x))

    Asteroid_Magnitude.append(sum(x.Asteroid_Magnitude) / len(x))

    Possible_Impacts.append(sum(x.Possible_Impacts) / len(x))
# visualization

f,ax = plt.subplots(figsize = (18,25))

sns.pointplot(x=Cumulative_Impact_Probability,y=ListOfObj,color='green',alpha = 0.5,label='Cumulative_Impact_Probability' )

sns.pointplot(x=Asteroid_Velocity,y=ListOfObj,color='blue',alpha = 0.7,label='Asteroid_Velocity')

sns.pointplot(x=Asteroid_Magnitude,y=ListOfObj,color='cyan',alpha = 0.6,label='Asteroid_Magnitude')

sns.pointplot(x=Possible_Impacts,y=ListOfObj,color='red',alpha = 0.6,label='Possible_Impacts')

ax.legend(loc='lower right',frameon = True)

ax.set(xlabel='Percentage of Asteroids', ylabel='Asteroids',title = "Name Of Asteroids percentage ")

Orbit_data.Object_Name.dropna(inplace = True)

labels = Orbit_data.Object_Name.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = Orbit_data.Object_Name.value_counts().values

# visual

plt.figure(figsize = (12,12))

plt.pie(sizes, explode=None, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Asteroids According to Names',color = 'blue',fontsize = 15)



Impact_data.describe()
import matplotlib.pyplot as plt

import seaborn as sns



colormap = plt.cm.viridis

sns.heatmap(Impact_data.corr(), annot=True, cmap=colormap)
TorinoScale = Impact_data['Maximum_Torino_Scale']

print(TorinoScale.max())



names = Impact_data['Object_Name']



Impact_data.drop(['Object_Name','Maximum_Torino_Scale','Asteroid_Magnitude'], axis=1, inplace=True)



#calculating the complete period for each asteroid

Impact_data['Period'] = Impact_data['Period_End']-Impact_data['Period_Start']

Impact_data.drop(['Period_End', 'Period_Start','Maximum_Palermo_Scale'], axis=1, inplace=True)
x = Impact_data['Period']

y = Impact_data['Possible_Impacts']

sns.regplot(x,y)

sns.regplot(Impact_data['Period'],Impact_data['Cumulative_Palermo_Scale'])
MaxDia = Impact_data['Asteroid_Diameter(km)'].max()



j = 0

for i in Impact_data['Asteroid_Diameter(km)']:

    if i == MaxDia:

        break

    j += 1



print(names[j])

Impact_data.loc[j]
MaxProbab = Impact_data['Cumulative_Impact_Probability'].max()



j = 0

for i in Impact_data['Cumulative_Impact_Probability']:

    if i == MaxProbab:

        break

    j += 1



print(names[j])    

Impact_data.loc[j]
MaxImpact = Impact_data['Possible_Impacts'].max()



j = 0

for i in Impact_data['Possible_Impacts']:

    if i == MaxImpact:

        break

    j += 1



print(names[j])    

Impact_data.loc[j]
from sklearn.linear_model import SGDClassifier,SGDRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.calibration import CalibratedClassifierCV

#import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm
Impact_Pred = Impact_data

y = Impact_Pred['Possible_Impacts'].values

X = Impact_Pred

Impact_Pred.head()
#split data in to train and test 50-50

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(Xtrain)

Xtrain = scaler.transform(Xtrain)

Xtest = scaler.transform(Xtest)
#Stochastic Gradient Descent

sdg = SGDRegressor()

sdg.fit(Xtrain, ytrain)

sgd = sdg.predict(Xtest)

print(sdg.score(Xtest, ytest))
#Random Forest

radm = RandomForestRegressor()

radm.fit(Xtrain, ytrain)

rmf = radm.predict(Xtest)

print(radm.score(Xtest, ytest))
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(Xtrain, ytrain, verbose=False)

xgb = my_model.predict(Xtest)

print(my_model.score(Xtest, ytest))
predictions = my_model.predict(Xtest)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, ytest)))
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')



#Decision trees

model=DecisionTreeClassifier()

model.fit(Xtrain, ytrain)

dt_prediction=model.predict(Xtest)

accuracy= print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction,ytest))



sns.heatmap(Impact_data[Impact_data.columns[:11]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(20,18)

plt.show()

#Stochastic Gradient 

from matplotlib.pyplot import plot

plot(sgd)
#Decision trees

plot(dt_prediction)
#XG boost

plot(xgb)
#random forest

plot(rmf)
import numpy as np

indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(Impact_Pred.shape[1]):

    print('%d. feature %d %s (%f)' % (f+1 , indices[f], Impact_Pred.columns[indices[f]],

                                      radm.feature_importances_[indices[f]]))