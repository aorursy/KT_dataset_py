# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_June20.csv')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize = (23,9))

sns.heatmap(df.corr(), annot = True )
df.drop(['Wind_Chill(F)', 'End_Lat', 'End_Lng'], axis = 1, inplace = True)
df.info()
df.count()/3513617  #Lets see the percentage of non-null values for each column
#It seems like Number is just missing too many values, TMC is also missing a lot but we may be able to feature_engineer it along with other

df.drop(['Number', 'ID'], axis = 1, inplace = True) #ID is also useless to us
df['TMC'].value_counts() #TMC doesn't really correlate with anything and is also a classification meaning we can't really replace any values for it
df.dropna(subset = ['TMC'], inplace = True)
df.isnull().sum()
#For Temperature, Humidity, Pressure, Visibility, Wind_speed, and Precipitation we can just get their means

values = {'Temperature(F)': df['Temperature(F)'].mean(), 'Humidity(%)': df['Humidity(%)'].mean(), 'Pressure(in)': df['Pressure(in)'].mean(), 'Visibility(mi)': df['Visibility(mi)'].mean(), 'Wind_Speed(mph)' : df['Wind_Speed(mph)'].mean(), 'Precipitation(in)': df['Precipitation(in)'].mean() }

df.fillna(value = values, inplace = True)

df.isnull().sum()
#Okay I think we can just drop everything else now

df.dropna(inplace = True)

df.isnull().sum()
df.info()
#All the twilights seem to be pretty much the same thing so I'll drop them

df.drop(['Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'], axis = 1, inplace = True)
df.info() #Okay so lets start making dummy variables
df['Source'].value_counts()
source = pd.get_dummies(df['Source'])

df = pd.concat([df.drop('Source', axis = 1), source], axis = 1)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors = 'coerce')

df['End_Time'] = pd.to_datetime(df['End_Time'], errors = 'coerce')



df['Year'] = df['Start_Time'].dt.year

df['Month'] = df['Start_Time'].dt.month

df['Day'] = df['Start_Time'].dt.day

df['Hour'] = df['Start_Time'].dt.hour



df['Duration'] = round((df['End_Time']- df['Start_Time'])/np.timedelta64(1,'m'))
neg_outliers=df['Duration']<=0



df[neg_outliers] = np.nan



df.dropna(subset=['Duration'],axis=0,inplace=True)
df.drop(['Start_Time', 'End_Time'], axis = 1, inplace = True)
df.info()
df['Country'].value_counts()
#Country is useless because this is only happening in the U.S. Also County and City are just too specific for me to use and have way too many categories

df.drop(['Country', 'County', 'City'], axis = 1, inplace = True)
#Zipcode, Timezone, Airport_Code, Weather_Timestamp are also pretty useless to me

df.drop(['Zipcode', 'Timezone', 'Airport_Code', 'Weather_Timestamp'], axis = 1, inplace = True)
df.info()
#I'm going to use street just so I can see if they were on a highway or not



def location(street):

    if 'I-' in street:

        return 1

    else:

        return 0



df['highway'] = df['Street'].apply(location)

df.drop('Street', axis = 1, inplace = True)
df['highway'].head()
df.info()
#State is just too broad to affect the severity of the accident and the description just has the information in the other variables

df.drop(['State', 'Description'], axis = 1, inplace = True)
df['Side'].value_counts()
#There seems to be one random value in side so lets get rid of it and then create dummy variables for it

value = df[(df['Side'] != 'R') & (df['Side'] != 'L')].index

df.drop(value, inplace = True)

df['Side'].value_counts()
sides = pd.get_dummies(df['Side'], drop_first = True)

sides = sides.rename({'R' : 'Side'}, axis = 1)

df = pd.concat([df.drop('Side', axis = 1), sides], axis = 1)
df.info()
df['Wind_Direction'].value_counts() #Way too many directions, lets just split it up into Calm, North, South, East, West, and Variable
df['Wind_Direction'] = df['Wind_Direction'].apply(lambda dire: dire[0])

df['Wind_Direction'].value_counts()
wind = pd.get_dummies(df['Wind_Direction'], drop_first = True)

df = pd.concat([df.drop('Wind_Direction', axis = 1), wind], axis = 1)

df.info()
df['Weather_Condition'].value_counts().head(30) #Rain (and drizzle), Snow, Thunder (and storm), Cloud (and Overcast), Clear (and Fair), haze (and Smoke and fog) 
def weather(kind):

    if 'Rain' in kind or 'Snow' in kind or 'Storm' in kind or 'Thunder' in kind or 'Drizzle' in kind:

        return 'Slippery'

    elif 'Fog' in kind or 'Smoke' in kind or 'Haze' in kind or 'Mist'in kind:

        return 'Vis_obstruct'

    else:

        return 'Fair'

    

weather = df['Weather_Condition'].apply(weather)

weather.value_counts()
weather_type = pd.get_dummies(weather, drop_first = True)

weather_type.head()
df = pd.concat([df.drop('Weather_Condition', axis = 1) , weather_type], axis = 1)

df.info()
df['Sunrise_Sunset'].value_counts()
sky = pd.get_dummies(df['Sunrise_Sunset'], drop_first = True)

sky.head()
df = pd.concat([df.drop('Sunrise_Sunset', axis = 1), sky], axis = 1)

df.info()
#Now lets see if we can reduce any of the columns by seeing how correlated they are with each other

plt.figure(figsize = (26,16))

sns.heatmap(df.corr(), annot = True)
df['Turning_Loop'].value_counts() #Turning_Loop is just all zeroes so it's useless
#It also seems like Visibility and Slippery are correlated also MapQuest and MapQuest_Bing

df.drop(['Turning_Loop', 'Slippery', 'MapQuest-Bing'], axis = 1, inplace = True)
df.info()
from sklearn.model_selection import train_test_split

X = df.drop('Severity', axis = 1)

y_rfc = df['Severity']

y_nn = df['Severity']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X,y_rfc, test_size = 0.3, random_state = 101)
#Because this is Multi-Classification, lets start with Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators= 100)

rfc.fit(X_train_r, y_train_r)
from sklearn.metrics import confusion_matrix, classification_report

y_rfc_pred = rfc.predict(X_test_r)
print(confusion_matrix(y_test_r, y_rfc_pred))

print('\n')

print(classification_report(y_test_r, y_rfc_pred))
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

encoder.fit(y_nn)

y_nn = encoder.transform(y_nn)

y_nn = to_categorical(y_nn)
X_train, X_test, y_train, y_test = train_test_split(X,y_nn, test_size = 0.3, random_state = 101)
#Lets try Neural Network

from sklearn.preprocessing import MinMaxScaler

scaler  = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping



early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
model = Sequential()



model.add(Dense(39, activation = 'relu', input_dim = len(df.columns) - 1))

model.add(Dropout(rate = 0.4))



model.add(Dense(20 , activation = 'relu'))

model.add(Dropout(rate = 0.4))



model.add(Dense(10 , activation = 'relu'))

model.add(Dropout(rate = 0.4))



model.add(Dense(4, activation = 'softmax'))





model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 30, callbacks = [early_stop],batch_size = 256, validation_data = (X_test, y_test))
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))