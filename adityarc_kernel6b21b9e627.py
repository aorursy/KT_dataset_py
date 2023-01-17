# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dfdict = {}

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.splitext(filename)[0])

        df = pd.read_csv(os.path.join(dirname, filename))

        dfdict.update({os.path.splitext(filename)[0]:df})

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dfdict["players_20"].head()
df15 = dfdict["players_15"]

df16 = dfdict["players_16"]

df17 = dfdict["players_17"]

df18 = dfdict["players_18"]

df19 = dfdict["players_19"]

df20 = dfdict["players_20"]
for x in df20.columns:

    print(x,",",end=" ")
vals = set(df20['wage_eur'])



print(vals)
fifa20_features = ['age','height_cm','weight_kg','nationality','international_reputation','weak_foot','work_rate','club','skill_moves']



fifa20_potential = ['age','height_cm','weight_kg','nationality','international_reputation','weak_foot','work_rate','club','skill_moves','overall']



fifa20_value = ['age','height_cm','weight_kg','nationality','international_reputation','weak_foot','work_rate','club','skill_moves','overall','potential']



fifa20_wage = ['age','height_cm','weight_kg','nationality','international_reputation','weak_foot','work_rate','club','skill_moves','overall','potential','value_eur']



X1 = df20[fifa20_features]



X2 = df20[fifa20_potential]



X3 = df20[fifa20_value]



X4 = df20[fifa20_wage]





y1 = df20['overall']



y2 = df20['potential']



y3 = df20['value_eur']



y4 = df20['wage_eur']
print(set(X1['work_rate']))
# Get list of categorical variables

s1 = (X1.dtypes == 'object')

object_cols1 = list(s1[s1].index)



print("Categorical variables for Base Stats:")

print(object_cols1)





s2 = (X2.dtypes == 'object')

object_cols2 = list(s2[s2].index)



print("Categorical variables for Potential:")

print(object_cols2)





s3 = (X3.dtypes == 'object')

object_cols3 = list(s3[s3].index)



print("Categorical variables for Player Value:")

print(object_cols3)





s4 = (X4.dtypes == 'object')

object_cols4 = list(s4[s4].index)



print("Categorical variables for Player Wage:")

print(object_cols4)
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

# Apply label encoder to each column with categorical data

label_X1 = X1.copy()

label_X2 = X2.copy()

label_X3 = X3.copy()

label_X4 = X4.copy()

label_encoder = LabelEncoder()

for col in object_cols1:

    label_X1[col+'_features'] = label_encoder.fit_transform(X1[col])

for col in object_cols2:

    label_X2[col+'_features'] = label_encoder.fit_transform(X2[col])

for col in object_cols3:

    label_X3[col+'_features'] = label_encoder.fit_transform(X3[col])

for col in object_cols4:

    label_X4[col+'_features'] = label_encoder.fit_transform(X4[col])
countryname = set(X1['nationality'])

countrycode = set(label_X1['nationality_features'])

countrydict = dict(zip(countryname,countrycode))



wrname = set(X1['work_rate'])

wrcode = set(label_X1['work_rate_features'])

wrdict = dict(zip(wrname,wrcode))



clubname = set(X1['club'])

clubcode = set(label_X1['club_features'])

clubdict = dict(zip(clubname,clubcode))

wrname
wrdict
clubdict
label_X1.head()

label_X2.head()
label_X3.head()
label_X4.head()
from sklearn.model_selection import train_test_split



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

#lX1 = label_X1['age','height_cm','weight_kg','international_reputation','weak_foot','skill_moves','nationality_features','work_rate_features','club_features']

#lX2 = label_X2['age','height_cm','weight_kg','international_reputation','weak_foot','skill_moves','overall','nationality_features','work_rate_features','club_features']



lX1 = label_X1.copy()

lX2 = label_X2.copy()

lX3 = label_X3.copy()

lX4 = label_X4.copy()

lX1.drop(['nationality','work_rate','club'],axis=1,inplace=True)

lX2.drop(['nationality','work_rate','club'],axis=1,inplace=True)

lX3.drop(['nationality','work_rate','club'],axis=1,inplace=True)

lX4.drop(['nationality','work_rate','club'],axis=1,inplace=True)

train_X1, val_X1, train_y1, val_y1 = train_test_split(lX1, y1, random_state = 0)

train_X2, val_X2, train_y2, val_y2 = train_test_split(lX2, y2, random_state = 0)

train_X3, val_X3, train_y3, val_y3 = train_test_split(lX3, y3, random_state = 0)

train_X4, val_X4, train_y4, val_y4 = train_test_split(lX4, y4, random_state = 0)
from xgboost import XGBRegressor



basestats_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

basestats_model.fit(train_X1, train_y1)



potential_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

potential_model.fit(train_X2, train_y2)



playervalue_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

playervalue_model.fit(train_X3, train_y3)



playerwage_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

playerwage_model.fit(train_X4, train_y4)
from sklearn.metrics import mean_absolute_error



base_predictions = basestats_model.predict(val_X1)

print("Mean Absolute Error: " + str(mean_absolute_error(base_predictions, val_y1)))



potential_predictions = potential_model.predict(val_X2)

print("Mean Absolute Error: " + str(mean_absolute_error(potential_predictions, val_y2)))



playervalue_predictions = playervalue_model.predict(val_X3)

print("Mean Absolute Error: " + str(mean_absolute_error(playervalue_predictions, val_y3)))



playerwage_predictions = playerwage_model.predict(val_X4)

print("Mean Absolute Error: " + str(mean_absolute_error(playerwage_predictions, val_y4)))
val_y3.head()

print("\n",playervalue_predictions[0:5])
for val in train_X1:

    print(val)
for val in train_X2:

    print(val)
for val in train_X3:

    print(val)
for val in train_X4:

    print(val)
import math



print("First Name: \t",end='')

FirstName = input()

print("Last Name:\t",end='')

LastName = input()

print("Enter the Age of the player (between 18 to 45):\t",end='')

a = int(input())

print("Enter the Player's height in centimeters:\t",end='')

h = int(input())

print("Enter the Player's weight in Kgs:\t",end='')

w = int(input())

print("Enter the Player's international reputation (1 to 5) \n [ranging from 1-Novice to 5-Gifted]:\t",end='')

ir = int(input())

print("Enter the Player's weak foot ability (1 to 5) \n [ranging from 1-Novice to 5-Gifted]:\t",end='')

wf = int(input())

print("Enter the Player's skill moves ability (1 to 5) \n [ranging from 1-Novice to 5-Gifted]:\t",end='')

sm = int(input())

print("Enter the Player's nationality:\t",end='')

nationality = input()

print("Enter the Player's work rate for Attack and Defence in the format \"att/def\" \n The work rate has to be any one for the following three (High,Medium,Low)\t",end='')

workrate = input()

#High/Low

print("Enter the Name of the Current Club:\t",end='')

club = input()



n = int(countrydict[nationality])

wr = int(wrdict[workrate])

c = int(clubdict[club])





lst1 = [[int(a),int(h),int(w),int(ir),int(wf),int(sm),int(n),int(wr),int(c)]]

l1 = pd.DataFrame(lst1,columns=train_X1.columns)

overall = math.floor(basestats_model.predict(l1))

baseerror = str(round(mean_absolute_error(base_predictions, val_y1)/2,2))



lst2 = lst1

lst2[0].insert(6,math.floor(overall))

l2 = pd.DataFrame(lst2,columns=train_X2.columns)

potential = math.floor(potential_model.predict(l2))

potentialerror = str(round(mean_absolute_error(potential_predictions, val_y2)/2,2))



lst3 = lst2

lst3[0].insert(7,math.floor(potential))

l3 = pd.DataFrame(lst3,columns=train_X3.columns)

playerval = math.floor(playervalue_model.predict(l3))

vallen = len(str(playerval))

if (vallen>6):

    vl = round((playerval/1000000),2)

    strval = str(vl) + " million"

elif (vallen <= 6 and len > 3):

    vl = round((playerval/1000),2)

    strval = str(vl) + " thousand"

else:

    strval = str(playerval)

valerror = str(round(mean_absolute_error(playervalue_predictions, val_y3)/2,2))



lst4 = lst3

lst4[0].insert(8,math.floor(playerval))

l4 = pd.DataFrame(lst4,columns=train_X4.columns)

playerwage = math.floor(playerwage_model.predict(l4))

wageerror = str(round(mean_absolute_error(playerwage_predictions, val_y4)/2,2))







print("\n\nPLAYER CARD \n\nName: {0} {1}\tAge: {2}\nNationality: {3}\tClub: {4}".format(FirstName,LastName,a,nationality,club))

print("Height(cms): {0}\tWeight(kgs): {1}".format(h,w))

print("\n\nPLAYER PREDICTED STATS \n\n")

print("Player Base Stats:\t {0} with Margin of (+/-{1})".format(overall,baseerror))

print("Player Future Potential:\t {0} with Margin of (+/-{1})".format(potential,potentialerror))

print("Player Est. Value (in Eur):\t {0} with Margin of (+/-{1})".format(strval,valerror))

print("Player Est. Wage (in Eur):\t {0} with Margin of (+/-{1})".format(playerwage,wageerror))


