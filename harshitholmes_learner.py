#Import the Libraries 
import os
import re
import pandas as pd
import numpy as np

#Select and give classifier the model 
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as aus

#To avoid unrelvent errors
import warnings
warnings.filterwarnings('ignore')

#Take input has the machine input
#Gender = input('Enter Male or Female :')

#age = str(int(input('Enter the player age :')))
#height = int(input('Enter the player height :'))
#weight = int(input('Enter the player weight :'))


#won = input("You want winner or fresher in yes or no:")

#Load the data from the csv file
file = pd.read_csv('../input/kcp1.csv')

#Here pick the root for the future apt
cat = file[file['Sex'] == 'M']

#Get the features and labels because here the ine big drawback will be the lenght of both will different.
#So, select from second level division
Features = cat[(cat['Age'] >= 24) & (cat['Height'] >= 170) & (cat['Weight'] >= 75)]

#Connecting the Label and Features
cd = [cat, Features]

#Data Cleaning from original data
#Retrive the salutation and Eliminating unused variable
for eachplayer in cd :
    eachplayer['Player_Name'] = eachplayer.Name.str.extract(' ([A-Za-z]+)\. ', expand = False)
    eachplayer['Player_Name'] = pd.factorize(eachplayer['Player_Name'])[0]
    eachplayer['sex'] = eachplayer.Sex.str.extract(' ([A-Za-z]+)\. ', expand = False)
    eachplayer['sex'] = pd.factorize(eachplayer['sex'])[0]

#Deleting the specific coulmn from our dataset
train = cat.drop(['ID', 'Games', 'City', 'Sport', 'Event'], axis = 1)
test = cat.drop(['ID', 'Games', 'City', 'Sport', 'Event'], axis = 1)

print(eachplayer)

#Detect and fill the missing data
def fill_miss(eachplayer) :

    for i in range(1, 4) :
        ma = eachplayer[eachplayer['Player_Name'] == i]['Age'].median()
        eachplayer['Age'] = eachplayer['Age'].fillna(ma)
        hw = eachplayer[eachplayer['Player_Name'] == i][['Height', 'Weight']].median()
        eachplayer[['Height', 'Weight']] = eachplayer[['Height', 'Weight']].fillna(hw)
        return eachplayer

#Diving the player with their medals and Gender
for eachplayer in cd :

    #Creating the category from the catalogue 
    #Making the Category for the height
    eachplayer.loc[(eachplayer['Height'] <= 127), 'Height']  =  0
    eachplayer.loc[(eachplayer['Height'] >127) & (eachplayer['Height'] <= 143), 'Height'] = 1
    eachplayer.loc[(eachplayer['Height'] >143) & (eachplayer['Height'] <= 159), 'Height'] = 2
    eachplayer.loc[(eachplayer['Height'] >159) & (eachplayer['Height'] <= 175), 'Height'] = 3
    eachplayer.loc[(eachplayer['Height'] >175) & (eachplayer['Height'] <= 191), 'Height'] = 4
    eachplayer.loc[(eachplayer['Height'] >191) & (eachplayer['Height'] <= 207), 'Height'] = 5
    eachplayer.loc[(eachplayer['Height'] >207) & (eachplayer['Height'] <= 226), 'Height'] = 6
    eachplayer.loc[(eachplayer['Height'] >226), 'Height'];

    #Making the category for the weight
    eachplayer.loc[(eachplayer['Weight'] <= 25), 'Weight']  =  0
    eachplayer.loc[(eachplayer['Weight'] >25) & (eachplayer['Weight'] <= 47), 'Weight'] = 1
    eachplayer.loc[(eachplayer['Weight'] >47) & (eachplayer['Weight'] <= 69), 'Weight'] = 2
    eachplayer.loc[(eachplayer['Weight'] >69) & (eachplayer['Weight'] <= 91), 'Weight'] = 3
    eachplayer.loc[(eachplayer['Weight'] >91) & (eachplayer['Weight'] <= 113), 'Weight'] = 4
    eachplayer.loc[(eachplayer['Weight'] >113) & (eachplayer['Weight'] <= 135), 'Weight'] = 5
    eachplayer.loc[(eachplayer['Weight'] >135) & (eachplayer['Weight'] <= 157), 'Weight'] = 6
    eachplayer.loc[(eachplayer['Weight'] >157) & (eachplayer['Weight'] <= 179), 'Weight'] = 7
    eachplayer.loc[(eachplayer['Weight'] >179) & (eachplayer['Weight'] <= 201), 'Weight'] = 8
    eachplayer.loc[(eachplayer['Weight'] >201) & (eachplayer['Weight'] <= 221), 'Weight'] = 9
    eachplayer.loc[(eachplayer['Weight'] >221), 'Weight'];

    #Make the category fot the age
    eachplayer.loc[(eachplayer['Age'] <= 10), 'Age']  =  0
    eachplayer.loc[(eachplayer['Age'] >10) & (eachplayer['Age'] <= 15), 'Age'] = 1
    eachplayer.loc[(eachplayer['Age'] >15) & (eachplayer['Age'] <= 20), 'Age'] = 2
    eachplayer.loc[(eachplayer['Age'] >20) & (eachplayer['Age'] <= 25), 'Age'] = 3
    eachplayer.loc[(eachplayer['Age'] >25) & (eachplayer['Age'] <= 30), 'Age'] = 4
    eachplayer.loc[(eachplayer['Age'] >30) & (eachplayer['Age'] <= 35), 'Age'] = 5
    eachplayer.loc[(eachplayer['Age'] >35) & (eachplayer['Age'] <= 40), 'Age'] = 6
    eachplayer.loc[(eachplayer['Age'] >40) & (eachplayer['Age'] <= 45), 'Age'] = 7
    eachplayer.loc[(eachplayer['Age'] >45) & (eachplayer['Age'] <= 50), 'Age'] = 8
    eachplayer.loc[(eachplayer['Age'] >50) & (eachplayer['Age'] <= 55), 'Age'] = 9
    eachplayer.loc[(eachplayer['Age'] >55) & (eachplayer['Age'] <= 60), 'Age'] = 10
    eachplayer.loc[(eachplayer['Age'] >60) & (eachplayer['Age'] <= 65), 'Age'] = 11
    eachplayer.loc[(eachplayer['Age'] >65) & (eachplayer['Age'] <= 70), 'Age'] = 12
    eachplayer.loc[(eachplayer['Age'] >70) & (eachplayer['Age'] <= 75), 'Age'] = 13
    eachplayer.loc[(eachplayer['Age'] >75) & (eachplayer['Age'] <= 80), 'Age'] = 14
    eachplayer.loc[(eachplayer['Age'] >80) & (eachplayer['Age'] <= 85), 'Age'] = 15
    eachplayer.loc[(eachplayer['Age'] >85) & (eachplayer['Age'] <= 90), 'Age'] = 16
    eachplayer.loc[(eachplayer['Age'] >90) & (eachplayer['Age'] <= 97), 'Age'] = 17
    eachplayer.loc[(eachplayer['Age'] >97), 'Age'];


    #Changing the datatype of Value given type of gender
    eachplayer['Sex'] = eachplayer['sex'].map({'M' : 0, "F" : 1}).astype(float)

    #Changing the datatype
    eachplayer['Medal'] = eachplayer['Medal'].map({"Bronze" : 1, "Silver" : 2, "Gold" : 3}).astype(float)

# Create function to replace missing data with the median value
tr = fill_miss(train)
tr['Medal'] = tr['Medal'].fillna('')
tr['Sex'] = tr['Sex'].apply(lambda x : x == Gender if type(x) == float else 1)
tr['Name'] = tr['Name'].apply(lambda x : x == Name if type(x) == float else 1)

show_data = tr[['Height', 'Weight', 'Age', 'Sex', 'Medal']]
print(show_data.describe())
print(show_data.head().reset_index())
#apply the missing age method to test dataset
tst = fill_miss(test)
tst['Medal'] = tst['Medal'].fillna('')

#Giving training set and test set
X = tr[['Age']].reset_index()
y = tr[['Height', 'Weight']].reset_index()

print(X.head())
print(y.head())
from sklearn.model_selection import cross_val_score as cs

#Split the your data as trainning and test sets
train_X, test_X, train_y, test_y = tts(X, y, train_size = 0.33, test_size = 0.33, random_state = 42)

#Classifying the  splited data
model = dtr()
model.fit(train_X, train_y)

print('Final Score : ', model.score(test_X, test_y))
print('With model and Original inputs : ', cs(model, X, y))

#Predict your data
prediction = model.predict(test_X)

ans = aus(test_y, prediction)
print(ans)

