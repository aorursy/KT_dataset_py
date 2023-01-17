import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import libraries

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns



#import files

#pokemon2 = pd.read_csv("../input/pokemon-index-edited/pokemon2.csv")

pokemon = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

combat = pd.read_csv("../input/pokemon-challenge/combats.csv")

#tests = pd.read_csv("../input/pokemon-challenge/tests.csv")

# rename the column in pokemon data with "#" as "number" as its name

pokemon = pokemon.rename(index=str, columns={"#": "Number"})

# Find total win number

total_Wins = len(combat.Winner.value_counts())

# get the number of wins for each pokemon

numberOfWins = combat.groupby('Winner').count()

countByFirst = combat.groupby('Second_pokemon').count()

countBySecond = combat.groupby('First_pokemon').count()

# Finding the total fights of each pokemon

numberOfWins['Total Fights'] = countByFirst.Winner + countBySecond.Winner

# Finding the win percentage of each pokemon

numberOfWins['Win Percentage']= numberOfWins.First_pokemon/numberOfWins['Total Fights']

print(numberOfWins)

# Merge the the original pokemon dataset with the winning dataset

results2 = pd.merge(pokemon, numberOfWins, right_index = True, left_on='Number')

results3 = pd.merge(pokemon, numberOfWins, left_on='Number', right_index = True, how='left')





#plot graph of Speed vs Win Percentage

import matplotlib.pyplot as plt

sns.regplot(x="Speed", y="Win Percentage", data=results3, logistic=True).set_title("Speed vs Win Percentage")
#import libraries to facilitate code testing

import matplotlib.pyplot as plt

from matplotlib import ticker

%matplotlib inline

import pandas as pd

import numpy as np

#Find the pokemons without Win Percentage data 

results3[results3['Win Percentage'].isnull()]
#Implement Linear Regression

from sklearn.linear_model import LinearRegression

results4=results3.dropna()



x = results4["Speed"].values.reshape(-1,1)

y = results4['Win Percentage'].values.reshape(-1,1)



plt.scatter(x,y, s= 10)

plt.title("Speed v.s Win Percentage ")

plt.xlabel("Speed")

plt.ylabel("Win Percentage")



model=LinearRegression()

model.fit(x,y)

y_pred = model.predict(x)

plt.scatter(x,y, s= 10)

plt.plot(x,y_pred,color="purple")
# Predict the Win Percentage of Blastoise (Speed=78)

y = 78 * model.coef_ + model.intercept_

y
#let's check the win percentage with other pokemons with the same Speed



R1=results3[results3['Speed']==78]



R1
# Splitting the dataset into the Training set and Test set (80:20 split)

X = results4.iloc[:, 5:11].values

Y = results4.iloc[:, 15].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    

# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'linear')

regressor.fit(X_train, Y_train)

print(regressor.score(X_train, Y_train))

#Predict Output and compare Actual Win Percentages with predicted Win Percentages

Y_pred= regressor.predict(X_test)

df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})

df
#Draw bar graph of Actual vs Predicted results

df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#get the Mean Absolute Error of the predictions

from sklearn.metrics import mean_absolute_error

from math import sqrt

mae = mean_absolute_error(Y_test, Y_pred)

print("Mean Absolute Error: " + str(mae))