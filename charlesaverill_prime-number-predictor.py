#Charles Averill, 2019
#Imports

import pandas as pd



import sklearn



from itertools import count, islice

from math import sqrt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
#Training method

def train(features, labels):

    #Separate features and labels into training and testing sets

    ftrain, ftest, ltrain, ltest = train_test_split(features, labels, test_size=0.1)

    #Create Random Forest Regressor with 100 trees

    regressor = RandomForestRegressor(n_estimators=100)

    #Fit model and get accuracy

    model = regressor.fit(ftrain, ltrain)

    accuracy = regressor.score(ftest, ltest) * 100



    return model, accuracy
df = pd.read_csv("../input/first-100000-prime-numbers/output.csv").drop(['Interval'], axis=1)
features = df.drop('Num', axis=1).values

labels = df['Num'].values
model, accuracy = train(features, labels)

print("Accuracy:", accuracy)
labels = ['Rank']



csv = df.values



#num = int(input("What index prime do you want? "))

num = 50
def isPrime(n):

    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))
#Predicting

inp = pd.DataFrame([[num]], columns = labels)



prediction1 = int(round(model.predict(inp)[0] - .5))

prediction2 = prediction1

#Makes predictions more accurate

while(not isPrime(prediction1) and not isPrime(prediction2)):

    prediction1 += 1

    prediction2 -= 1



#Print values

if(isPrime(prediction1)):

    print("Prediction:", prediction1)

else:

    print("Prediction:", prediction2)

#Only prints actual value if the csv has it

if(num < len(csv)):

    print("Actual Value:", csv[num - 1][1])