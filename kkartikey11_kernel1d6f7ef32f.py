import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
titanic = pd.read_csv("../input/training_titanic_x_y_train.csv")
titanic.head()
# delete Name, sibsp, Parch, Ticket, Cabin, Fare as no need
del titanic['Name']
del titanic['SibSp']
del titanic['Parch']
del titanic['Ticket']
del titanic['Fare']
del titanic['Cabin']
titanic.head()
titanic.isnull().sum()
survivedQ = titanic[titanic.Embarked == 'Q'][titanic.Survived == 1].shape[0]
survivedC = titanic[titanic.Embarked == 'C'][titanic.Survived == 1].shape[0]
survivedS = titanic[titanic.Embarked == 'S'][titanic.Survived == 1].shape[0]
print("Q = ",survivedQ)
print("C = ",survivedC)
print("S = ",survivedS)
def getNumber(str):
    if str=="male":
        return 1
    else:
        return 2
titanic["gender"]=titanic["Sex"].apply(getNumber)
del titanic['Sex']
titanic.head()
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

survivedMale= titanic[titanic.gender==1][titanic.Survived==1].shape[0]
NotsurvivedMale= titanic[titanic.gender==1][titanic.Survived==0].shape[0]

survivedWomen= titanic[titanic.gender==2][titanic.Survived==1].shape[0]
NotsurvivedWomen= titanic[titanic.gender==2][titanic.Survived==0].shape[0]

print('No. of Male survived :',survivedMale)
print('No. of Male not survived :',NotsurvivedMale)
print('No. of Woman survived :',survivedWomen)
print('No. of Woman not survived :',NotsurvivedWomen)


l = [survivedMale,NotsurvivedMale,survivedWomen,NotsurvivedWomen,]
plt.pie(l, labels=["Survived male","Not Survived male","Survived female", "NOt Survived female"] , colors=['orange','green','blue','yellow'],explode=(0.15,0.15,0.15,0.15) , autopct = "%.2f%%")
plt.axis('equal')
plt.show()
