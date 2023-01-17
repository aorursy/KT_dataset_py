import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

from matplotlib.colors import ListedColormap

print('basic load')

#reading file

def convertTxtToList():

    f=open("/kaggle/input/p2hdata/PH2Dataset/PH2_dataset.txt", "r")

    lines = f.readlines()

    returnList = []

    for line in lines:

        line = line.replace('||', '|')

        line = line.replace('\n', '')

        lista = list(str(line))

        if len(lista) > 0:

            lista[0] = ''

            lista[len(lista)-1] = ''

            newLine = ''.join(lista)

            arr = newLine.split('|')

            arr = [x.strip() for x in arr]

            if len(arr) > 1:

                returnList.append(arr)

        

    f.close()   

    return returnList
#loading list

clearFile = convertTxtToList()
#convert list to a dataframe

df = pd.DataFrame(clearFile[1:-1], columns=clearFile[0])

df.head(5)
#number of lines and columns

df.shape
#creating column ['number colors']

def createNumberColors(row):

    arr = row['Colors'].strip().split(' ')

    arr = [x for x in arr if x != '']

    return len(arr)



df['number colors'] = df.apply(createNumberColors, axis=1)
#removing ['Colors']

df = df.drop('Colors', 1)

df.head(5)

#normatize ['Pigment Network']

df['Pigment Network'] = df['Pigment Network'].map({'T':1, 'AT':0})

df['Pigment Network'].value_counts()
#normatize ['Dots/Globules']

df['Dots/Globules'] = df['Dots/Globules'].map({'T':1, 'AT':0, 'A':2})

df['Dots/Globules'].value_counts()
#normatize ['Streaks']

df['Streaks'] = df['Streaks'].map({'A':1, 'P':0})

df['Streaks'].value_counts()
#normatize ['Regression Areas']

df['Regression Areas'] = df['Regression Areas'].map({'A':1, 'P':0})

df['Regression Areas'].value_counts()
#normatize ['Blue-Whitish Veil']

df['Blue-Whitish Veil'] = df['Blue-Whitish Veil'].map({'A':1, 'P':0})

df['Blue-Whitish Veil'].value_counts()
df.head(10)
#graphic to check clinical diagnosis with number colors

df[['Clinical Diagnosis', 'number colors']].plot()
#selecting data

x = df.iloc[:,3:10].values

y = df.iloc[:,2].values
#creating test and train matriz

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#using scaling for the matrix

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#training the algorithm

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(x_train, y_train)
#using KNN to predict

y_pred = classifier.predict(x_test)
#building confuse matrix

confusion_matrix(y_test, y_pred)
#hit %

def hitScore(lis1, lis2):    

    score = 0

    for i in range(len(lis1)):

        if lis1[i] == lis2[i]:

            score += 1

            

    return round((score/len(lis2)) * 100, 2)

print('Hit: '+str(hitScore(y_pred, y_test))+'%')