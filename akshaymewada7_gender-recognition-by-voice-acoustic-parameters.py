# importing modules



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline
# opening csv file using pandas creating Dataframe

Music = pd.read_csv('../input/voice.csv')



# printing head of Dataframe

print(Music.head())
# Finding Correalation of Features

Music.corr()
# printing shape of Dataframe

print('Shape of Music DataFrame',Music.shape)
# printing keys or features of Music

print('Features: \n', Music.keys())
# Counting Number of Male and Number of Female in Data



print('Number of Males:',Music[Music['label']=='male'].shape[0])

print('Number of Femles:',Music[Music['label']=='female'].shape[0])
# Function to Plotting 10 features out of 20 of Music 



def Plotting_Features(Fun,f):

    

    i=0 # initial index of features  

    j=0 # initial index of color  

    

    color = ['r','g','b','y','c','darkblue','lightgreen',

             'purple','k','orange','olive'] # colors for plots

    

    # Number of rows

    nrows =5

    

    # Creating Figure and Axis to plot 

    fig, axes = plt.subplots(nrows,2)

    

    # Setting Figure size

    fig.set_figheight(20)

    fig.set_figwidth(20)

    

    for row in axes:

        

        plot1 = Fun[f[i]]

        plot2 = Fun[f[i+3]]

        

        col = [color[j],color[j+1]]

        label = [f[i],f[i+1]]

        

        plot(row, plot1,plot2,col,label)

        

        i=i+4

        

        j=j+2

        

    plt.show()



def plot(axrow, plot1, plot2, col, label):

    

    axrow[0].plot(plot1,label=label[0],color=col[0])

    axrow[0].legend()

    

    axrow[1].plot(plot2,label=label[1],color=col[1])

    axrow[1].legend()
# Setting Male Acoustic Parameters

Male = Music[Music['label']=='male']

Male = Male.drop(['label'],axis=1)

features = Male.keys()

Plotting_Features(Male,features)
# Setting female Acoustic Parameters

Female = Music[Music['label']=='female']

Female = Female.drop(['label'],axis=1)

features = Female.keys()

Plotting_Features(Female,features)
# creating Train and Target data from data



# dropping label we get our Train data 

X = Music.drop(['label'],axis=1)



# taking only label as target 

Y = Music['label'] 
# Preprocessing Data before prediction



# Using LabelEncoder to Encode our label {'male':1, 'female':0} 

from sklearn.preprocessing import LabelEncoder



Label_Encoder = LabelEncoder()



# Fitting Y data in Label_Encoder for Encoding 

Label_Encoder.fit(Y)



# Transforming Data to {'male':1, 'female':0}

Y_Encoded = Label_Encoder.transform(Y)

Y_Encoded
# Using StandardScaler to scale Train Data 

from sklearn.preprocessing import StandardScaler



Standard_Scaler = StandardScaler()



# fitting X data in Standard_Scaler

Standard_Scaler.fit(X)



# Transforming to scale data

X_Scaled = Standard_Scaler.transform(X)



# priting scaled data

X_Scaled
# splitting Data into training and testing Data using cross_validation.train_test_split

# Train - Test data ratio of 75%-25%

# Random State to Randomize data = 123





from sklearn.cross_validation import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X_Scaled,Y_Encoded,test_size = 0.25, random_state=123)
# importing Support Vector Machine Algorithm for Prediction

from sklearn import svm



# creating Classifier

clf = svm.SVC(C =200, gamma = 0.1)
# Training our Classifier 

clf.fit(X_train,Y_train)
# predicting our test data

Prediction = clf.predict(X_test)

Prediction
# Calculating Accuracy of prediction of Our model 



from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(Prediction,Y_test)

Accuracy
# importing cross_val_score to calculate score

from sklearn.cross_validation import cross_val_score
# Defining three different kernels

kernels = ['linear','rbf','poly']



score = []



for i in kernels:

    clf = svm.SVC(kernel = i)

    Accuracy = cross_val_score(clf,X_Scaled,Y_Encoded,cv = 15, scoring='accuracy')

    score.append(Accuracy.mean())

for i in range(len(kernels)):

    print(kernels[i],':',score[i])
score = []

for i in range(10):

    clf = svm.SVC(C = i+1)

    Accuracy = cross_val_score(clf,X_Scaled,Y_Encoded,cv = 15, scoring='accuracy')

    score.append(Accuracy.mean())

for i in range(10):

    print('C =',i+1,': Score =',score[i])
score = [] 

gamma_values = [0.0001,0.001,0.01,0.1,1.0,100.0,1000.0]

for i in gamma_values:

    clf = svm.SVC(gamma = i)

    Accuracy = cross_val_score(clf,X_Scaled,Y_Encoded,cv = 15, scoring='accuracy')

    score.append(Accuracy.mean())

for i in range(len(gamma_values)):

    print('gamma:',gamma_values[i],': Score:',score[i])