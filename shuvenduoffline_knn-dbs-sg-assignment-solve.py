from sklearn import datasets

myiris = datasets.load_iris()

x = myiris.data

y = myiris.target

type(x)

x.shape

type(y)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

type(x_train)
print(x_train)
import numpy as np

class MyClassifier:

    def __init__(self,n_neighbors = 3):

        self.n_neighbors = n_neighbors

    def fit(self,x,y):

        self.x_train = x

        self.y_train = y

        print("Done tranning...")

    def predict(self,x):

        y_pred = []                 #list to store the predicted values

        for test_item in x:         #loop total test case and predict for each of them

            lis = []                #tempurary list to store the howmuch match and the predction for this match 

            for train_items,train_ans in zip(self.x_train,self.y_train):    #loopthrough the x_train and y_train simultaneously 

                summ = 0                #temp variable to store the sum of sequre of difference

                for d,f in zip(test_item,train_items):  #loop simultaneously in each value in test_item and train_items

                    summ += (d-f)**2                    #calculating (dx - dx) ** 2 ,,,, i measured euclidean distance on my own

                dist = np.sqrt(summ)                    #squre root of the summ 

                te = [dist,train_ans]                   # te is temporay list wich store [how_much_match,predict_value_for_this_match]

                lis.append(te)                          #add this to the list of all matches

            lis.sort()                  #this will short the match list accordingly how much is most match first to low match last

            lis = lis[:self.n_neighbors]  #this will cut the list and take the last n most match item

            tem = []                 #temporary list to store the predict values of the sortlisted matches

            for k in lis:            #loop through all element of sortlisted list of match

                tem.append(k[1])    #get the predict value and append to the tem array

            y_pred.append(max(set(tem), key=tem.count))  #Get the most occurard prediction from the list of predict, 

                                                          #i.e from 'tem' in case of tie its know how to dealwith it

        return np.array(y_pred)          #create a numpy array from the predicted values list and return the numpy arrray

    

    def myscore(self,x,y):      #function to calculate the acuracy

        match = 0               #variable to stroe number of match

        total_element_checked = 0   #variable to count number of check we did

        for i , j in zip(x,y):     #loop simultaneously on the elements of y_test and y_pred

            total_element_checked += 1  #increment total number of items we checked

            if i == j:                  # if it was a correct predction then 

                match += 1              #increment the match by 1

        return (match /  total_element_checked) * 100    # calculate the percentage of match and return

                

            

        
myknn = MyClassifier()

print(myknn)
myknn.fit(x_train,y_train)
y_pred = myknn.predict(x_test)

print("Actual Ans :",y_test)

print("MyModel Ans:",y_pred)
#Accuracy

print('Accuracy = ', myknn.myscore(y_test, y_pred))



#Confusion Matrix

from sklearn.metrics import confusion_matrix

print('\nConfusion matrix')

print(confusion_matrix(y_test, y_pred))



#Classification Report

from sklearn.metrics import classification_report

print('\nClassification Report')

print(classification_report(y_test, y_pred))  