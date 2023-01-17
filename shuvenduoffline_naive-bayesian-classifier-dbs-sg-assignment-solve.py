from sklearn import datasets

myiris = datasets.load_iris()

x = myiris .data

y = myiris .target

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
print(y_train)
import numpy as np

class MyClassifier:

    def __init__(self):

        print("Successfully Created Classifier")

    def fit(self,x,y):

        self.x_train = x

        self.y_train = y

        self.output_domain = list(set(list(y)))      #getting all possiable class lebels

        self.output_domain_len = len(self.output_domain)  #number of possiable class lebel

        print("Out Put Domain :",self.output_domain)

        print("Out Put Domain Length :",self.output_domain_len)

        print("Done tranning...")

    def predict(self,x):

        y_pred = []                 #list to store the predicted values

        length_of_item = len(x[0]) #getting the length of singel item

        length_of_item_train = len(self.x_train[0]) #train data set singel item length

        if length_of_item != length_of_item_train:

            print("Train and Test dataset size don't match! Error !")

            return np.array([0])

        for test_item in x:         #loop total test case and predict for each of them

            arr_count = [0] * self.output_domain_len  # count number of occurance of certain class

            #creating the arry wich will hold count of entry E(age == youth | buy_computer == yes)

            arr_ans = [[0 for i in range(length_of_item)] for j in range(self.output_domain_len)]  #arrry of array to store result

            for train_items,train_ans in zip(self.x_train,self.y_train):    #loopthrough the x_train and y_train simultaneously 

                arr_count[train_ans] += 1                                   #cont  the total type of present in certain class

                for i in range(length_of_item):                             # loop through all the attribute

                    if train_items[i] == test_item[i] :                     # if the attribute match the train data then

                        arr_ans[train_ans][i] += 1                          #count the occurance

            #Do laplecian correction if needed

            for i in range(self.output_domain_len):

                for j in range(length_of_item):

                    if arr_ans[i][j] == 0 :                              #If any value is zero then do laplus correction

                        for j in range(length_of_item):

                            arr_ans[i][j] += 1

                        arr_count[i] += length_of_item

                        break

            total_entry = sum(arr_count)                                # Total number of touple after laplatian correction

            each_class_probablity = [1] * self.output_domain_len        # This is like P(age = youth | buy_computer = yes) ans all others

            for i in range(self.output_domain_len):

                for j in range(length_of_item):

                    each_class_probablity[i] = each_class_probablity[i] * (arr_ans[i][j] / arr_count[i])   #calculate P(X|Ci)

                each_class_probablity[i] = each_class_probablity[i] * (arr_count[i] / total_entry)     #to maximize P(X|Ci) P(Ci)

            ans_class = each_class_probablity.index(max(each_class_probablity))  #let max funtion work in normal and tie, then decide ans

            y_pred.append(ans_class)

        return np.array(y_pred)          #create a numpy array from the predicted values list and return the numpy arrray

    

    def myscore(self,x,y):      #function to calculate the acuracy

        match = 0               #variable to stroe number of match

        total_element_checked = 0   #variable to count number of check we did

        for i , j in zip(x,y):     #loop simultaneously on the elements of y_test and y_pred

            total_element_checked += 1  #increment total number of items we checked

            if i == j:                  # if it was a correct predction then 

                match += 1              #increment the match by 1

        return (match /  total_element_checked) * 100    # calculate the percentage of match and return

                

            

        
myNBC = MyClassifier()

print(myNBC)
myNBC.fit(x_train,y_train)
y_pred = myNBC.predict(x_test)

print("Actual Ans :",y_test)

print("MyModel Ans:",y_pred)
#Accuracy

print('Accuracy = ', myNBC.myscore(y_test, y_pred))



#Confusion Matrix

from sklearn.metrics import confusion_matrix

print('\nConfusion matrix')

print(confusion_matrix(y_test, y_pred))



#Classification Report

from sklearn.metrics import classification_report

print('\nClassification Report')

print(classification_report(y_test, y_pred))  