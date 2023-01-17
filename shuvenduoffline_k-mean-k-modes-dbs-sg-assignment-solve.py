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

from scipy.spatial import distance

from scipy import stats

class MyClassifier:

    def __init__(self,k = 2):

        self.k = k

    def fit(self,x):

        if x.shape[0] < self.k :

            print("K value less than traning size. Try with large dataset")

            return

        clasters = []

        #initially select k number of values from starting as claster points

        for i in range(self.k):

            clasters.append(x[i])

        clasters = np.array(clasters)

        itr = 1

        while True:

            #creating the D matrix

            d_mat = np.zeros(shape=(self.k,x.shape[0]))

            for i in range(self.k):

                for j in range(x.shape[0]) :

                    d_mat[i][j] = distance.euclidean(clasters[i], x[j])

            #creating the G matrix

            g_mat = np.zeros(shape=(self.k,x.shape[0]))

            for i in range(x.shape[0]) :

                min_dis = 99999

                min_dis_pos = 0

                for j in range(self.k):

                    if min_dis > d_mat[j][i] :

                        min_dis = d_mat[j][i]

                        min_dis_pos = j

                g_mat[min_dis_pos][i] = 1

            new_clasters = []

            for i in range(self.k):

                arr = []

                for j in range(x.shape[0]) :

                    if g_mat[i][j] == 1 :

                        arr.append(x[j])

                arr = np.array(arr)

                #getting the mean to update the clusters, U hv to altr it to get mean() or modes() classifier

                new_clasters.append(arr.mean(axis=0))      

                #new_clasters.append(stats.mode(arr)[0])     #Un-comment this and comment the above to convert it to k-mode 

            new_clasters = np.array(new_clasters)

            if np.array_equal(clasters,new_clasters) :

                self.clasters = clasters

                print(clasters)

                print("Done Traning for the model.")

                break

            else:

                clasters = new_clasters

                print("Interation ",itr," is done ...")

                itr += 1



    def predict(self,x):

        y_pred = []                 #list to store the predicted values

        for test_item in x:         #loop total test case and predict for each of them

            dist = 9999

            pos = -1

            for i in range(self.k):

                dt = distance.euclidean(test_item,self.clasters[i])

                if dt < dist :

                    dist = dt

                    pos = i

            y_pred.append(pos)  #Get the most occurard prediction from the list of predict, 

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

                

            

        
mykmean = MyClassifier(k=3)

print(mykmean)
mykmean.fit(x_train)
y_pred = mykmean.predict(x_test)

print("Actual Ans :",y_test)

print("MyModel Ans:",y_pred)
#Accuracy

print('Accuracy = ', mykmean.myscore(y_test, y_pred))



#Confusion Matrix

from sklearn.metrics import confusion_matrix

print('\nConfusion matrix')

print(confusion_matrix(y_test, y_pred))



#Classification Report

from sklearn.metrics import classification_report

print('\nClassification Report')

print(classification_report(y_test, y_pred))  