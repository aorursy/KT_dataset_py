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

import numpy as np
print(x_train)
print(np.column_stack((x_train,x_train[:,0])))


class Node:

    def __init__(self,att,dis,child,lebel,domain):

        self.child = child

        self.q_attribute = att

        self.dis = dis

        self.class_lebel = lebel

        self.domain = domain



class MyClassifier:

    def __init__(self):

        print("Model created")

    def select_attribute(self,x,y):

        info_d = 0

        domain_y = list(set(list(y)))

        count = [0] * len(domain_y)

        for i in y:

            count[domain_y.index(i)] += 1

        sum_count = sum(count)

        for i in count:

            info_d += (-1*(i/sum_count)*np.log2(i/sum_count))

        temp = x[0]

        length = len(temp)

        info_gain = [0] * length

        for sel in range(length):

            colum = x[:,sel]

            domain = list(set(list(colum)))

            domain_col_len = len(domain)

            temp_arr = [[0 for i in range(len(domain_y))] for j in range(domain_col_len)]

            for item,target in zip(colum,y):

                temp_arr[domain.index(item)][domain_y.index(target)] += 1

            info_attr = 0

            sum_all = np.sum(temp_arr)

            for item in temp_arr:

                sum_item = sum(item)

                for i in item:

                    info_attr += (-1*(sum_item/sum_all) * (i/sum_item)*np.log2(i/sum_item))

            info_gain[sel] = info_d - info_attr

        return info_gain.index(max(info_gain))

        

    def createNode(self,x,y):

        domain = list(set(list(y)))

        if len(domain) == 1:

            return Node(None,{},True,domain[0],None)       #creating a leaf node

        selected_attribute = self.select_attribute(x,y)      #select attribute method

        domain_x = list(set(list(x[ : ,selected_attribute])))   #get the domain of selected attribute

        dis = {}                   #dictonary to store childs, tree is not binary

        x = np.column_stack((x,y))

        for i in domain_x:

            x_copy = x[x[:,selected_attribute] == i]

            y_copy =  x_copy[:,-1]

            x_copy = np.delete(x_copy,selected_attribute,axis=1)

            x_copy = np.delete(x_copy,-1,axis=1)

            dis[i] = self.createNode(x_copy,y_copy)

        node = Node(selected_attribute,dis,False,None,domain_x)

        return node

            

    def fit(self,x,y):

        self.x_train = x

        self.y_train = y

        self.head = self.createNode(x,y)

        print("Done tranning...")

        

    def predict(self,x):

        y_pred = []                 #list to store the predicted values

        for test_item in x:         #loop total test case and predict for each of them

            node = self.head

            while node.child != True :

                question = test_item[node.q_attribute]

                if question in node.domain :

                    node = node.dis[question]

                else:

                    dif = 999

                    loc = node.domain[0]

                    for i in node.domain :

                        if abs(i-question) < dif :

                            dif = abs(i-question)

                            loc = i

                    node = node.dis[loc]

            ans = node.class_lebel

            y_pred.append(int(ans))  #Get the most occurard prediction from the list of predict, 

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

                

            

        
mydtree = MyClassifier()

print(mydtree)

mydtree.fit(x_train,y_train)
y_pred = mydtree.predict(x_test)

print("Actual Ans :",y_test)

print("MyModel Ans:",y_pred)
#Accuracy

print('Accuracy = ', mydtree.myscore(y_test, y_pred))



#Confusion Matrix

from sklearn.metrics import confusion_matrix

print('\nConfusion matrix')

print(confusion_matrix(y_test, y_pred))



#Classification Report

from sklearn.metrics import classification_report

print('\nClassification Report')

print(classification_report(y_test, y_pred))  