# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go

warnings.filterwarnings("ignore", category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import data



data = pd.read_csv("../input/column_2C_weka.csv")

print(data.info())
#split data to x, y 

x = data.drop(["class"], axis = 1)

y = data["class"].values

#normalized data

x = (x - np.min(x)) / (np.max(x) - np.min(x)).values



x.head()
#%% Show the ratio of normal/abnormal

import seaborn as sns



rate = pd.Series(y).value_counts()

plt.figure(figsize=[5,5])

plt.pie(rate.values, explode = [0, 0], labels = rate.index,  autopct = "%1.1f%%")

plt.show()
plt.figure(figsize=[15,5])



# Create dataframe and reshape

columns = list(x.columns) #column names



df = x.copy()

df["class"] = y #df = x_data + y_data

df = pd.melt(df, value_vars=columns, id_vars='class') #id = class olsun,  diğer columnları variable olarak dağıt





#Plot

plt.figure(figsize=(16,6))

pal = sns.cubehelix_palette(2, rot=.5, dark=.3)

sns.swarmplot(x="variable",y="value", hue="class", palette=pal, data=df)

plt.show()
#change y values abnormal/normal to 0/1

y = np.array( [1 if each == "Abnormal" else 0 for each in y] )
#  SPLIT DATA TO train and test

from sklearn.model_selection import train_test_split



#x = checkup, y = classes

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
#%% PARAMETER INITIALIZE 

def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1), 0.01)

    b = 0.0

    return w,b



def sigmoid(z):

    y_head = 1/(1 + np.exp(-z))

    return y_head
def forward_backward_propagation(w, b, x_train, y_train):

    #foward propagation

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = - y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)

    cost = (np.sum(loss)) / x_train.shape[1] # Bölme sebebi çıkan sonucu normalize etmek



    #backward propagation   

    derivative_weight = (np.dot(x_train, ((y_head-y_train).T))) / x_train.shape[1]

    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]

    

    gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}

    

    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):

    cost_list = [] #Tüm costları depolamak için, analiz için

    cost_list2 = [] #Her 10 adımda bir cost değerlerini depolar

    index = [] # Cost2'nin kaçıncı i değerlerine denk geldiğini gösterir

    

    #updating parameters

    for i in range(number_of_iteration):

        #make forward and backward propagation and find cost and gradients

        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)

        cost_list.append(cost) 

        #Update et

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 100 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i : %f" %(i, cost))

    

    parameters = {"weight" : w, "bias" : b} #Elimdeki son weight ve bias değerleri

    

    #Parametrelerin güncelleme çizimleri

    plt.plot(index, cost_list2)

    plt.xticks(index, rotation='vertical')

    plt.xlabel("Number of iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list 
#%% PREDICT, TEST İÇİN VERİLEN DATA'NIN SONUÇLARINI TAHMİN ET



def predict(w, b, x_test):

    # test için verilen data x_test

    z = sigmoid(np.dot(w.T, x_test) + b)

    Y_prediction = np.zeros((1, x_test.shape[1])) #Tahmin sonuçları için bir array oluştur. Ör : 1,150...

    

    for i in range(z.shape[1]): #her sütün için gezilecek

        if z[0,i] <= 0.5:

            Y_prediction[0, i] = 0

        else:

            Y_prediction[0, i] = 1

    

    return Y_prediction
#%% NE KADAR DOĞRU TAHMİN EDİLDİ

    

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    #initialize

    dimension = x_train.shape[0] #that is 4096

    w,b = initialize_weights_and_bias(dimension)

    

    #W ve b değerlerini güncelle. Train ve Test datalarını tahmin et

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    

    #y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    

    #Ne kadar yanlış var

    print("My Test Accuracy : {} %" .format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return y_prediction_test
#%% CLASSIFICATION WITH MY LOGISTIC REGRESYON



#BENIM REGRESSION TAHMINLERIM

my_predict =logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 5, num_iterations = 1000).reshape(-1,1)





#CONFUSION MATRIX, TAHMINLER NE KADAR DOGRU

from sklearn.metrics import confusion_matrix

my_cm = confusion_matrix(y_test, my_predict)



import seaborn as sns

import matplotlib.pyplot as plt



#MY LR CONFUSION MATRIX PLOT 

plt.figure(figsize=(5,5))

sns.heatmap(my_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title("MY CONFUSION MATRIX PLOT")

plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T, y_train.T)



#SKLEARN REGRESSION PREDICTS

y_sk_predict =  lr.predict(x_test.T)



#ACCURACY

lr_score = lr.score(x_test.T, y_test.T) * 100

print("Test Accuracy According To (Sklearn)Logistic Reg: {}".format(lr_score))



#CONFUSION MATRIX

sk_cm = confusion_matrix(y_test, y_sk_predict)



#SKLEARN lR CONFUSİON MATRİX PLOT

plt.figure(figsize=(5,5))

sns.heatmap(sk_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title("SK CONFUSİON MATRİX PLOT")

plt.show()
#%% CLASSIFICATION WITH KNN



from sklearn.neighbors import KNeighborsClassifier



knn_score = []

#k degelerine gore score'ları bul

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train.T, y_train.T)

    knn_score.append( knn.score(x_test.T, y_test.T) )



df = pd.DataFrame(knn_score)

#K DEGERLERINE GORE DOGRULUK ORANLARINI CIZ

plt.figure(figsize=(7,5))

plt.plot(df.index+1, df.values, color="blue")

plt.title("K Degerlerine Göre Accuracy")

plt.xlabel("K value")

plt.ylabel("Accuracy")

plt.show()



#K = 15 EN IYI DEGER (K= 14 ICIN TAHMINLER YAP)

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(x_train.T, y_train.T)

y_knn_predict = knn.predict(x_test.T)



#ACCURACY YAZ

knn_score = knn.score(x_test.T, y_test.T) * 100

print("Test Accuracy According To KNN(K=15): {}".format(knn_score))



##CONFUSION MATRIX, TAHMINLER NE KADAR DOGRU

knn_cm = confusion_matrix(y_test, y_knn_predict)



#KNN CONFUSİON MATRİX PLOT

plt.figure(figsize=(6,5))

sns.heatmap(knn_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title("K=15 CONFUSİON MATRİX PLOT")

plt.show()
#%% CLASSIFICATION WITH SVM  (SUPPORT VECTOR MACHINE)

from sklearn.svm import SVC



svm = SVC(random_state = 42)

svm.fit(x_train.T, y_train.T)



#ACCURACY YAZ

svm_score = svm.score(x_test.T, y_test.T) * 100

print("Test Accuracy According To SVM : {}".format(svm_score))



#PREDICT WITH SVM

svm_predict = svm.predict(x_test.T)



#CONFUSION MATRIX

svm_cm = confusion_matrix(y_test, svm_predict)



#SVM CONFUSİON MATRIX PLOT

plt.figure(figsize=(5,5))

sns.heatmap(svm_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title("SK CONFUSİON MATRİX PLOT")

plt.show()
# CLASSIFICATION WITH NAIVE BAYES

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



#ACCURACY

nb_score = nb.score(x_test.T, y_test.T) * 100

print("Test Accuracy According To Naive Bayes : {}".format(nb_score))



#PREDICT WITH NAIVE BAYES

nb_predict = nb.predict(x_test.T)



#CONFUSION MATRIX

nb_cm = confusion_matrix(y_test, nb_predict)



#NAIVE BAYES CONFUSİON MATRIX PLOT

plt.figure(figsize=(5,5))

sns.heatmap(nb_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title("NAIVE BAYES CONFUSİON MATRİX PLOT")

plt.show()
#%% CLASSIFICATION WITH DESCION TREE



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train.T, y_train.T)



#ACCURACY

dt_score = dt.score(x_test.T, y_test.T) * 100

print("Test Accuracy According To Decision Tree : {}".format(dt_score))



#PREDICT WITH decision tree

dt_predict = dt.predict(x_test.T)



#CONFUSION MATRIX

dt_cm = confusion_matrix(y_test, nb_predict)



#DESCION TREE CONFUSİON MATRIX PLOT

plt.figure(figsize=(5,5))

sns.heatmap(dt_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title("DECISION TREE CONFUSİON MATRİX")

plt.show()
#%% CLASSIFICATION WITH RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=300, random_state=1)

rf.fit(x_train.T, y_train.T)



#ACCURACY YAZ

rf_score = rf.score(x_test.T, y_test.T) * 100

print("Test Accuracy According To Random Forest Algorithm : {}".format(rf_score))



#PREDICT WITH RANDOM FOREST

rf_predict = rf.predict(x_test.T)



#CONFUSION MATRIX

rf_cm = confusion_matrix(y_test, rf_predict)



#RANDOM FOREST CONFUSİON MATRIX PLOT

plt.figure(figsize=(5,5))

sns.heatmap(rf_cm, annot = True, linewidth = 0.5, linecolor="red", fmt = ".0f")

plt.xlabel("Predict Values")

plt.ylabel("True Values")

plt.title(" RANDOM FOREST ALGORITHM CONFUSİON MATRİX")

plt.show()
trace = go.Bar(

    x=['Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest'],

    y=[lr_score, knn_score, svm_score, nb_score, dt_score, rf_score],

    marker=dict(color=['#008BF8', '#0FFF95', '#EE6C4D', '#A30000', '#2081C3', '#FF7700']),

)



layout = go.Layout(

    title='Accuracy Comparison The All Algorithms',

)



fig = go.Figure(data=[trace], layout=layout)

iplot(fig)