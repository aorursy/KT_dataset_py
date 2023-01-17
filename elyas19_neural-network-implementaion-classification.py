import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
data = pd.read_csv("../input/student-mat.csv",sep=";")

data.head()

data.info()
from sklearn.preprocessing import LabelEncoder

lab_enc = LabelEncoder()



for i in range(data.shape[1]):

    #print(data.dtypes[i])

    if data.dtypes[i] == 'object':

        #print(data.dtypes[i])

        #print(data.columns[i])

        data[data.columns[i]] = lab_enc.fit_transform(data[data.columns[i]])
import seaborn as sns

corr_matrix = data.corr()



datac = data.copy()

low_corr = []

for ind,c_val in enumerate(corr_matrix["G3"]):

    if abs(c_val) < 0.10:

        #print(ind)

        #print(datac.columns[ind])

        low_corr.append(datac.columns[ind])



for name in low_corr:

    #print(name)

    datac.pop(name)
datac.head()
corr_matrix = datac.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

corr_matrix["G3"].sort_values(ascending=False)
plt.figure(figsize=(10,5))

ax1 = plt.subplot(1, 2, 1)

sns.distplot(data["G3"], label="Whole Group Avg = " + str(np.round(np.mean(data["G3"]),2)))

plt.ylabel("density")

plt.xlabel("Final Grade")

plt.legend()

ax2 = plt.subplot(1, 2, 2)

sns.distplot(data["G3"][data["sex"]==0], label="Female")

sns.distplot(data["G3"][data["sex"]==1], label="Male")

plt.ylabel("density")

plt.xlabel("Final Grade")

plt.legend(('Female Avg = '+str(np.round(np.mean(data["G3"][data["sex"]==0]),2)),

            'Male Avg = '+str(np.round(np.mean(data["G3"][data["sex"]==1]),2))))

plt.show()
plt.figure(figsize=(10,5))

ax1 = plt.subplot(1, 2, 1)

plt.scatter(datac["G1"],datac["G3"],color='r')

plt.ylabel("G3")

plt.xlabel("G1")

plt.title("corr_val = " + str(0.8))

ax2 = plt.subplot(1, 2, 2)

plt.scatter(datac["G2"],datac["G3"],color='k')

plt.ylabel("G3")

plt.xlabel("G2")

plt.title("corr_val = " + str(0.9))

plt.show()
student_behav = np.array([[np.mean(datac["G3"][datac["higher"]==1]),np.mean(datac["G3"][datac["higher"]==0])],

        [np.mean(datac["G3"][datac["romantic"]==1]),np.mean(datac["G3"][datac["romantic"]==0])],

         [np.mean(datac["G3"][datac["paid"]==1]),np.mean(datac["G3"][datac["paid"]==0])]])



plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

sns.heatmap(student_behav,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

plt.title("Average score based classification")

plt.xticks([0.5,1.5],("Yes","No"),fontsize = 16)

plt.yticks([0.5,1.5,2.5],("higher","romantic","paid"),fontsize = 16)

plt.show()
ax = sns.boxplot(x="failures", y="G3", data=datac)

ax = sns.swarmplot(x="failures", y="G3", data=datac, color=".25")

plt.xlabel('Failures', fontsize=18)

plt.ylabel('G3', fontsize=18)

plt.xticks(fontsize = 16)

plt.yticks(fontsize = 16)

plt.show()
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

fe1_1hot = encoder.fit_transform(datac['Medu'].values.reshape(-1,1))

fe2_1hot = encoder.fit_transform(datac['Fedu'].values.reshape(-1,1))

fe3_1hot = encoder.fit_transform(datac['Mjob'].values.reshape(-1,1))

fe4_1hot = encoder.fit_transform(datac['traveltime'].values.reshape(-1,1))

fe5_1hot = encoder.fit_transform(datac['goout'].values.reshape(-1,1))





datadrop = datac.drop(columns=["Medu","Fedu","Mjob","traveltime","goout"])

X_new = datadrop.iloc[:,:-1].values

Y_new = datadrop.iloc[:,-1].values

X_new = np.concatenate((X_new,fe1_1hot.toarray(),fe2_1hot.toarray(),fe3_1hot.toarray(),fe4_1hot.toarray(),fe5_1hot.toarray()),axis=1)
for i in range(len(Y_new)):

    if Y_new[i]>=12:

        Y_new[i] = 1

    else:

        Y_new[i] = 0
from sklearn.preprocessing import StandardScaler

stand_sca = StandardScaler()

X_trans = stand_sca.fit_transform(X_new)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_trans,Y_new, test_size = 0.2, random_state = 33)
def Neural_Network_Simple(LR,beta,max_ite,input_nodes,hidden_nodes,output_nodes,X,Y):



    # weight initialization

    W_1 = np.random.randn(hidden_nodes, input_nodes)

    W_2 = np.random.randn(output_nodes, hidden_nodes)

    B_1 = np.zeros((hidden_nodes, 1))

    B_2 = np.zeros((output_nodes, 1))



    # gradient descent with momentum

    Vdw_1 = np.random.randn(hidden_nodes, input_nodes)

    Vdw_2 = np.random.randn(output_nodes, hidden_nodes)

    Vdb_1 = np.zeros((hidden_nodes, 1))

    Vdb_2 = np.zeros((output_nodes, 1))



    # cost function

    N = np.size(Y,0)

    Cost = np.zeros((max_ite,1))



    for i in range(max_ite):

        A_1 = W_1.dot(X) + np.tile(B_1, (1, N))

        # Relu activation

        Z_1 = A_1

        Z_1[Z_1 < 0] = 0



        A_2 = W_2.dot(Z_1) + np.tile(B_2, (1, N))

        Z_2 = 1 / (1 + np.exp(-A_2))

        

        del_2 = Z_2 - Y

        #derivative of Relu

        de_2_acti = A_1

        de_2_acti[de_2_acti>0] = 1

        de_2_acti[de_2_acti<=0] = 0

        del_1 = W_2.T.dot(del_2) * de_2_acti

        #backprop

        dw_2 = del_2.dot(Z_1.T)

        dw_1 = del_1.dot(X.T)

        db_2 = np.sum(del_2, 1)

        db_1 = np.sum(del_1, 1).reshape(hidden_nodes,1)



        Vdw_2 = beta * Vdw_2 + (1 - beta) * dw_2

        Vdw_1 = beta * Vdw_1 + (1 - beta) * dw_1

        Vdb_2 = beta * Vdb_2 + (1 - beta) * db_2

        Vdb_1 = beta * Vdb_1 + (1 - beta) * db_1

        #update weights and bias

        W_2 = W_2 - LR * Vdw_2

        W_1 = W_1 - LR * Vdw_1

        B_2 = B_2 - LR * Vdb_2

        B_1 = B_1 - LR * Vdb_1



        Cost[i] = 0.5 * np.sum(del_2**2)/N

        

        print("iteration #",i , 'and accuracy is ' + str(Cost[i]))



        

    return W_1,W_2,B_1,B_2,Cost
def forwardNN_clf(W_1,W_2,B_1,B_2,X):

    A_1 = W_1.dot(X) + np.tile(B_1, (1, 1))

    Z_1 = A_1

    Z_1[Z_1 < 0] = 0



    A_2 = W_2.dot(Z_1) + np.tile(B_2, (1, 1))

    pred = 1/(1 + np.exp(-A_2))

    return pred
W_1,W_2,B_1,B_2,Cost = Neural_Network_Simple(0.01,0.8,300,34,17,1,x_train.T,y_train)
plt.plot(np.linspace(0,len(Cost)-1,len(Cost)),Cost)

plt.xlabel("Number of itertation"); plt.ylabel("loss")

plt.show()
from sklearn.metrics import classification_report, confusion_matrix



ann_predict_implem = forwardNN_clf(W_1,W_2,B_1,B_2,x_test.T)

ann_predict_implem = (ann_predict_implem > 0.5)

print(classification_report(y_test,ann_predict_implem.T))
from tensorflow import keras

from tensorflow.keras import layers

class_NN = keras.models.Sequential()



class_NN.add(layers.Dense(units=17,activation='relu',input_dim=34))

#class_NN.add(layers.Dense(units=17,activation='relu'))

class_NN.add(layers.Dense(units=1,activation='sigmoid'))

class_NN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

class_NN.fit(x_train,y_train,batch_size=10,epochs=100)
ann_predict = class_NN.predict(x_test)

ann_predict = (ann_predict > 0.5) 
print(classification_report(y_test,ann_predict))