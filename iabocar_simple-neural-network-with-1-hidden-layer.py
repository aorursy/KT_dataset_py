import pandas as pd

import numpy as np

import csv as csv

import re

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

    

def getNumber(x):

    search = re.search(r"\d+", x)

    if search:

        return search.group(0)

    return 0

def getLetter(x):

    search = re.search(r"\D+", x)

    if search:

        return ord(search.group(0)[0])

    return 0



def getLastName(x):

    search = re.search(r"\w+", x)

    if search:

        return search.group(0)

    return ""



def getTitle(x):

    search = re.search(r" (\S+)\.", x)

    if search:

        return search.group(1)

    return ""    





train_input = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})

test_input = pd.read_csv("../input/test.csv", dtype={"Age": np.float64})



TRAINSIZE = train_input.shape[0]



df = pd.concat([train_input, test_input], ignore_index=True)



TitleMap = {}





df['NameTitle'] = df['Name'].apply(getTitle)

df.loc[df[df['NameTitle'] == 'Ms'].index, 'NameTitle'] = 'Miss'



for index, item in enumerate(df['NameTitle'].unique()):

    TitleMap[item] = index + 1



df['Gender']  = df.Sex.map({'female':0, 'male':1}).astype(int)

df["Family"] = df.Name.map(getLastName)  

df['NameLength'] = df["Name"].apply(lambda x: len(x))



df['NameTitleCat'] = df.NameTitle.map(TitleMap).astype(int)



df['CabinInt'] = df.Cabin.dropna().map(getNumber).astype(int)

df['CabinLetter'] = df.Cabin.dropna().map(getLetter).astype(int)    



uniqueFamily = df["Family"].unique()



df["FamilyMemberOnBoard"]  = 1

for name in uniqueFamily:

    number = df[df['Family'] == name].groupby('Family').PassengerId.nunique()[0]

    df.loc[df['Family'] == name,"FamilyMemberOnBoard"] = number;



df['AgeIsNull'] = pd.isnull(df.Age).astype(int)



features_for_age_prediction = ['Pclass', 'SibSp','Parch','Gender','FamilyMemberOnBoard','NameLength','NameTitleCat']    

age_prediction_linear_regressor = LinearRegression()

age_X_train = df[features_for_age_prediction][df['Age'].notnull()]

age_Y_train = df['Age'][df['Age'].notnull()]

age_prediction_linear_regressor.fit(age_X_train, np.ravel(age_Y_train))



df['AgeFill'] = df['Age']

df.loc[df[df['Age'].isnull()].index, 'AgeFill'] = age_prediction_linear_regressor.predict(df[features_for_age_prediction][df['Age'].isnull()])



uniqueChildFamily = df[df["AgeFill"] <= 15]["Family"].unique();

df["HasChild"] = 0



df.loc[(df["AgeFill"] > 15 )& (df["Family"].isin(uniqueChildFamily)),"HasChild"] = 1



df['FamilySize'] = df.SibSp + df.Parch



features_for_fare_prediction = ['Pclass', 'SibSp','Parch','Gender','FamilyMemberOnBoard','NameLength','NameTitleCat']    

fare_prediction_linear_regressor = LinearRegression()

fare_X_train = df[features_for_age_prediction][df['Fare'].notnull()]

fare_Y_train = df['Fare'][df['Fare'].notnull()]

fare_prediction_linear_regressor.fit(fare_X_train, np.ravel(fare_Y_train))



df.loc[df[df['Fare'].isnull()].index, 'Fare'] = fare_prediction_linear_regressor.predict(df[features_for_fare_prediction][df['Fare'].isnull()])





if len(df.Fare[ df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]



if len(df.Cabin[ df.Cabin.isnull() ]) > 0:

    median_cabinLetter = np.ones(df.FamilySize.max()+1)

    for f in range(0,df.FamilySize.max()+1):       

        if len(df.Cabin[ (df.FamilySize == f)].dropna()) > 0:

            median_cabinLetter[f] = df[ df.FamilySize == f ]['CabinLetter'].dropna().median()

        else:

            median_cabinLetter[f] = df['CabinLetter'].dropna().median()

    for f in range(0,df.FamilySize.max()+1):                                              # loop 0 to 2

        df.loc[ (df.CabinLetter.isnull()) & (df.FamilySize == f), 'CabinLetter'] = median_cabinLetter[f]



    df.loc[df['CabinInt'].isnull(),'CabinInt'] = df['CabinInt'].dropna().median()



df["Embarked"] = df["Embarked"].fillna("S")

df["EmbarkedInt"] =df["Embarked"].map({"S":0, "C":1, "Q":2}).astype(int)





# randomize our data to have train and test set different every time

#df.reindex(np.random.permutation(df.index))

df_number = df.drop(['PassengerId', 'Family','Name', 'Age','Sex', 'Ticket', 'Cabin', 'Embarked','NameTitle'], axis=1)



testids = df['PassengerId'].values[TRAINSIZE::];



df_number = df_number/(df_number.max() - df_number.min())



key_features = ['Pclass','FamilySize','Fare','Gender','AgeFill','HasChild','FamilyMemberOnBoard','NameLength', 'NameTitleCat']



test_data = df_number[key_features].values[TRAINSIZE::]



X_data = df_number[key_features].values[0:TRAINSIZE]

y_data= df_number[['Survived']].values[0:TRAINSIZE]
X_train = X_data[0:700]

y_train = y_data[0:700]



X_test = X_data[700::]

y_test = y_data[700::]
learning_rate = 0.3

trainning_epochs= 10000

display_step = 500



threshold = 0.75



TRAINSIZE = tf.constant( np.float32(X_train.shape[0]))

LAMBDA = tf.constant(0.0001)



n_hidden_1 = 5

n_input = X_data.shape[1]

n_output = 1

n_sampels = X_data.shape[0]



X = tf.placeholder("float",[None,n_input])

y = tf.placeholder("float",[None,n_output])



weights_1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))

weights_2 = tf.Variable(tf.random_normal([n_hidden_1, n_output]))

bias_1 = tf.Variable(tf.random_normal([n_hidden_1]))

bias_2 = tf.Variable(tf.random_normal([n_output]))



def forwardprop(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_1),bias_1))

    #tf.summary.histogram('weights_1', weights_1)

    #tf.summary.histogram('bias_1', weights_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights_2),bias_2)

    layer_2_sigmoid = tf.nn.sigmoid(layer_2)

    #tf.summary.histogram('weights_2', weights_2)

    #tf.summary.histogram('bias_2', bias_2)

    return layer_2_sigmoid,  layer_2, layer_1



y_hat, y_hat_witout_sigmoid,_ = forwardprop(X)



is_greater = tf.greater(y_hat, threshold)

prediction = tf.to_int32(is_greater)

correct_prediction = tf.equal(prediction, tf.to_int32(y_hat))

accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))



cost_J= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat_witout_sigmoid, y))

#tf.summary.scalar('cost_J', cost_J)

cost_reg = tf.mul(LAMBDA , tf.add(tf.reduce_sum(tf.pow(weights_1, 2)),tf.reduce_sum(tf.pow(weights_2, 2))))

#tf.summary.scalar('cost_reg', cost_reg)



cost = cost_J + cost_reg

#tf.summary.scalar('cost', cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
75#summary = tf.summary.merge_all()



init = tf.global_variables_initializer()

J = []

testJ = []

config = tf.ConfigProto()

config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:

    #summary_writer = tf.summary.FileWriter("log4/", sess.graph)

    sess.run(init)

    

    for epoch in range(trainning_epochs):

        

        #_,c,summary_str = sess.run([optimizer, cost,summary], feed_dict={X: batch_xs})

        #_,c,summary_str = sess.run([optimizer, cost,summary], feed_dict={X: X_data, y: y_data}) 

        _,c = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train}) 

        c_test = sess.run([cost], feed_dict={X: X_test, y: y_test})      

        #summary_writer.add_summary(summary_str,epoch)    

        J.append(c)

        testJ.append(c_test)

        if epoch % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1),

                  "cost=", "{:.9f}".format(c))

            

    print("Optimization Finished!")

    weights_1_data = sess.run([weights_1], feed_dict={X: X_test, y: y_test})[0]  

    weights_2_data = sess.run([weights_2], feed_dict={X: X_test, y: y_test})[0]  

    y_predict_test, accuracy = sess.run([prediction,accuracy_op], feed_dict={X: X_test})  

    test_result  = sess.run([prediction], feed_dict={X: test_data})[0]  
%matplotlib inline

plt.plot(J,label="J")

plt.plot(testJ, label="testJ")

plt.grid(1)

plt.legend(loc='upper center', shadow=True)

plt.xlabel('Iterations')

plt.ylabel('Cost')
from sklearn import metrics



print ("validation accuracy:", accuracy)

print ("Precision", metrics.precision_score(y_test, y_predict_test))

print ("Recall", metrics.recall_score(y_test, y_predict_test))

print ("f1_score", metrics.f1_score(y_test, y_predict_test))

print ("confusion_matrix")

print (metrics.confusion_matrix(y_test, y_predict_test))
%matplotlib inline

neuron_weight = weights_1_data



plt.set_cmap("plasma")

plt.axis('off')

plt.imshow(neuron_weight)
%matplotlib inline

neuron_weight = weights_2_data



plt.set_cmap("plasma")

plt.axis('off')

plt.imshow(neuron_weight)
submission = pd.DataFrame({

        "PassengerId": testids,

        "Survived": test_result.T[0]

    })



submission.to_csv("titanic_NN_tensorflow.csv", index=False)