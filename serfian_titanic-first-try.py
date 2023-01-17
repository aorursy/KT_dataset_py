import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tflearn



# Any results you write to the current directory are saved as output.



trainingData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")



def handle_fields(hdata):

    hdata.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)

    hdata['Embarked_S'] = (hdata['Embarked'] == 'S').astype(float)

    hdata['Embarked_C'] = (hdata['Embarked'] == 'S').astype(float)

    hdata['Embarked_Q'] = (hdata['Embarked'] == 'S').astype(float)

    hdata['Sex_male'] = (hdata['Sex'] == 'male').astype(float)

    hdata['Sex_female'] = (hdata['Sex'] == 'female').astype(float)

    hdata.drop(['Embarked', 'Sex'], 1, inplace=True)

    hdata.fillna(0, inplace=True)

    return hdata



trainingData = handle_fields(trainingData)



labels = np.array([trainingData['Survived']], dtype=np.float32)

labels = labels.reshape((891,1))



trainingData.drop(['Survived'], 1, inplace=True)

data = np.array(trainingData, dtype=np.float32)





# Build neural network

net = tflearn.input_data(shape=[None, 10])

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 1, activation='softmax')

net = tflearn.regression(net)



# Define model

model = tflearn.DNN(net)



# Start training (apply gradient descent algorithm)

model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=False)



testIds = testData['PassengerId']

testData = handle_fields(testData)



data = np.array(testData, dtype=np.float32)

labels = model.predict(data)



print(labels)



#testPredictions = pd.concat([testIds, pd.DataFrame({'Survived':labels})], axis=1)

#testPredictions.head()
