import pandas as pd

import numpy as np



data = pd.read_csv('../input/data.csv').iloc[:, 1:32]

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

data.head(n=10)
train, validate, test = np.split(data.sample(frac=1, random_state=42),

                                 [int(.6*len(data)), int(.8*len(data))])
from sklearn import linear_model, metrics



list_C = np.arange(100 , 1000, 1)

score_train = np.zeros(len(list_C))

score_val = np.zeros(len(list_C))

precision_val= np.zeros(len(list_C))

score_test = np.zeros(len(list_C))

recall_test = np.zeros(len(list_C))

precision_test= np.zeros(len(list_C))

count = 0

for C in list_C:

    reg = linear_model.LogisticRegression(C=C)

    reg.fit(train.iloc[:,2:32], train['diagnosis'])

    score_train[count]= metrics.accuracy_score(

        train['diagnosis'], reg.predict(train.iloc[:, 2:32]))

    score_val[count] = metrics.accuracy_score(

        validate['diagnosis'], reg.predict(validate.iloc[:, 2:32]))

    precision_val[count] = metrics.precision_score(

        validate['diagnosis'], reg.predict(validate.iloc[:, 2:32]))

    score_test[count] = metrics.accuracy_score(

        test['diagnosis'], reg.predict(test.iloc[:, 2:32]))

    recall_test[count] = metrics.recall_score(

        test['diagnosis'], reg.predict(test.iloc[:, 2:32]))

    precision_test[count] = metrics.precision_score(

        test['diagnosis'], reg.predict(test.iloc[:, 2:32]))

    count = count + 1
matrix = np.matrix(np.c_[list_C, score_train, score_val, precision_val,

                         score_test, recall_test, precision_test])

models = pd.DataFrame(data = matrix, columns = 

             ['C', 'Train Accuracy', 'Validation Accuracy', 'Validation Precision' ,

              'Test Accuracy', 'Test Recall', 'Test Precision'])

models.head(n=10)
best_index = models['Validation Accuracy'].idxmax()

models.iloc[best_index, :]
reg = linear_model.LogisticRegression(C=list_C[best_index])

reg.fit(train.iloc[:,2:32], train['diagnosis'])
print('Train Set')

m_confusion_train = metrics.confusion_matrix(train['diagnosis'],

            reg.predict(train.iloc[:, 2:32]))

pd.DataFrame(data = m_confusion_train, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])
print('Validation Set')

m_confusion_validate = metrics.confusion_matrix(validate['diagnosis'],

                         reg.predict(validate.iloc[:, 2:32]))

pd.DataFrame(data = m_confusion_validate, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])
print('Test Set')

m_confusion_test = metrics.confusion_matrix(test['diagnosis'],

                         reg.predict(test.iloc[:, 2:32]))

pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])