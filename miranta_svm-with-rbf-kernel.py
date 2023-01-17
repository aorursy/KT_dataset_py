import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import classification_report



data = pd.read_csv('../input/bank.csv',sep=',',header='infer')

data = data.drop(['day','poutcome','contact'],axis=1)



def normalize(data):

    # Before we can feed the data to train

    # and test the classifier, we need to normalize

    # the data to acceptable and convenient values

    # for cross validation and prediction purposes later on 

    data.y.replace(('yes', 'no'), (1, 0), inplace=True)

    data.default.replace(('yes','no'),(1,0),inplace=True)

    data.housing.replace(('yes','no'),(1,0),inplace=True)

    data.loan.replace(('yes','no'),(1,0),inplace=True)

    data.marital.replace(('married','single','divorced'),(1,2,3),inplace=True)

    data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)

    data.education.replace(('primary','secondary','tertiary','unknown'),(1,2,3,4),inplace=True)

    data.job.replace(('technician','services','retired','blue-collar','entrepreneur','admin.',

                      'housemaid','student','self-employed','management',

                      'unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True )

    return data





def experiment_generator(train_feats, train_class):

    accuracy = []

    penalties = []

    G = .00000001

    penalty = 1

    N = 10

    for item in range(N):

        clf = SVC(kernel='rbf', random_state = 0, gamma = G, C = penalty)

        clf.fit(train_feats, train_class.values.ravel())

        pred_train = clf.predict(train_feats)

        # Accuracy score

        s_train = accuracy_score(train_class, pred_train)

        # Store values for plotting

        penalties.append(penalty)

        accuracy.append(s_train)

        # Increase experiment parameters

        penalty += 1

        G += .00000001

    plt.scatter(penalties, accuracy)

    plt.ylabel('Accuracy (%')

    plt.xlabel('Penalty - C Parameter')

    plt.show()





    

data = normalize(data)

plt.hist((data.duration),bins=100)

plt.ylabel('Occurences (Frequency)')

plt.xlabel('Client Call Duration')

plt.show()

plt.hist((data.job),bins=10)

plt.ylabel('Occurences (Frequency)')

plt.xlabel('Client Job Indices')

plt.show()

plt.hist((data.balance),bins=10)

plt.ylabel('Occurences (Frequency)')

plt.xlabel('Client Balance')

plt.show()



# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(data, data.y, test_size=0.2)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)



df_train = X_train

df_test = X_test



df_train_class = pd.DataFrame(df_train['y'])    

df_train_features = df_train.loc[:, df_train.columns != 'y']



df_test_class = pd.DataFrame(df_test['y'])

df_test_features = df_test.loc[:, df_test.columns != 'y']



g = .0001

c = 1



mlp_classifier = SVC(kernel='rbf', random_state = 0, gamma = g, C = c)

mlp_classifier.fit(df_train_features, df_train_class.values.ravel())

                     

predicted_train = mlp_classifier.predict(df_train_features)

predicted_test = mlp_classifier.predict(df_test_features)



# Accuracy score

score_train = accuracy_score(df_train_class, predicted_train)

score_test = accuracy_score(df_test_class, predicted_test)



print('Training Accuracy Score: {}'.format(score_train))

print('Testing Accuracy Score: {}'.format(score_test))

   

# Precision, Recall  

precision_train = precision_score(df_train_class, predicted_train)

precision_test = precision_score(df_test_class, predicted_test)

print('Training Precision: {}'.format(precision_train))

print('Testing Precision: {}'.format(precision_test))



recall_train = recall_score(df_train_class, predicted_train)

recall_test = recall_score(df_test_class, predicted_test)

print('Training Recall: {}'.format(recall_train))

print('Testing Recall: {}'.format(recall_test))



# Classification Report

print('Training Classification Report: ')

print(classification_report(df_train_class, predicted_train))

print('Testing Classification Report: ')

print(classification_report(df_test_class, predicted_test))



# Experiments

experiment_generator(df_train_features, df_train_class)
