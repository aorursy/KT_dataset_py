import csv

import numpy

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt 

from sklearn.cross_validation import train_test_split



def SexToNum(s):

    if s == 'male':

        return 1

    elif s == 'female':

        return 0

    else:

        return -1

    

def readFeature(file_name, isTest):

    i = 0

    if isTest:

        feature_idx = [0, 1, 3, 4]

    else:

        feature_idx = [0, 1, 2, 4, 5]

        

    X = numpy.zeros((len(feature_idx), 1));

    valid_num = 0

    with open(file_name) as csvfile:

        reader = csv.reader(csvfile)

        for row in reader:

            i = i + 1

            if i <= 1:

                print(row)

                continue



            feature = numpy.zeros((len(feature_idx), 1))

            j = 0

            if isTest:

                feature[j] = int(row[0])

                j = j + 1

                

                feature[j] = float(row[1])

                j = j + 1



                feature[j] = SexToNum(row[3])

                j = j + 1

                if row[4] == '':

                    continue

                feature[j] = float(row[4])

                j = j + 1

            else:

                feature[j] = int(row[0])

                j = j + 1

                feature[j] = float(row[1])

                j = j + 1



                feature[j] = float(row[2])

                j = j + 1



                feature[j] = SexToNum(row[4])

                j = j + 1

                if row[5] == '':

                    continue

                feature[j] = float(row[5])

                j = j + 1



                valid_num = valid_num + 1

           # print(valid_num)

           # print(row)

          #  print(feature)

            X = numpy.concatenate((X, feature), axis=1)

    return X[:, 1:]



def outputPrediction(test_file_name, output_file_name, prediction, t_passenger_id):

    i = 0

    j = 0

    

    new_rows = []

    with open(test_file_name) as csvfile:

        reader = csv.reader(csvfile)

        for row in reader:

            i = i + 1

            if i <= 1:

                new_row = [row[0], 'Survived']

              #  new_row.extend(row[1:])

                new_rows.append(new_row)

              #  print(new_row)

            elif j < t_passenger_id.shape[0] and int(row[0]) == t_passenger_id[j]:

                    new_row = [row[0], t[j]]

                  #  new_row.extend(row[1:])

                    new_rows.append(new_row)

                  #  print(new_row)

                    j = j + 1

            else:

                new_row = [row[0], 0]

                new_rows.append(new_row)

                    

                    

    with open(output_file_name, 'w', newline='') as f:

        writer = csv.writer(f)

        for row in new_rows:

            writer.writerow(row)

                

X = readFeature('../input/train.csv', False)

print(X.shape)

T = readFeature('../input/test.csv', True)

print(T.shape)



y = X[1,:]

X = X[2: ,:]

X = numpy.transpose(X)

t_passenger_id = T[0, :]

T = numpy.transpose(T[1:, :])

print(X.shape)

print(y.shape)

print(T.shape)



X_train,X_test,Y_train,Y_test = train_test_split(X, y,test_size=0.25)



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)



classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, Y_train)

print(classifier.score(X_train, Y_train))

print(classifier.score(X_test, Y_test))

t = classifier.predict(T)

print(t.shape)

print(t_passenger_id.shape)

outputPrediction('../input/test.csv', 'submission.csv', t, t_passenger_id)
