#First we import a few libraries we will use extensively

import pandas as pd

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
from sklearn import model_selection, preprocessing



def open_data(file):

    #Readind training data

    data = pd.read_csv("../input/"+file)



    #Droping Name, Cabin and Ticket as probably irrelevant

    data = data.drop(["Name", "Ticket", "Cabin"], 1)

    

    #Converting string features into integers

    le = preprocessing.LabelEncoder()

    data["Sex"] = le.fit_transform(list(data["Sex"]))

    data["Embarked"] = le.fit_transform(list(data["Embarked"]))



    #Filling NaN values for age with the average value

    data["Age"] = data["Age"].fillna(value = data.Age.mean())

    data["Fare"] = data["Fare"].fillna(value = data.Fare.mean())

    

    return data
def param_label(data):

    data = data.drop(["PassengerId"], 1)

    return data.drop(["Survived"], 1), data[["Survived"]]
def subset_data(X, Y, n):

    return model_selection.train_test_split(X, Y, test_size = n)
data = open_data("train.csv")



y_true = data[["Survived"]]

y_test = np.array([1 for i in range(len(y_true))])

print("Accuracy for survived = 1: ",metrics.accuracy_score(y_true, y_test))
data = open_data("train.csv")



y_true = data[["Survived"]]

y_test = np.array([random.choice((0, 1)) for i in range(len(y_true))])

print("Accuracy for random survival: ",metrics.accuracy_score(y_true, y_test))
data = open_data("train.csv")



y_true = data[["Survived"]]

y_test = np.array([0 for i in range(len(y_true))])

print("Accuracy for survived = 1: ",metrics.accuracy_score(y_true, y_test))
#Loading test sample

data_test = open_data("test.csv")



#Setting all values to 0

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], 0] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])



#Saving as csv

solution.to_csv("solution_naive.csv", index=False)
from sklearn.neighbors import KNeighborsClassifier
data = open_data("train.csv")

X, Y = param_label(data)

x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)
#Model creation

model = KNeighborsClassifier()

model.fit(x_train, y_train)

acc = metrics.accuracy_score(model.predict(x_test), y_test)



print("Accuracy : " + str(acc))
neighboors = [i for i in range(1, 101)]



averages = []

mins = []

maxs = []



for n in neighboors:

    average_acc = 0

    min_acc = 1

    max_acc = 0

    

    #The accurracy may vary depending on the subset used, so we try 100 times with different subsets to get a better assessment.

    for i in range(100):

        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

        model = KNeighborsClassifier(n_neighbors = n)

        model.fit(x_train, y_train)

        acc = metrics.accuracy_score(model.predict(x_test), y_test)

        

        average_acc = average_acc + acc

        if acc > max_acc: max_acc = acc

        if acc < min_acc: min_acc = acc

        

    averages = averages + [average_acc/100]

    mins = mins + [min_acc]

    maxs = maxs + [max_acc]

    

#Ploting results

plt.figure(figsize=(24,8))

plt.plot(averages, color = 'r', linewidth=2)

plt.plot(mins, color = 'r', linestyle='--')

plt.plot(maxs, color = 'r', linestyle='--')

plt.xticks(neighboors)
#Model training

model = KNeighborsClassifier(n_neighbors = 12)

model.fit(X, Y)



#Results prediction and submission

data_test = open_data("test.csv")

prediction = model.predict(data_test.drop(["PassengerId"], 1))

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_KNN_allfeatures.csv", index=False)
data = open_data("train.csv")

X, Y = param_label(data)

X = X[["Pclass", "Sex"]]

x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)



model = KNeighborsClassifier()

model.fit(x_train, y_train)

acc = metrics.accuracy_score(model.predict(x_test), y_test)



print("Accuracy : " + str(acc))
model = KNeighborsClassifier(n_neighbors = 8)

model.fit(x_train, y_train)

acc = metrics.accuracy_score(model.predict(x_test), y_test)



print("Accuracy : " + str(acc))
#Model training

model = KNeighborsClassifier(n_neighbors = 8)

model.fit(X[["Pclass", "Sex"]], Y)



#Results prediction and submission

data_test = open_data("test.csv")

prediction = model.predict(data_test[["Pclass", "Sex"]])

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_KNN_Class_Sex.csv", index=False)
data["FamilyMembers"] = data["SibSp"]+data["Parch"]



X, Y = param_label(data)

X = X[["Pclass", "Sex", "FamilyMembers"]]

x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)



model = KNeighborsClassifier(n_neighbors = 8)

model.fit(x_train, y_train)

acc = metrics.accuracy_score(model.predict(x_test), y_test)



print("Accuracy : " + str(acc))
#Model training

model = KNeighborsClassifier(n_neighbors = 8)

model.fit(X[["Pclass", "Sex", "FamilyMembers"]], Y)



#Results prediction and submission

data_test = open_data("test.csv")

data_test["FamilyMembers"] = data_test["SibSp"]+data_test["Parch"]

prediction = model.predict(data_test[["Pclass", "Sex","FamilyMembers"]])

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_KNN_Class_Sex_Family.csv", index=False)
from sklearn import svm



data = open_data("train.csv")

X, Y = param_label(data)

x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)



model = svm.SVC()

model.fit(x_train, y_train)



y_predict = model.predict(x_test)

    

acc = metrics.accuracy_score(y_predict, y_test)

    

print("Accuracy:",acc)
kernels = ["rbf", "linear", "sigmoid", "poly"]



for kernel in kernels:

    model = svm.SVC(kernel = kernel)

    model.fit(x_train, y_train)

    

    y_predict = model.predict(x_test)

    

    acc = metrics.accuracy_score(y_predict, y_test)

    

    print("Accuracy with kernel =", kernel, ": ",acc)
#Model training

model = svm.SVC(kernel = 'linear')

model.fit(X,Y)



#Results prediction and submission

data_test = open_data("test.csv")

prediction = model.predict(data_test.drop(["PassengerId"], 1))

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_SVM.csv", index=False)
cs = [1, 5, 10, 15, 20]

gammas = [0.005, 0.01, 0.02, 0.05, 0.1]



for gamma in gammas:



    for c in cs:



        model = svm.SVC(kernel = 'rbf', C = c, gamma = gamma)

        model.fit(x_train, y_train)

        acc = metrics.accuracy_score(model.predict(x_test), y_test)

        print("Accuracy with gamma = ",gamma,"c = ",c,": ",acc)

        

    print("")
#Model training

model = svm.SVC(kernel = 'rbf', gamma =  0.01, C = 10)

model.fit(X,Y)



#Results prediction and submission

data_test = open_data("test.csv")

prediction = model.predict(data_test.drop(["PassengerId"], 1))

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_SVM_para.csv", index=False)
from sklearn.ensemble import RandomForestClassifier
data = open_data("train.csv")

X, Y = param_label(data)

x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)
model = RandomForestClassifier()

model.fit(x_train, y_train)



y_predict = model.predict(x_test)

y_predict

acc = metrics.accuracy_score(y_predict, y_test)



print(acc)
trees = [5, 10, 20, 50, 100]



averages = []

mins = []

maxs = []



for tree in trees:

    average_acc = 0

    min_acc = 1

    max_acc = 0

    

    for i in range(100):

        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)



        model = RandomForestClassifier(n_estimators = tree)

        model.fit(x_train, y_train)



        y_predict = model.predict(x_test)

        y_predict

        acc = metrics.accuracy_score(y_predict, y_test)



        average_acc = average_acc + acc

        if acc > max_acc: max_acc = acc

        if acc < min_acc: min_acc = acc



    averages = averages + [average_acc/100]

    mins = mins + [min_acc]

    maxs = maxs + [max_acc]

    

#Ploting results

plt.figure(figsize=(24,8))

plt.plot(averages, color = 'r', linewidth=2)

plt.plot(mins, color = 'r', linestyle='--')

plt.plot(maxs, color = 'r', linestyle='--')
depths = [1, 2, 5, 10, 15, 20]



averages = []

mins = []

maxs = []



for depth in depths:

    average_acc = 0

    min_acc = 1

    max_acc = 0

    

    for i in range(100):

        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)



        model = RandomForestClassifier(n_estimators = 20, max_depth = depth)

        model.fit(x_train, y_train)



        y_predict = model.predict(x_test)

        y_predict

        acc = metrics.accuracy_score(y_predict, y_test)



        average_acc = average_acc + acc

        if acc > max_acc: max_acc = acc

        if acc < min_acc: min_acc = acc



    averages = averages + [average_acc/100]

    mins = mins + [min_acc]

    maxs = maxs + [max_acc]

    

#Ploting results

plt.figure(figsize=(24,8))

plt.plot(averages, color = 'r', linewidth=2)

plt.plot(mins, color = 'r', linestyle='--')

plt.plot(maxs, color = 'r', linestyle='--')
#Model training

model = RandomForestClassifier(n_estimators = 20, max_depth = 10)

model.fit(X,Y)



#Results prediction and submission

data_test = open_data("test.csv")

prediction = model.predict(data_test.drop(["PassengerId"], 1))

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_Random_Forest.csv", index=False)
feature_w = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)

print(feature_w)
#Model training

model = RandomForestClassifier(n_estimators = 20, max_depth = 10)

model.fit(X[["Sex", "Age", "Fare", "Pclass"]],Y)



#Results prediction and submission

data_test = open_data("test.csv")

prediction = model.predict(data_test[["Sex", "Age", "Fare", "Pclass"]])

solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),

                        columns=['PassengerId', 'Survived'])

solution.to_csv("solution_Random_Forest_4features.csv", index=False)
import tensorflow as tf

from tensorflow import keras
data = open_data("train.csv")

X, Y = param_label(data)

x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)
#I use .describe() to get mean and standard deviation

stats = X.describe()

stats = stats.transpose()



def norm(x):

    return (x - stats['mean']) / stats['std']



normed_x_train = norm(x_train)

normed_x_test = norm(x_test)
def build_model():

    model = keras.Sequential()

    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer = 'uniform', input_shape=[len(X.keys())]))

    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer = 'uniform'))

    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer = 'uniform'))

    

    model.compile(loss='binary_crossentropy',

                optimizer='adam',

                metrics=['accuracy'])

    return model



model = build_model()

model.summary()
def plot_history(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.plot(hist['epoch'], hist['acc'])

    plt.ylim([0,1])

    plt.legend()



    plt.show()
model = build_model()



# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: print('')

        print('.', end='')



EPOCHS = 1000



# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)



history = model.fit(normed_x_train, y_train, epochs=EPOCHS, 

                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])



plot_history(history)
y_pred = model.predict(x_test)

y_pred = (y_pred > 0.5).astype(int).reshape(x_test.shape[0])

metrics.accuracy_score(y_pred, y_test)
architectures = [[12, 6],

                 [32, 16],

                 [64, 32],

                 [12, 12, 6],

                 [32, 16, 8],

                 [64, 32, 16],

                 [12, 12, 6, 6],

                 [32, 32, 16, 8],

                 [64, 32, 16, 8],

                 [64, 64, 32, 16, 8]]
def build_model(architecture):

    model = keras.Sequential()

    model.add(keras.layers.Dense(architecture[0], activation='relu', kernel_initializer = 'uniform', input_shape=[len(X.keys())]))

    

    for i in range(1, len(architecture)):

        n = architecture[i]

        model.add(keras.layers.Dense(n, activation='relu', kernel_initializer = 'uniform'))

        

    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer = 'uniform'))

    

    model.compile(loss='binary_crossentropy',

                optimizer='adam',

                metrics=['accuracy'])

    return model
EPOCHS = 1000



averages = []

mins = []

maxs = []



for architecture in architectures:

    model = build_model(architecture)

    

    average_acc = 0

    min_acc = 1

    max_acc = 0

    

    for i in range(100):

        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

        history = model.fit(normed_x_train, y_train, epochs=EPOCHS, 

                        validation_split = 0.2, verbose=0, callbacks=[early_stop])



        y_pred = model.predict(x_test)

        y_pred = (y_pred > 0.5).astype(int).reshape(x_test.shape[0])

        acc = metrics.accuracy_score(y_pred, y_test)

        

        average_acc = average_acc + acc

        if acc > max_acc: max_acc = acc

        if acc < min_acc: min_acc = acc

            

    averages = averages + [average_acc/100]

    mins = mins + [min_acc]

    maxs = maxs + [max_acc]

    

    print("Accuracy with architecture ", architecture, ": ", average_acc/100)

    

#Ploting results

plt.figure(figsize=(24,8))

plt.plot(averages, color = 'r', linewidth=2)

plt.plot(mins, color = 'r', linestyle='--')

plt.plot(maxs, color = 'r', linestyle='--')