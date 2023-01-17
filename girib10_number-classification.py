import numpy as np 

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
#Loading dataset

digits = load_digits()



#Prints data shape

print("Image Data Shape" , digits.data.shape)

print("Label Data Shape", digits.target.shape)
plt.figure(figsize=(15,4))

#Prints first 5 data values

for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):

    plt.subplot(1, 10, index + 1)

    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)

    plt.title('Training: %i\n' % label, fontsize = 8)
#Splits training and test data (25% and 75%)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression

import time

score_max_i_100 = []

times_max_i_100 = []

score_conv = []

times_conv = []

solvers = ["Newton-cg-> ","L-BFGS-> ", "SAG-> ","SAGA-> "]
#Trains model using Logistic Regression model

#Solver "newton-cg"

logisticRegr = LogisticRegression(random_state=0, solver='newton-cg',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_max_i_100.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_max_i_100.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "lbfgs"

logisticRegr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_max_i_100.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_max_i_100.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "sag"

logisticRegr = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_max_i_100.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_max_i_100.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "saga"

logisticRegr = LogisticRegression(random_state=0, solver='saga',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_max_i_100.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_max_i_100.append(score)

print(score)
for i in range(4):

    print(solvers[i],"Accuracy:",round(100*score_max_i_100[i],2)," Time: ",round(times_max_i_100[i],4))
#Trains model using Logistic Regression model

#Solver "newton-cg"

logisticRegr = LogisticRegression(random_state=0, max_iter=37, solver='newton-cg',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_conv.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_conv.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "lbfgs"

logisticRegr = LogisticRegression(random_state=0, max_iter=3045, solver='lbfgs',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_conv.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_conv.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "sag"

logisticRegr = LogisticRegression(random_state=0, max_iter=689, solver='sag',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_conv.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_conv.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "saga"

logisticRegr = LogisticRegression(random_state=0, max_iter=1009, solver='saga',multi_class='multinomial')

start_time = time.time()

logisticRegr.fit(x_train, y_train)

times_conv.append(time.time()-start_time)
#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_conv.append(score)

print(score)
for i in range(4):

    print(solvers[i],"Accuracy:",round(100*score_conv[i],2)," Time: ",round(times_conv[i],4))
#Splitting dataset 5% for test and 95% for training

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=0)



score_sag = []

score_saga = []
#Trains model using Logistic Regression model

#Solver "sag"

logisticRegr = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial')

logisticRegr.fit(x_train, y_train)



#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_sag.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "sag"

logisticRegr = LogisticRegression(random_state=0, max_iter = 930, solver='sag',multi_class='multinomial')

logisticRegr.fit(x_train, y_train)



#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_sag.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "saga"

logisticRegr = LogisticRegression(random_state=0, solver='saga',multi_class='multinomial')

logisticRegr.fit(x_train, y_train)



#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_saga.append(score)

print(score)
#Trains model using Logistic Regression model

#Solver "saga"

logisticRegr = LogisticRegression(random_state=0, max_iter = 1173, solver='saga',multi_class='multinomial')

logisticRegr.fit(x_train, y_train)



#Calculating model accuracy

score = logisticRegr.score(x_test, y_test)

score_saga.append(score)

print(score)
print("SAG:")

print("Accuracy(without convergeance):",round(100*score_sag[0],2))

print("Accuracy(with convergeance):",round(100*score_sag[1],2))

print("SAGA:")

print("Accuracy(without convergeance):",round(100*score_saga[0],2))

print("Accuracy(with convergeance):",round(100*score_saga[1],2))