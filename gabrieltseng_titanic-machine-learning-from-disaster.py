%matplotlib inline 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as p



#Using Scikit-Learn for the machine learning algorithms

from sklearn import linear_model 

from sklearn import preprocessing

from sklearn import metrics

from sklearn import svm
train = pd.read_csv('../input/train.csv') #contains whether or not the passengers survived

test = pd.read_csv("../input/test.csv") # does not; this is to be submitted to Kaggle 

full = train.append( test , ignore_index = True ) #appends, ignoring the Survived column 

train.head() #hello, data! 
train["Embarked"].replace(to_replace = ['C','S','Q'], value = [0.3,0.6,0.9], inplace=True);

train["Sex"].replace(to_replace=['male', 'female'], value = [0,1], inplace=True);



test["Embarked"].replace(to_replace = ['C','S','Q'], value = [0.3,0.6,0.9] ,inplace=True );

test["Sex"].replace(to_replace=['male', 'female'], value = [0,1],inplace=True );
train["StringTickets"] = train["Ticket"].astype(str)

train["StringTicketsSplit"] = train.StringTickets.str.split()

train.StringTicketsSplit = train.StringTicketsSplit.apply(lambda x: x[-1])

train.StringTicketsSplit = pd.to_numeric(train.StringTicketsSplit, errors='coerce', downcast = 'integer')

del train["StringTickets"] #This just cleans up an unnecessary column 



#And, for the test dataset too

test["StringTickets"] = test["Ticket"].astype(str)

test["StringTicketsSplit"] = test.StringTickets.str.split()

test.StringTicketsSplit = test.StringTicketsSplit.apply(lambda x: x[-1])

test.StringTicketsSplit = pd.to_numeric(test.StringTicketsSplit)

del test["StringTickets"]

train.head()
train.isnull().sum(), test.isnull().sum()
train['Embarked'].fillna(train['Embarked'].mean(), inplace = True)

train["StringTicketsSplit"].fillna(train["StringTicketsSplit"].mean(), inplace = True)



test["Fare"].fillna(test["Fare"].mean(), inplace = True)



train.isnull().sum(), test.isnull().sum()
pd.DataFrame.hist(train, column = "Age")
mean_test = train["Age"].copy()

filled = mean_test.fillna(mean_test.mean())

pd.Series.hist(filled), mean_test.mean()
# Step 1; seperating the data into the training data (where Age != NaN), and the test data



lin_train = train[np.isfinite(train["Age"])]

lin_test = train[np.isfinite(train["Age"])==False]

lin_train.shape
#Sweet; one last thing is to create a 'validation' dataset. I'll just shuffle the data and take the first

#100 points to be the validation data. 



shuffled = lin_train.sample(frac = 1)



X_train = shuffled.iloc[:int(len(shuffled.index)*0.75),:];

Y_train = shuffled["Age"].iloc[:int(len(shuffled.index)*0.75)];



X_cv = shuffled.iloc[int(len(shuffled.index)*0.75):,:];

Y_cv = shuffled["Age"].iloc[int(len(shuffled.index)*0.75):];



X_train.head()
def error_vs_m_linreg(X_train, Y_train, X_cv, Y_cv, plot=True):

    error_cv = [];

    error_train = [];

    linreg = linear_model.LinearRegression();

    

    for m in range(10, len(X_train.index)-1):

        linreg.fit(X_train.iloc[:m,:], Y_train.iloc[:m]);

        error_cv.append(metrics.mean_absolute_error(linreg.predict(X_cv), Y_cv)); 

        error_train.append(metrics.mean_absolute_error(linreg.predict(X_train.iloc[:m,:]), Y_train.iloc[:m]));



    if plot: 

        m = range(10, len(X_train.index)-1);

        p.plot(m, error_cv, label = 'cv');

        p.plot(m, error_train, label = 'train');

        p.xlabel("Training set size")

        p.ylabel("Mean Absolute Error")

        p.legend()

        p.show()

    

    return error_cv[len(m)-1], error_train[len(m)-1]
#First, testing the use of PClass and SibSp



X1 = X_train.iloc[:,[2,6]]

X1_cv = X_cv.iloc[:,[2,6]]



error_vs_m_linreg(X1, Y_train, X1_cv, Y_cv)
#Now, including Parch

X2 = X_train.iloc[:,[2,6,7]]

X2_cv = X_cv.iloc[:,[2,6,7]]



error_vs_m_linreg(X2, Y_train, X2_cv, Y_cv)
#And finally, including Fare

X3 = X_train.iloc[:,[2,6,7,9]]

X3_cv = X_cv.iloc[:,[2,6,7,9]]

error_vs_m_linreg(X3, Y_train, X3_cv, Y_cv)
#First, making the model: 

linreg = linear_model.LinearRegression()

linreg.fit(X2, Y_train)
for i in range(train.shape[0]): 

    if (np.isfinite(train["Age"][i])==False):

        Age_pred = linreg.predict(train.iloc[i, [2,6,7]].values.reshape(1,-1))[0]

        train["Age"][i] = Age_pred
train.head()
test.iloc[1, [1,5,6,8]]

for i in range(test.shape[0]):

    if (np.isfinite(test["Age"][i])==False):

        Age_pred = linreg.predict(test.iloc[i, [1,5,6]].values.reshape(1,-1))[0]

        test["Age"][i] = Age_pred
test.head()
train.isnull().sum(), test.isnull().sum()
#First, mean normalization 

train.StringTicketsSplit = train.StringTicketsSplit.apply(lambda x: x - train.StringTicketsSplit.mean())

test.StringTicketsSplit = test.StringTicketsSplit.apply(lambda x: x - train.StringTicketsSplit.mean())

#Feature scaling the ticket numbers

ticket_max = train.StringTicketsSplit.max()

ticket_min = train.StringTicketsSplit.min()



train.StringTicketsSplit = train.StringTicketsSplit.apply(lambda x: (x-ticket_min)/(ticket_max-ticket_min))

test.StringTicketsSplit = test.StringTicketsSplit.apply(lambda x: (x-ticket_min)/(ticket_max-ticket_min))



#Mean normalization on the fare

train.Fare = train.Fare.apply(lambda x: x-train.Fare.mean())

test.Fare = test.Fare.apply(lambda x: x-train.Fare.mean())



#Feature scaling the fare

fare_max = train.Fare.max()

fare_min = train.Fare.min()



train.Fare = train.Fare.apply(lambda x: (x-fare_min)/(fare_max-fare_min))

test.Fare = test.Fare.apply(lambda x: (x-fare_min)/(fare_max-fare_min))



#Mean normalization of the ages

train.Age = train.Age.apply(lambda x: x-train.Age.mean())

test.Age = test.Age.apply(lambda x: x-train.Age.mean())



#Feature scaling the Ages

age_max = train.Age.max()

age_min = train.Age.min()



train.Age = train.Age.apply(lambda x: (x-age_min)/(age_max - age_min))

test.Age = test.Age.apply(lambda x: (x-age_min)/(age_max-age_min))







train.head()
#To split the data, I first need to shuffle it

shuffled = train.sample(frac = 1); #By making frac=1, I shuffle it all

shuffled.head()
#Now that its shuffled, I want to split it 75:25 into the training and cross validation sets 

X_train = shuffled.iloc[:int(len(shuffled.index)*0.75),[2,4,5,6,7,9,11,12]];

Y_train = shuffled["Survived"].iloc[:int(len(shuffled.index)*0.75)];



X_cv = shuffled.iloc[int(len(shuffled.index)*0.75):,[2,4,5,6,7,9,11,12]];

Y_cv = shuffled["Survived"].iloc[int(len(shuffled.index)*0.75):];



print (X_train.shape), (Y_train.shape), (X_cv.shape), (Y_cv.shape)

#X_train.head()
logreg = linear_model.LogisticRegression();

logreg.fit(X_train, Y_train)

logreg.score(X_train, Y_train), logreg.score(X_cv, Y_cv)
def error_vs_m_logreg(X_train, Y_train, X_cv, Y_cv, C=1, plot=True):

    error_cv = [];

    error_train = [];

    logreg = linear_model.LogisticRegression(C=C);

    for m in range(10, len(X_train.index)-1):

        logreg.fit(X_train.iloc[:m,:], Y_train.iloc[:m]);

        error_cv.append(metrics.mean_absolute_error(logreg.predict(X_cv), Y_cv)); 

        error_train.append(metrics.mean_absolute_error(logreg.predict(X_train.iloc[:m,:]), Y_train.iloc[:m]));



    if plot: 

        m = range(10, len(X_train.index)-1);

        p.plot(m, error_cv, label = 'cv');

        p.plot(m, error_train, label = 'train');

        p.xlabel("Training set size")

        p.ylabel("Mean Absolute Error")

        p.legend()

        p.show()

    

    f1_cv = metrics.f1_score(Y_cv, logreg.predict(X_cv))

    f1_train = metrics.f1_score(Y_train, logreg.predict(X_train))

    return error_cv[len(m)-1], error_train[len(m)-1], f1_cv, f1_train



error_vs_m_logreg(X_train, Y_train, X_cv, Y_cv)
possible_C = np.logspace(-6, 1, num=25, base = 3);

def optimize_c_logreg(possible_C, X_train, Y_train, X_cv, Y_cv, plot = True):

    C_error_train = [];

    C_error_cv = [];

    for i in range(len(possible_C)):

        logreg = linear_model.LogisticRegression(C = possible_C[i])

        logreg.fit(X_train, Y_train);

        C_error_train.append(metrics.mean_absolute_error(logreg.predict(X_train), Y_train));

        C_error_cv.append(metrics.mean_absolute_error(logreg.predict(X_cv), Y_cv)); 

    if plot:

        p.plot(possible_C, C_error_train, label = 'train');

        p.plot(possible_C, C_error_cv, label = 'cv'); 

        p.xlabel("C value (1/lambda)")

        p.ylabel("Mean Absolute Error")

        p.legend()

        p.show()

    C_min = possible_C[np.argmin(C_error_cv)]

    return C_min



C_min = optimize_c_logreg(possible_C, X_train, Y_train, X_cv, Y_cv)

    
cv_error, train_error, f1_cv, f1_train = error_vs_m_logreg(X_train, Y_train, X_cv, Y_cv, C=C_min)



cv_error, train_error, f1_cv, f1_train
svm_model = svm.SVC();

svm_model.fit(X_train, Y_train);

svm_model.score(X_train, Y_train), svm_model.score(X_cv, Y_cv)
def error_vs_m_svc(X_train, Y_train, X_cv, Y_cv, C=1, plot=True):

    error_cv = [];

    error_train = [];

    svc = svm.SVC(C=C);

    for m in range(10, len(X_train.index)-1):

        svc.fit(X_train.iloc[:m,:], Y_train.iloc[:m]);

        error_cv.append(metrics.mean_absolute_error(svc.predict(X_cv), Y_cv)); 

        error_train.append(metrics.mean_absolute_error(svc.predict(X_train.iloc[:m,:]), Y_train.iloc[:m]));



    if plot: 

        m = range(10, len(X_train.index)-1);

        p.plot(m, error_cv, label = 'cv');

        p.plot(m, error_train, label = 'train');

        p.xlabel("Training set size")

        p.ylabel("Mean Absolute Error")

        p.legend()

        p.show()

    

    f1_cv = metrics.f1_score(Y_cv, svc.predict(X_cv))

    f1_train = metrics.f1_score(Y_train, svc.predict(X_train))

    return error_cv[len(m)-1], error_train[len(m)-1], f1_cv, f1_train



error_vs_m_svc(X_train, Y_train, X_cv, Y_cv)
possible_C = np.logspace(-6, 1, num=25, base = 3);

def optimize_c_svc(possible_C, X_train, Y_train, X_cv, Y_cv, plot = True):

    C_error_train = [];

    C_error_cv = [];

    for i in range(len(possible_C)):

        svc = svm.SVC(C = possible_C[i])

        svc.fit(X_train, Y_train);

        C_error_train.append(metrics.mean_absolute_error(svc.predict(X_train), Y_train));

        C_error_cv.append(metrics.mean_absolute_error(svc.predict(X_cv), Y_cv)); 

    if plot:

        p.plot(possible_C, C_error_train, label = 'train');

        p.plot(possible_C, C_error_cv, label = 'cv'); 

        p.xlabel("C value (1/lambda)")

        p.ylabel("Mean Absolute Error")

        p.legend()

        p.show()

    C_min = possible_C[np.argmin(C_error_cv)]

    return C_min



C_min = optimize_c_svc(possible_C, X_train, Y_train, X_cv, Y_cv)

C_min
cv_error, train_error, f1_cv, f1_train = error_vs_m_svc(X_train, Y_train, X_cv, Y_cv, C=C_min)

cv_error, train_error, f1_cv, f1_train