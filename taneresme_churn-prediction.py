import pandas as pd



# read data

dataset = pd.read_csv("../input/Churn_Modelling.csv", header=0)

dataset.head()
dataset.describe()
# Unique geographic values in visual_ds

if dataset["Gender"][0] in ("Male", "Female"):

    unique_geography = dataset[["Geography"]].groupby("Geography").size()

print(unique_geography)
# we are going to modify the copy of dataset for visualation

visual_ds = dataset.copy()



# transform gender values geographic values

if visual_ds["Gender"][0] in ("Male", "Female"):

    for row in [visual_ds]:

        row["Gender"] = row["Gender"].map( {"Male": 0, "Female": 1} ).astype(int)

        row["Geography"] = row["Geography"].map( {"France": 0, "Germany": 1, "Spain": 2} ).astype(int)

    # end of for loop

# end of if clause



# transform age values

if visual_ds["Age"][0] > 17:

    visual_ds.loc[ visual_ds["Age"] <= 16, "AgeRange"] = 0

    visual_ds.loc[(visual_ds["Age"] > 16) & (visual_ds["Age"] <= 32), "AgeRange"] = 1

    visual_ds.loc[(visual_ds["Age"] > 32) & (visual_ds["Age"] <= 48), "AgeRange"] = 2

    visual_ds.loc[(visual_ds["Age"] > 48) & (visual_ds["Age"] <= 64), "AgeRange"] = 3

    visual_ds.loc[ visual_ds["Age"] > 64, "AgeRange"] = 4

# end of if clause



# transform CreditScore values

if visual_ds["CreditScore"][0] > 4:

    visual_ds.loc[ visual_ds["CreditScore"] <= 450, "CreditScoreRange"] = 0

    visual_ds.loc[(visual_ds["CreditScore"] > 450) & (visual_ds["CreditScore"] <= 550), "CreditScoreRange"] = 1

    visual_ds.loc[(visual_ds["CreditScore"] > 550) & (visual_ds["CreditScore"] <= 650), "CreditScoreRange"] = 2

    visual_ds.loc[(visual_ds["CreditScore"] > 650) & (visual_ds["CreditScore"] <= 750), "CreditScoreRange"] = 3

    visual_ds.loc[ visual_ds["CreditScore"] > 750, "CreditScoreRange"] = 4

# end of if clause



# transform CreditScore values

if visual_ds["Balance"][1] > 4:

    visual_ds.loc[ visual_ds["Balance"] <= 50000, "BalanceRange"] = 0

    visual_ds.loc[(visual_ds["Balance"] > 50000) & (visual_ds["Balance"] <= 100000), "BalanceRange"] = 1

    visual_ds.loc[(visual_ds["Balance"] > 100000) & (visual_ds["Balance"] <= 150000), "BalanceRange"] = 2

    visual_ds.loc[(visual_ds["Balance"] > 150000) & (visual_ds["Balance"] <= 200000), "BalanceRange"] = 3

    visual_ds.loc[ visual_ds["Balance"] > 200000, "BalanceRange"] = 4

# end of if clause



# transform EstimatedSalary values

if visual_ds["EstimatedSalary"][0] > 4:

    visual_ds.loc[ visual_ds["EstimatedSalary"] <= 40000, "EstimatedSalaryRange"] = 0

    visual_ds.loc[(visual_ds["EstimatedSalary"] > 40000) & (visual_ds["EstimatedSalary"] <= 80000), \

                "EstimatedSalaryRange"] = 1

    visual_ds.loc[(visual_ds["EstimatedSalary"] > 80000) & (visual_ds["EstimatedSalary"] <= 120000), \

                "EstimatedSalaryRange"] = 2

    visual_ds.loc[(visual_ds["EstimatedSalary"] > 120000) & (visual_ds["EstimatedSalary"] <= 160000), \

                "EstimatedSalaryRange"] = 3

    visual_ds.loc[ visual_ds["EstimatedSalary"] > 160000, "EstimatedSalaryRange"] = 4

# end of if clause





if "RowNumber" in visual_ds.columns:

    visual_ds = visual_ds.drop(["RowNumber"], axis=1)

if "CustomerId" in visual_ds.columns:

    visual_ds = visual_ds.drop(["CustomerId"], axis=1)

if "Surname" in visual_ds.columns:

    visual_ds = visual_ds.drop(["Surname"], axis=1)



visual_ds.describe()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col='Exited')

g.map(plt.hist, 'Geography', bins=5)



means = visual_ds[['Geography', 'Exited']].groupby(["Geography"], as_index=False).mean()



labels = '0-France', '1-Germany', '2-Spain'

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total, means["Exited"][2] / total]



explode = (0.1, 0.1, 0.1) 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. Geography")



plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col = 'Exited')

g.map(plt.hist, 'Gender', bins = 3)



means = visual_ds[['Gender', 'Exited']].groupby(["Gender"], as_index = False).mean()



labels = '0-Male', '1-Female'

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total]



explode = (0.1, 0.1) 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. Gender")



plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col = "Exited")

g.map(plt.hist, "AgeRange", bins = 5)



means = visual_ds[["AgeRange", "Exited"]].groupby(["AgeRange"], as_index = False).mean()



labels = "16-32", "32-48", "48-64", "64+"

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total \

        , means["Exited"][2] / total, means["Exited"][3] / total]



explode = (0.1, 0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. AgeRange")



plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col = "Exited")

g.map(plt.hist, "CreditScoreRange", bins = 5)



means = visual_ds[["CreditScoreRange", "Exited"]].groupby(["CreditScoreRange"], as_index = False).mean()



labels = "0-450", "450-550", "550-650", "650-750", "750+"

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total \

        , means["Exited"][2] / total, means["Exited"][3] / total, means["Exited"][4] / total]



explode = (0.1, 0.1, 0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. CreditScoreRange")



plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col = "Exited")

g.map(plt.hist, "BalanceRange", bins = 9)



means = visual_ds[["BalanceRange", "Exited"]].groupby(["BalanceRange"], as_index = False).mean()



labels = "0-50000", "50000-100000", "100000-150000", "150000-200000", "200000+"

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total \

        , means["Exited"][2] / total, means["Exited"][3] / total, means["Exited"][4] / total]



explode = (0.1, 0.1, 0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels = labels, autopct = "%1.1f%%", shadow = True, startangle = 90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. BalanceRange")



plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col = "Exited")

g.map(plt.hist, "EstimatedSalaryRange", bins = 9)



means = visual_ds[["EstimatedSalaryRange", "Exited"]].groupby(["EstimatedSalaryRange"], as_index = False).mean()



labels = "0-40000", "40000-80000", "80000-120000", "120000-160000", "160000+"

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total \

        , means["Exited"][2] / total, means["Exited"][3] / total, means["Exited"][4] / total]



explode = (0.1, 0.1, 0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels = labels, autopct = "%1.1f%%", shadow = True, startangle = 90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. EstimatedSalaryRange")



plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.FacetGrid(visual_ds, col = "Exited")

g.map(plt.hist, "Tenure", bins = 21)



means = visual_ds[["Tenure", "Exited"]].groupby(["Tenure"], as_index = False).mean()



labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

total = means["Exited"].sum()

sizes = [means["Exited"][0] / total, means["Exited"][1] / total, \

         means["Exited"][2] / total, means["Exited"][3] / total, \

         means["Exited"][4] / total, means["Exited"][5] / total, \

         means["Exited"][6] / total, means["Exited"][7] / total, \

         means["Exited"][8] / total, means["Exited"][9] / total, \

        means["Exited"][10] / total]



explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels = labels, autopct = "%1.1f%%", shadow = True, startangle = 90)

ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle("Exit rate w.r.t. Tenure")



plt.show()
# transform gender values geographic values

if dataset["Gender"][0] in ("Male", "Female"):

    for row in [dataset]:

        row["Gender"] = row["Gender"].map( {"Male": 0, "Female": 1} ).astype(int)

        row["Geography"] = row["Geography"].map( {"France": 0, "Germany": 1, "Spain": 2} ).astype(int)

    # end of for loop

# end of if clause



# transform age values

if dataset["Age"][0] > 17:

    max_age = dataset["Age"].max()

    # mapping

    dataset["Age"] = dataset["Age"] * (4 / max_age)

# end of if clause



# transform CreditScore values

if dataset["CreditScore"][0] > 4:

    max_creditScore = dataset["CreditScore"].max()

    # mapping

    dataset["CreditScore"] = dataset["CreditScore"] * (4 / max_creditScore)

# end of if clause



# transform CreditScore values

if dataset["Balance"][1] > 4:

    max_balance = dataset["Balance"].max()

    # mapping

    dataset["Balance"] = dataset["Balance"] * (4 / max_balance)

# end of if clause



# transform EstimatedSalary values

if dataset["EstimatedSalary"][0] > 4:

    max_estimatedSalary = dataset["EstimatedSalary"].max()

    # mapping

    dataset["EstimatedSalary"] = dataset["EstimatedSalary"] * (4 / max_estimatedSalary)

# end of if clause



# transform Tenure values

if dataset["Tenure"][0] > 0:

    max_tenure = dataset["Tenure"].max()

    # mapping

    dataset["Tenure"] = dataset["Tenure"] * (4 / max_tenure)

# end of if clause



if "RowNumber" in dataset.columns:

    dataset = dataset.drop(["RowNumber"], axis=1)

if "CustomerId" in dataset.columns:

    dataset = dataset.drop(["CustomerId"], axis=1)

if "Surname" in dataset.columns:

    dataset = dataset.drop(["Surname"], axis=1)



dataset.describe()
test_indis = 10



train_dataset = dataset[dataset.index % test_indis != 0]

train_dataset.describe()
test_dataset = dataset[dataset.index % test_indis == 0]

test_dataset.describe()
from sklearn.linear_model import LogisticRegression



X_train = train_dataset.drop(["Exited"], axis=1)

Y_train = train_dataset.Exited

X_test  = test_dataset.drop(["Exited"], axis=1).copy()

Y_test  = test_dataset.Exited



# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

#print(Y_pred)

acc_log = round(logreg.score(X_test, Y_test), 3)

print("accuracy:", acc_log)
# import pytorch libraries

import os

import numpy as np

import torch as py

from torch.autograd import Variable



columns = train_dataset.columns.drop("Exited")



# determine learning rate

eta = 0.00002



# create a linear model. y = w*x + b

model = py.nn.Linear(10, 1)



# define an optimizer. I have defined Stochastic Gradient Descent.

optim = py.optim.SGD(model.parameters(), lr=eta)



# define loss function. It is Binary Cross Entropy as a loss function.

loss = py.nn.BCELoss(size_average=True)



# definition of sigmoid funtion to normalize calculation to between 0 and 1 for BCELoss.

sigmoid = py.nn.Sigmoid()



def gradient(X, y, epoch):

    # gradient steps

    for i in range(epoch):

        # compute loss value.

        dE = loss(sigmoid(model(Variable(X))) , Variable(y))

        # reset of gradient buffer

        optim.zero_grad()

        # compute gradients with respect to vectors of X by using auto differitiation.

        dE.backward()

        # update optimizer.

        optim.step()

# end of def gradient



def train():

    # get train visual_ds

    X = py.from_numpy(np.array(train_dataset[columns])).float()

    # get classes from train visual_ds

    y = py.from_numpy(np.array([[1] if v == 1 else [0] for v in np.array(train_dataset[["Exited"]])])).float()

    # train them.

    gradient(X, y, 100)

# end of def train



def test():

    # get test_visual_ds

    X = py.from_numpy(np.array(test_dataset[columns])).float()

    # get classes from test visual_ds.

    y = [1 if v == 1 else 0 for v in np.array(test_dataset[["Exited"]])]

    #print("y:",y)

    # classify them

    Z = sigmoid(model(Variable(X)))

    # get classification results.

    Z_dist = np.array([1 if (v.data >= 0.5).numpy() else 0 for v in Z])

    #print("s:", Z_dist)

    

    # compute accuracy.

    correct = (y == Z_dist).sum()

    

    print("accuracy:", correct / len(y))

# end of def test



train()

test()
# import libraries

import pandas as pd

import numpy as np

import math

import numpy as np



def sigmoid(t):

    return np.exp(t)/(1+np.exp(t))

# end of def sigmoid



def gradientAscentStep(eta, X, y, w):

    dL = np.dot(X.T, y - sigmoid(np.dot(X, w)))

    w = w + eta * dL

    return w

# end of def gradientAscentStep



def gradientAscent(eta, X, y, w, epoch):

    for i in range(epoch):

        w = gradientAscentStep(eta, X, y, w)

        #print("w:",w)

    return w

# end of def gradientAscent



# total count of sample space

total = float(len(train_dataset))



columns = train_dataset.columns.drop("Exited")



def train(eta, w):

    X = np.array(train_dataset[columns])

    y = np.array([1 if v == 1 else 0 for v in np.array(train_dataset[["Exited"]])])

    w = gradientAscent(eta, X, y, w, 1000)

    return w

# end of def train



def test(w):

    X = np.array(test_dataset[columns])

    y = np.array([1 if v == 1 else 0 for v in np.array(test_dataset[["Exited"]])])

    #print("y:\n",y)

    Z = np.dot(X, w)

    #print("Z:", sigmoid(Z))

    Z_dist = [1 if v >= 0.5 else 0 for v in sigmoid(Z)]

    #print("s:\n", np.array(Z_dist))

    

    # compute accuracy

    correct = (y == Z_dist).sum()

    

    print("accuracy:", correct / len(y))

# end of def test



# intial values

w = np.array(train_dataset.drop("Exited", axis=1).mean())

#w = np.array(np.zeros(len(columns)))

eta = 0.000009

#eta = 0.00009



w = train(eta, w)

#print("w:", w)

test(w)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



X_train = train_dataset.drop(["Exited"], axis=1)

Y_train = train_dataset.Exited

X_test  = test_dataset.drop(["Exited"], axis=1).copy()

Y_test  = test_dataset.Exited



gbc = GradientBoostingClassifier(n_estimators = 20, learning_rate = 1.0, max_depth = 10, random_state = 0)

gbc.fit(X_train, Y_train)

Y_pred = gbc.predict(X_test)

print("accuracy:", gbc.score(X_test, Y_test))
# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor



X_train = train_dataset.drop(["Exited"], axis = 1)

Y_train = train_dataset.Exited

X_test  = test_dataset.drop(["Exited"], axis = 1).copy()

Y_test  = test_dataset.Exited



gbc = GradientBoostingRegressor(n_estimators = 20, learning_rate = 1.0, max_depth = 10, random_state = 0, loss = "ls")

gbc.fit(X_train, Y_train)

Y_pred = gbc.predict(X_test)

print("accuracy:", gbc.score(X_test, Y_test))