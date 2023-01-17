import numpy as np
import pandas as pd
from matplotlib import pyplot
import re 
import os
import seaborn as sns
import sklearn.metrics as Metrics_tools
import warnings 

warnings.filterwarnings('ignore')

np.random.seed(1)
data_path = "../input/titanic"
print(f"Available files in this path '{data_path}' are: \n\t{os.listdir(data_path)}")

load_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
load_test = pd.read_csv(os.path.join(data_path, 'test.csv'))
load_train.head(4)
load_test.head(4)
PassengerId = {
    "train": load_train.loc[:,"PassengerId"].values,
    "test": load_test.loc[:,"PassengerId"].values
}

train_set = load_train.drop(['PassengerId'], axis=1)
test_set = load_test.drop(['PassengerId'], axis=1)
fig, ax_array = pyplot.subplots(2, 2, figsize=(8, 8))
fig.subplots_adjust(wspace=0.25, hspace=0.25)

ax_array = ax_array.ravel()
sets = [train_set, train_set, test_set, test_set]

for i, ax in enumerate(ax_array):
    ax.hist(sets[i].loc[:,"Sex"].map({'male':0, 'female':1}), bins=2, edgecolor="red", facecolor="lightblue")
    ax.set_title("Training set" if (i>=0 and i<=1) else "Test set")
    ax.axis([0., 1., 0., 600.])
    
pyplot.show()
Pos_training_examples = load_train[load_train.loc[:, "Survived"] == 1]
Neg_training_examples = load_train[load_train.loc[:, "Survived"] == 0]
fig = pyplot.figure(figsize=(10, 6))

ax_array = pyplot.axes()

# Plotting the positive examples.
ax_array.scatter(Pos_training_examples.loc[:, "Age"], Pos_training_examples.loc[:, "Fare"], 
                    s=90, edgecolor="black", facecolor="blue", marker="o", alpha=0.9, label="Survived")
# Plotting the negative examples.
ax_array.scatter(Neg_training_examples.loc[:, "Age"], Neg_training_examples.loc[:, "Fare"],
                    s=90, edgecolor="black", facecolor="red", marker="+", alpha=0.9, label="Non-survived")

ax_array.set_xlabel('Age')
ax_array.set_ylabel('Fare')
ax_array.set_title('Survived vs. non-surivived passengers.')    
ax_array.legend(loc="upper right", fancybox=True, shadow=True, frameon=True, framealpha=0.9)
    
pyplot.show()    
sns.distplot(a=Pos_training_examples.loc[:,'Pclass'],label='Alive',bins=5)
sns.distplot(a=Neg_training_examples.loc[:,'Pclass'],label='Dead',bins=5)
pyplot.legend(loc="upper right", fancybox=True, frameon=True, framealpha=0.9, shadow=True)
pyplot.xticks([1,2,3])
pyplot.show()
first_class_dead = Neg_training_examples[Neg_training_examples.loc[:, "Pclass"] == 1]
percentages = {}
percentages["female"] = np.mean(first_class_dead.loc[:, "Sex"].map({'male':0, 'female':1}) == 1)*100 
percentages["male"] = np.mean(first_class_dead.loc[:, "Sex"].map({'male':0, 'female':1}) == 0)*100  

fig = pyplot.figure(figsize=(6, 6))
pyplot.pie([percentages["female"], percentages["male"]], autopct="%.2f", labels=["Male", "Female"], shadow=True, explode=[0.02, 0.05])
pyplot.title("Percentage of male and female in the first class.")

pyplot.show()
first_class_survived = Pos_training_examples[Pos_training_examples.loc[:,"Pclass"] == 1]
percentages = {}
percentages["female"] = np.mean(first_class_survived.loc[:, "Sex"].map({'male':0, 'female':1}) == 1)*100 
percentages["male"] = np.mean(first_class_survived.loc[:, "Sex"].map({'male':0, 'female':1}) == 0)*100  

fig = pyplot.figure(figsize=(6, 6))
pyplot.pie([percentages["female"], percentages["male"]], autopct="%.2f", labels=["Male", "Female"], shadow=True, explode=[0.02, 0.05])
pyplot.title("Percentage of male and female in the first class.")

pyplot.show()
train_set.loc[:, "Sex"] = train_set.loc[:, "Sex"].map({"male":0, "female":1})
test_set.loc[:, "Sex"] = test_set.loc[:, "Sex"].map({"male":0, "female":1})
def get_names_importance(names):
    names = list(names)
    new_names = []
    
    important_names = np.random.uniform(10, 16, size=(1, 10000))
    not_important_names = np.random.uniform(1, 8, size=(1, 10000))
    
    for i in range(len(names)):
        if ('Mr' in names[i].split(sep=".") or 'Miss' in names[i].split(sep=".") or
            "Mrs" in names[i].split(sep=".") or "Mme" in names[i].split(sep=".")):
            
            new_names.append(float(important_names[:, i]))
        else:
            new_names.append(float(not_important_names[:, i]))
            
    return new_names        
train_set.loc[:, "Name"] = get_names_importance(train_set.loc[:, "Name"])
test_set.loc[:, "Name"] = get_names_importance(test_set.loc[:, "Name"])
train_set.loc[:, "Embarked"] = train_set.loc[:, "Embarked"].map({"S":0, "C":1, "Q":3})
test_set.loc[:, "Embarked"] = test_set.loc[:, "Embarked"].map({"S":0, "C":1, "Q":3})
def get_new_age_column(Age):
    new_Age = []
    
    Kids = np.random.uniform(0, 1, size=(1, 1000))
    Teens = np.random.uniform(2, 7, size=(1, 1000))
    Adults = np.random.uniform(9, 14, size=(1, 1000))
    Old = np.random.uniform(15, 16, size=(1, 1000))
    
    for i in range(len(Age)):
            if (Age[i]>=0 and Age[i]<=14):
                new_Age.append(float(Kids[:, i]))
            elif (Age[i]>=15 and Age[i]<=22):
                new_Age.append(float(Teens[:, i]))
            elif (Age[i]>=23 and Age[i]<=35):
                new_Age.append(float(Adults[:, i]))
            elif (Age[i]>=36 and Age[i]<=80):
                new_Age.append(float(Old[:, i])) 
            else:
                new_Age.append(0.0)
                
    new_Age = np.array(new_Age)            
    new_Age[new_Age == 0.0] = np.mean(new_Age)             
    return new_Age         
train_set.loc[:, "Age"] = get_new_age_column(train_set.loc[:, "Age"])
test_set.loc[:, "Age"] = get_new_age_column(test_set.loc[:, "Age"])
train_set = train_set.drop(["Cabin"], axis=1)
test_set  = test_set.drop(["Cabin"], axis=1)
train_set = train_set.drop(["Ticket"], axis=1)
test_set  = test_set.drop(["Ticket"], axis=1)
Parch = train_set.loc[:, "Parch"].values
SibSp = train_set.loc[:, "SibSp"].values

Family = Parch + SibSp
Hasfamily = np.array([1 if value >= 1 else 0 for value in Family])
train_set["Hasfamily"] = Hasfamily
train_set["Familysize"] = Family + 1
Parch = test_set.loc[:, "Parch"].values
SibSp = test_set.loc[:, "SibSp"].values

Family = Parch + SibSp
Hasfamily = np.array([1 if value >= 1 else 0 for value in Family])
test_set["Hasfamily"] = Hasfamily
test_set["Familysize"] = Family
def ticket_desc(Fare):
    output = []
    mean = sum(Fare) / len(Fare)
    
    for i in range(len(Fare)):
        if (Fare[i] > mean):
            output.append(2)
            
        elif (Fare[i] < mean):
            output.append(0)
            
        else:
            output.append(1)
        
    return output    
train_set["Ticket desc"] = ticket_desc(train_set.loc[:, "Fare"])
test_set["Ticket desc"] = ticket_desc(test_set.loc[:, "Fare"])
from sklearn.model_selection import train_test_split as Split

X_train, X_dev, Y_train, Y_dev = Split(train_set.iloc[:, 1:], train_set.iloc[:, 0], shuffle=True, random_state=1, test_size=0.2)
X_test = test_set.copy()
print("[Printing shapes to check]\n".center(40, " "))
print(f"The shape of X_train is: {X_train.shape}")
print(f"The shape of Y_train is: {Y_train.shape}")
print(f"The shape of X_dev is: {X_dev.shape}")
print(f"The shape of Y_dev is: {Y_dev.shape}")
from sklearn.impute import SimpleImputer
Cleaner = SimpleImputer()

X_test.iloc[:, :] = Cleaner.fit_transform(X_test)
X_train.iloc[:, :] = Cleaner.fit_transform(X_train)
X_dev.iloc[:, :] = Cleaner.fit_transform(X_dev)
X_train, X_test, X_dev = np.array(X_train), np.array(X_test), np.array(X_dev)
Y_train, Y_dev = np.array(Y_train), np.array(Y_dev)
Precision = lambda y_pred, y_true: Metrics_tools.precision_score(y_pred, y_true, average="macro")
Recall = lambda y_pred, y_true: Metrics_tools.recall_score(y_pred, y_true, average="macro")
def Feedforward_propagation(X, parameters, g):
    
    L = len(parameters) // 2 # Total number of layers
    
    A, Z = {"A0": X.T}, {}
    for l in range(1, L + 1):
        Z['Z' + str(l)] = np.dot(parameters["W" + str(l)], A["A" + str(l-1)]) + parameters["b" + str(l)]
        A['A' + str(l)] = g[str(l)](Z['Z' + str(l)])
        
    cache = (A, Z, g)
    
    return cache
def Backward_propagation(parameters, cache, Y):
    A, Z, g = cache
    L = len(parameters) // 2 # Total number of layers.
    
    
    g_der = g.copy()
    for activation in g.values():
        for l in range(L):
            if g[str(l+1)](-7777) == -1:
                g_der[str(l+1)] = lambda z: 1 -  np.square(np.tanh(z))
            elif g[str(l+1)](-7777) == 0:
                g_der[str(l+1)] = lambda z: g[str(l+1)](z)*(1 - g[str(l+1)](z))
            elif g[str(l+1)](777) == 777:
                g_der[str(l+1)] = lambda z: (z >= 0) + ((z < 0)*0)         
    

    m = A['A0'].shape[1]
    dZ, dW, db = {}, {}, {}
    
    i = 0
    for l in reversed(range(1, L+1)):
        if (l == L):
            dZ["dZ" + str(L)] = A['A' + str(L)] - Y
        else:
            dZ['dZ' + str(l)] = np.dot(parameters["W" + str(l+1)].T, dZ["dZ" + str(l+1)]) * g_der[str(l)](Z["Z"+str(l)])
        dW["dW" + str(l)] = 1./m * np.dot(dZ["dZ" + str(l)], A["A" + str(l-1)].T)
        db["db" + str(l)] = 1./m * np.sum(dZ["dZ" + str(l)], axis=1, keepdims=True)
        
    return dW, db
def Initialize_parameters(L, n):
    parameters = {}
    
    for l in range(1, L+1):
        parameters["W" + str(l)] = np.random.randn(n[l], n[l-1]) * 0.001
        parameters["b" + str(l)] = np.zeros((n[l], 1))
        
    return parameters    
def Binary_cross_entropy_loss(y_pred, y_true, parameters, lambd):
    Loss = Metrics_tools.log_loss(y_pred, y_true)  
    
    L = len(parameters) // 2
    
    Regularization = 0
    for l in range(L):
        Regularization += np.sum(np.sum(np.square(parameters["W"+str(l+1)][1:, :]), axis=1), axis=0)
    Regularization = Regularization * (lambd/(2*y_pred.shape[1]))
    
    return Loss + Regularization
def Update_parameters(parameters, learning_rate, dW, db, lambd):
    L = len(parameters) // 2 # total number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*(dW["dW" + str(l+1)]+lambd*parameters["W"+str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*db["db" + str(l+1)]
        
    return parameters    
def model(X_train, Y_train, learning_rate, n, g, print_cost, num_iters, L, lambd):
    Errors = []
    Precisions = []
    Recalls = []
    Y_train = Y_train[None, :]
    
    parameters = Initialize_parameters(L, n)
    for iteration in range(num_iters):
        
        # Forward propagation
        cache = Feedforward_propagation(X_train, parameters, g)
        # Backward propagation
        dW, db = Backward_propagation(parameters, cache, Y_train)
        # Calculating the cost or the error.
        Y_pred = cache[0]["A" + str(L)]
        Cost = Binary_cross_entropy_loss(Y_train, Y_pred, parameters, lambd)
        precision = Precision((Y_pred>=0.5).astype(int), Y_train)
        recall = Recall((Y_pred>=0.5).astype(int), Y_train)
        
        Errors.append(Cost)
        Precisions.append(precision)
        Recalls.append(recall)
        # Printing the cost
        if (print_cost and iteration % 50 == 0):
            print("\tIteration={}   , Cost=    {:.4f}".format(iteration, Cost))
            
        # Upating the parameters
        parameters = Update_parameters(parameters, learning_rate, dW, db, lambd)
                
    return Errors, Precisions, Recalls, parameters
tanh = lambda x: np.tanh(x)
sigmoid = lambda x: 1/(1+np.exp(-x))
ReLU = lambda x: np.maximum(0, x)
L = 2
n = [X_train.shape[1], 24, 1]
g = {
    "1":ReLU, "2":sigmoid
    
}
learning_rate = 0.002
num_iters = 2000
lambd = 0.0001
Errors, Precisions, Recalls, parameters = model(X_train, Y_train, learning_rate,
                                                n, g, True, num_iters, L, lambd)
def Predict(X, parameters, g):
    cache = Feedforward_propagation(X, parameters, g)
    
    L = len(parameters) // 2
    predictions = (cache[0]["A" + str(L)] >= 0.5).astype(int)
    
    return np.squeeze(predictions)
print("    DEVELOPMENT SET:")
Y_pred = Predict(X_dev, parameters, g)
print("\tPrecision : {:.5f}".format(Precision(Y_pred, Y_dev)))
print("\tRecall : {:.5f}\n".format(Recall(Y_pred, Y_dev)))

print("    TRAINING SET:")
Y_pred = Predict(X_train, parameters, g)
print("\tPrecision : {:.5f}".format(Precision(Y_pred, Y_train)))
print("\tRecall : {:.5f}".format(Recall(Y_pred, Y_train)))
fig, ax_array = pyplot.subplots(1, 3, figsize=(15, 4))
fig.subplots_adjust(wspace=0.07, hspace=0.025)

ax_array = ax_array.ravel()
y_axis = [Errors, Precisions, Recalls]
x_axis = range(num_iters)
x_label = 'Number of iterations'
y_labels = ["Cost function", "Precision", "Recall"]

for i, ax in enumerate(ax_array):
    ax.plot(x_axis, y_axis[i], linewidth=1.2, color="darkred", label=y_labels[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_labels[i])
    ax.legend(loc="upper right", fancybox=True, shadow=True, frameon=True, framealpha=0.8)
    ax.set_title(y_labels[i])
Y_pred = Predict(X_test, parameters, g)
submission = pd.DataFrame()
submission["PassengerId"] = PassengerId["test"]
submission["Survived"] = Y_pred
submission.to_csv("submission.csv", index=False)