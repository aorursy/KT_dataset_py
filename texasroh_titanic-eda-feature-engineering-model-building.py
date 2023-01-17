import warnings

warnings.filterwarnings('ignore')



import pandas as pd



train_data = pd.read_csv('../input/titanic/train.csv')

print(train_data.shape)

train_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

print(test_data.shape)

test_data.head()
data = pd.concat([train_data, test_data], sort=False)

print(data.shape)

data.head()
data.info()
import matplotlib.pyplot as plt

%matplotlib inline



null_check = (data.isnull().sum() / len(data)).sort_values(ascending = False)



plt.bar(null_check.index, null_check.values)

plt.xticks(rotation = 45)

plt.ylabel("% of total row")

plt.title("Missing value percentage")

data.isnull().sum().sort_values(ascending = False)
data = data.drop('Cabin', axis=1)

data['Age'].fillna(data['Age'].median(), inplace=True)

data['Fare'].fillna(data['Fare'].median(), inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data.isnull().sum()
data = data.drop(['Ticket'], axis=1)

data.head()
data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0].unique()
data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
tmp = data['Title'].value_counts()

plt.bar(tmp.index, tmp)

plt.xlabel('Title')

plt.ylabel('# of Counts')

plt.xticks(rotation=90)

tmp
minimum_count = 10

title_filter = data['Title'].value_counts() < minimum_count

data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_filter.loc[x] ==True else x)

tmp = data['Title'].value_counts()

plt.bar(tmp.index, tmp)

plt.xlabel('Title')

plt.ylabel('Counts')

plt.xticks(rotation=90)

tmp
data = data.drop('Name', axis=1)

data.head()
data['Isalone'] = data['SibSp'] + data['Parch']

data['Isalone'] = data['Isalone'].apply(lambda x: 1 if x ==0 else 0)

data.head()
num_col = ['Age','SibSp','Parch','Fare']

cat_col = ['Pclass','Sex','Embarked','Title','Isalone']

fix, ax = plt.subplots(3,3, figsize=(15,15))

axi = ax.flat

idx = 0



for col in num_col:

    axi[idx].set_title(col)

    axi[idx].hist(data[col])

    idx += 1

    

for col in cat_col:

    axi[idx].set_title(col)

    tmp = data[col].value_counts()

    axi[idx].bar(tmp.index, tmp)

    idx += 1
cleaned_train_data = data.iloc[:891].copy()

cleaned_test_data = data.iloc[891:].copy()
import numpy as np



num_col = ['Age','SibSp','Parch','Fare']

cat_col = ['Pclass','Sex','Embarked','Title','Isalone']



fix, ax = plt.subplots(3,3, figsize=(15,15))

axi = ax.flat

idx = 0



for col in num_col:

    tmp = []

    for i in [1,0]:

        data_filter = cleaned_train_data['Survived'] == i

        tmp.append(cleaned_train_data.loc[data_filter,col])

    axi[idx].hist(tmp, stacked=True, label=['Survived','Dead'])

    axi[idx].set_title(col)

    axi[idx].legend()

    idx += 1



for col in cat_col:

    

    survived = cleaned_train_data.groupby(col).Survived.sum().reset_index(name="Survived")

    total = cleaned_train_data.groupby(col).size().reset_index(name="Total")

    

    x = np.arange(len(total))

    width = 0.35

    

#     axi[idx].bar(x-width/2, survived['Survived'], width, label='Survived')

#     axi[idx].bar(x+width/2, total['Total'], width, label='Total')



    axi[idx].bar(x, survived['Survived'], width, label='Survived')

    axi[idx].bar(x, total['Total']-survived['Survived'], width, bottom=survived['Survived'], label='Dead')

    

    axi[idx].legend()

    axi[idx].set_title(col)

    plt.sca(axi[idx])

    plt.xticks(x, total[col].values)

    

    idx += 1
# mapping male => 1, femail => 0

data.Sex = data.Sex.map({"male":1, "female":0})

data.head()
# get dummies in columns 'Pclass', 'Embarked' since it is categorical variable

data = pd.get_dummies(data, columns=['Pclass', 'Embarked','Title'])



# set Pclass_3 and Embarked_S as base (Avoid Heteroskedasticity --comply with Homoskedasticity)

# drop these two columns

print(data.shape)

data = data.drop(['Pclass_3', 'Embarked_S','Title_Mr'], axis = 1)



print(data.shape)

data.head()
# normalize numerical features

num_col = ['Age','SibSp','Parch','Fare']

for col in num_col:

    data[col] = (data[col] - data[col].mean())/data[col].max()
# Divide dataset into train and test

train_dataset = data[:891]

test_dataset = data[891:]   # this is for submission
X = train_dataset.drop(['Survived','PassengerId'], axis=1)

y = train_dataset['Survived']
# We need to divide train_dataset into train, validation set



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



LR_model = LogisticRegression()

LR_model.fit(X_train, y_train)

y_test_hat = LR_model.predict(X_test)

y_train_hat = LR_model.predict(X_train)



print("Train Accuracy: ",accuracy_score(y_train, y_train_hat))

print("Test Accuracy: ",accuracy_score(y_test, y_test_hat))
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits = 5, random_state = 0, shuffle=True)

LR_accuracy = cross_val_score(LR_model, X, y, cv=cv, scoring='accuracy')

print("Mean Accuracy Score for LR: {}".format(np.mean(LR_accuracy)))
from sklearn.naive_bayes import GaussianNB



NB_model = GaussianNB()

NB_model.fit(X_train, y_train)

y_train_hat = NB_model.predict(X_train)

y_test_hat = NB_model.predict(X_test)



print("Train Accuracy: ",accuracy_score(y_train, y_train_hat))

print("Test Accuracy: ",accuracy_score(y_test, y_test_hat))
from sklearn.model_selection import cross_val_score, StratifiedKFold



cv = StratifiedKFold(n_splits = 5, random_state = 0, shuffle=True)

NB_accuracy = cross_val_score(NB_model, X, y, cv=cv, scoring='accuracy')

print("Mean Accuracy Score for LR: {}".format(np.mean(NB_accuracy)))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV



model = KNeighborsClassifier()



param_grid = {'n_neighbors': range(1,20)}

cv = StratifiedKFold(n_splits = 5, random_state = 0, shuffle=True)

grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', return_train_score=True)

grid.fit(X,y)



print("Best Parameter: {}".format(grid.best_params_))

print("Best Cross Validation Score: {}".format(grid.best_score_))



KN_model = grid.best_estimator_
from sklearn.svm import SVC



model = SVC()

param_grid={'C': [.1, 1, 5, 10, 50],

            'gamma': [0.0001, 0.0005, 0.001, 0.005]}

cv = StratifiedKFold(n_splits = 5, random_state=0, shuffle=True)

grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')

grid.fit(X,y)



print("Best param: {}".format(grid.best_params_))

print("Best accuracy: {}".format(grid.best_score_))



SVC_model = grid.best_estimator_
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

param_grid = {"max_depth": range(1,21)}

cv = StratifiedKFold(n_splits = 5, random_state = 0, shuffle=True)

grid = GridSearchCV(model, param_grid, cv=cv, return_train_score=True, scoring='accuracy')

grid.fit(X,y)



print('Best Parameter: {}'.format(grid.best_params_))

print('Best Cross Validation Score: {}'.format(grid.best_score_))



DT_model = grid.best_estimator_
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

param_grid={'n_estimators':[50,100,150,200,250,300],

            'random_state': [0],

            'max_depth': range(1,21)}

cv = StratifiedKFold(n_splits=5, random_state =0, shuffle=True)

grid = GridSearchCV(model, param_grid, cv=cv, return_train_score=True, scoring='accuracy')

grid.fit(X,y)



print('Best Parameter: {}'.format(grid.best_params_))

print('Best Cross Validation Score: {}'.format(grid.best_score_))



RF_model = grid.best_estimator_
# make progressbar

from IPython.display import clear_output



def update_progress(progress):

    bar_length = 20

    if isinstance(progress, int):

        progress = float(progress)

    if not isinstance(progress, float):

        progress = 0

    if progress < 0:

        progress = 0

    if progress >= 1:

        progress = 1



    block = int(round(bar_length * progress))



    clear_output(wait = True)

    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)

    print(text)
# SKlearn neural network (MLP Classifier)

import warnings

warnings.filterwarnings('ignore')





from sklearn.neural_network import MLPClassifier



hidden_layers = []



for i in range(10,102,10):

    hidden_layers.append([i])

    

for i in range(1, 21):

    for j in range(1,21):

        hidden_layers.append([i,j])

        

best_acc = 0

        

for idx, hidden_layer in enumerate(hidden_layers):

    MLP_model = MLPClassifier(random_state = 0, hidden_layer_sizes=hidden_layer)

    MLP_model.fit(X_train, y_train)

    

    y_train_hat = MLP_model.predict(X_train)

    y_test_hat = MLP_model.predict(X_test)

    

    train_acc = accuracy_score(y_train, y_train_hat)

    test_acc = accuracy_score(y_test, y_test_hat)

    

    if test_acc > best_acc:

        best_train = train_acc

        best_acc = test_acc

        best_layer = hidden_layer

    

    update_progress((idx+1)/len(hidden_layers))



print("Best train: ", best_train)

print("Best acc: ", best_acc)

print("Best hidden layer: ", best_layer)
from sklearn.neural_network import MLPClassifier



hidden_layers = []

for i in range(10,201,10):

    hidden_layers.append([i])



for i in range(1,21):

    for j in range(1,21):

        hidden_layers.append([i,j])

        

model = MLPClassifier()

param_grid = {'random_state':[0],

              'hidden_layer_sizes':hidden_layers}

cv = StratifiedKFold(n_splits = 5, random_state = 0, shuffle=True)

grid = GridSearchCV(model, param_grid, cv=cv, return_train_score=True, scoring='accuracy')

grid.fit(X,y)



print('Best Parameter: {}'.format(grid.best_params_))

print('Best Cross Validation Score: {}'.format(grid.best_score_))



NN_model = grid.best_estimator_
test_dataset
test_X = test_dataset.iloc[:,2:]

test_y = test_dataset.iloc[:, 1:2]

submit_csv = test_dataset.iloc[:, :2]
## Logistic Regression

y_hat = LR_model.predict(test_X)

submit_csv.iloc[:,1] = np.array(y_hat, dtype=int)

submit_csv.to_csv('Logistic_ver.csv', index=False)
## SVC

y_hat = SVC_model.predict(test_X)

submit_csv.iloc[:,1] = np.array(y_hat, dtype=int)

submit_csv.to_csv("SVC_ver.csv", index=False)
## KNN

y_hat = KN_model.predict(test_X)

submit_csv.iloc[:, 1] = np.array(y_hat, dtype=int)

submit_csv.to_csv("KNN_ver.csv", index=False)
## Decision Tree

y_hat = DT_model.predict(test_X)

submit_csv.iloc[:, 1] = np.array(y_hat, dtype=int)

submit_csv.to_csv("DecisionTree_ver.csv", index=False)
## RandomForest

y_hat = RF_model.predict(test_X)

submit_csv.iloc[:,1] = np.array(y_hat, dtype=int)

submit_csv.to_csv("RandomForest_ver.csv", index=False)
## Neural Network - MLP

y_hat = NN_model.predict(test_X)

submit_csv.iloc[:,1] = np.array(y_hat, dtype = int)

submit_csv.to_csv("NeuralNetwork_ver.csv", index=False)
import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf_X_train = np.array(X_train)

tf_X_test = np.array(X_test)



tf_y_train = np.array(y_train).reshape(-1,1)

tf_y_test = np.array(y_test).reshape(-1,1)
X_train
tf.reset_default_graph()



X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])

Y = tf.placeholder(tf.float32, shape=[None, 1])



num_t = 20



W1 = tf.Variable(tf.random_normal([X_train.shape[1], num_t]), name='weight1')

b1 = tf.Variable(tf.random_normal([num_t]), name='bias1')

layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)



# W2 = tf.Variable(tf.random_normal([20,10]), name = 'weight2')

# b2 = tf.Variable(tf.random_normal([10]), name = 'bias2')

# layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)



W3 = tf.Variable(tf.random_normal([num_t,1]), name = 'weight3')

b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')



hypothesis = tf.sigmoid(tf.matmul(layer1, W3) + b3)



cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

train = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)

# train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)



predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    

    feed_train = {X:tf_X_train, Y:tf_y_train}

    feed_test = {X:tf_X_test, Y:tf_y_test}

    for step in range(10001):

#         print(step, sess.run([hypothesis,cost,W], feed_dict=feed))

        sess.run(train, feed_dict=feed_train)

#         print(step, sess.run(W, feed_dict=feed))

        if step % 2000 ==0:

            print(step, sess.run(cost, feed_dict=feed_train))

            

    train_acc = sess.run(accuracy, feed_dict = feed_train)

    test_acc = sess.run(accuracy, feed_dict = feed_test)

    print("Train Accuracy: ", train_acc)

    print("Test Accuracy: ", test_acc)

    y_hat = sess.run(predicted, feed_dict={X:test_X})

    

submit_csv.iloc[:,1] = np.array(y_hat, dtype = int)

submit_csv.to_csv("tensorflow_ver.csv", index=False)