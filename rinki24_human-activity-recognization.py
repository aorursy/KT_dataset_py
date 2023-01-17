import pandas as pd

test  = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/test.csv")  

train = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv") 
train.head()
train.Activity.value_counts()
train.shape,test.shape
test.Activity.value_counts()
# suffling data 

from sklearn.utils import shuffle



test  = shuffle(test)

train = shuffle(train)
# separating data inputs and output lables 

trainData  = train.drop('Activity' , axis=1).values

trainLabel = train.Activity.values



testData  = test.drop('Activity' , axis=1).values

testLabel = test.Activity.values
# encoding labels 

from sklearn import preprocessing



encoder = preprocessing.LabelEncoder()



# encoding test labels 

encoder.fit(testLabel)

testLabelE = encoder.transform(testLabel)



# encoding train labels 

encoder.fit(trainLabel)

trainLabelE = encoder.transform(trainLabel)
# target variable is categorical

# IV's are num
# classification models:

# Decision tree

# SVM

# NN

# RF

# GBM

# DNN

# ANN

# DNN
# applying supervised neural network using multi-layer preceptron 

import sklearn.neural_network as nn 

mlpSGD  =  nn.MLPClassifier(hidden_layer_sizes=(90,)  \

                        , max_iter=1000 , alpha=1e-4  \

                        , solver='sgd' , verbose=10   \

                        , tol=1e-19 , random_state=1  \

                        , learning_rate_init=.001)
mlpADAM =  nn.MLPClassifier(hidden_layer_sizes=(90,)  \

                        , max_iter=1000 , alpha=1e-4  \

                        , solver='adam' , verbose=10  \

                        , tol=1e-19 , random_state=1  \

                        , learning_rate_init=.001) 
mlpLBFGS =  nn.MLPClassifier(hidden_layer_sizes=(90,)  \

                        , max_iter=1000 , alpha=1e-4  \

                        , solver='lbfgs' , verbose=10  \

                        , tol=1e-19 , random_state=1  \

                        , learning_rate_init=.001) 
nnModelSGD  = mlpSGD.fit(trainData , trainLabelE)
nnModelSGD  = mlpLBFGS.fit(trainData , trainLabelE)

nnModelSGD
nnModelADAM = mlpADAM.fit(trainData , trainLabelE)
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sb

%matplotlib inline
# load data

#https://s3.amazonaws.com/hackerday.datascience/112/test.csv

#https://s3.amazonaws.com/hackerday.datascience/112/train.csv



train = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv")

test = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/test.csv")

print('Train Data', train.shape,'\n', train.columns)

print('\nTest Data', test.shape)
print('Train labels', train['Activity'].unique(), '\nTest Labels', test['Activity'].unique())
pd.crosstab(train.subject, train.Activity)
sub15 = train.loc[train['subject']==1]
sub15.head()
train.head()
train.subject.value_counts()
fig = plt.figure(figsize=(32,24))

ax1 = fig.add_subplot(221)

ax1 = sb.stripplot(x='Activity', y=sub15.iloc[:,0], data=sub15, jitter=True)

ax2 = fig.add_subplot(222)

ax2 = sb.stripplot(x='Activity', y=sub15.iloc[:,1], data=sub15, jitter=True)

plt.show()
fig = plt.figure(figsize=(32,24))

ax1 = fig.add_subplot(221)

ax1 = sb.stripplot(x='Activity', y=sub15.iloc[:,2], data=sub15, jitter=True)

ax2 = fig.add_subplot(222)

ax2 = sb.stripplot(x='Activity', y=sub15.iloc[:,3], data=sub15, jitter=True)

plt.show()
fig = plt.figure(figsize=(32,24))

ax1 = fig.add_subplot(221)

ax1 = sb.stripplot(x='Activity', y=sub15.iloc[:,4], data=sub15, jitter=True)

ax2 = fig.add_subplot(222)

ax2 = sb.stripplot(x='Activity', y=sub15.iloc[:,5], data=sub15, jitter=True)

plt.show()



# We need to check the spread as of activity column as it can have outliers which has to be

# taken care if we are using standard levels models like random forest and decision tree

# But for deep learning it will take care of outliers

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#https://s3.amazonaws.com/hackerday.datascience/112/test.csv

#https://s3.amazonaws.com/hackerday.datascience/112/train.csv

train_df = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/train.csv")

test_df = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/112/test.csv")

train_df.columns
unique_activities = train_df.Activity.unique()

print("NUmber of unique activities: {}".format(len(unique_activities)))

replacer = {}

for i, activity in enumerate(unique_activities):

    replacer[activity] = i

train_df.Activity = train_df.Activity.replace(replacer)

test_df.Activity = test_df.Activity.replace(replacer)

train_df.head(10)

train_df = train_df.drop("subject", axis=1)

test_df = test_df.drop("subject", axis=1)

def get_all_data():

    train_values = train_df.values

    test_values = test_df.values

    np.random.shuffle(train_values)

    np.random.shuffle(test_values)

    X_train = train_values[:, :-1]

    X_test = test_values[:, :-1]

    y_train = train_values[:, -1]

    y_test = test_values[:, -1]

    return X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = get_all_data()
model = LogisticRegression(C=10)#complexity parameter

model
model = LogisticRegression()

model
model.fit(X_train, y_train)

model.score(X_test, y_test)
#logistic regression : 87%
# Try some transformations

from sklearn.decomposition import PCA



X_train, X_test, y_train, y_test = get_all_data() #generating the training set

pca = PCA(n_components=200) # initializing the PCA

pca.fit(X_train) #applying PCA

X_train = pca.transform(X_train) # transforming the dataset

X_test = pca.transform(X_test)



model.fit(X_train, y_train) #creating model

model.score(X_test, y_test) #score

# Worse performance, but trains faster
# Scale features to be between -1 and 1

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



model.fit(X_train, y_train)

model.score(X_test, y_test)

# Better performance
# Neural network

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils.np_utils import to_categorical
X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
n_input = X_train.shape[1] # number of features

n_output = 6 # number of possible labels

n_samples = X_train.shape[0] # number of training samples

n_hidden_units = 40

Y_train = to_categorical(y_train)

Y_test = to_categorical(y_test)

print(Y_train.shape)

print(Y_test.shape)
def create_model():

    model = Sequential()

    model.add(Dense(n_hidden_units,

                    input_dim=n_input,

                    activation="relu"))

    model.add(Dense(n_hidden_units,

                    input_dim=n_input,

                    activation="relu"))

    model.add(Dense(n_output, activation="softmax"))

    # Compile model

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model
estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=10, verbose=False)

estimator.fit(X_train, Y_train)

print("Score: {}".format(estimator.score(X_test, Y_test)))

# accuracy 88.7%
from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



model = RandomForestClassifier(n_estimators=500)

model.fit(X_train, y_train)

model.score(X_test, y_test)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD,Adam
#Feature matrix

train_data = train.iloc[:,:561].as_matrix()

test_data = test.iloc[:,:561].as_matrix()



train_labels = train.iloc[:,562:].as_matrix()

test_labels = test.iloc[:,562:].as_matrix()



train_labelss=np.zeros((len(train_labels),6))

test_labelss=np.zeros((len(test_labels),6))
for k in range (0,len(train_labels)):

    if train_labels[k] =='STANDING':

        train_labelss[k][0]=1

    elif train_labels[k] =='WALKING':

        train_labelss[k][1]=1

    elif train_labels[k] =='WALKING_UPSTAIRS':

        train_labelss[k][2]=1

    elif train_labels[k] =='WALKING_DOWNSTAIRS':

        train_labelss[k][3]=1

    elif train_labels[k] =='SITTING':

        train_labelss[k][4]=1

    else:

        train_labelss[k][5]=1
for k in range (0,len(test_labels)):

    if test_labels[k] =='STANDING':

        test_labelss[k][0]=1

    elif test_labels[k] =='WALKING':

        test_labelss[k][1]=1

    elif test_labels[k] =='WALKING_UPSTAIRS':

        test_labelss[k][2]=1

    elif test_labels[k] =='WALKING_DOWNSTAIRS':

        test_labelss[k][3]=1

    elif test_labels[k] =='SITTING':

        test_labelss[k][4]=1

    else:

        test_labelss[k][5]=1
model = Sequential()

model.add(Dense(64, activation='relu', input_dim=561))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(train_data, train_labelss,nb_epoch=30,batch_size=128)

score = model.evaluate(test_data, test_labelss, batch_size=128)

print(score)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_data, train_labelss,nb_epoch=30,batch_size=128)

score = model.evaluate(test_data, test_labelss, batch_size=128)

print(score)
###### Random Forest #######

trainData  = train.drop('Activity' , axis=1).values

trainLabel = train.Activity.values



testData  = test.drop('Activity' , axis=1).values

testLabel = test.Activity.values



encoder = LabelEncoder()
# encoding test labels 

encoder.fit(testLabel)

testLabelEncoder = encoder.transform(testLabel)



# encoding train labels 

encoder.fit(trainLabel)

trainLabelEncoder = encoder.transform(trainLabel)



rf = RandomForestClassifier(n_estimators=200,  n_jobs=4, min_samples_leaf=10)    

#train

rf.fit(trainData, trainLabelEncoder)



y_te_pred = rf.predict(testData)



acc = accuracy_score(testLabelEncoder, y_te_pred)

print("Random Forest Accuracy: %.5f" % (acc))

##### K-Nearest Neighbors ######

clf = KNeighborsClassifier(n_neighbors=24)



knnModel = clf.fit(trainData , trainLabelEncoder)

y_te_pred = clf.predict(testData)



acc = accuracy_score(testLabelEncoder, y_te_pred)

print("K-Nearest Neighbors Accuracy: %.5f" % (acc))
import numpy as np

import pandas as pd

import time
print("Number of features in Train : ", train.shape[1])

print("Number of records  in Train : ",train.shape[0])

print("Number of features in Test  : ",test.shape[1])

print("Number of records  in Test  : ",test.shape[0])



trainData  = train.drop(['subject','Activity'] , axis=1).values

trainLabel = train.Activity.values



testData  = test.drop(['subject','Activity'] , axis=1).values

testLabel = test.Activity.values



print("Train Data shape  : ",trainData.shape)

print("Train Label shape : ",trainLabel.shape)

print("Test Data  shape  : ",testData.shape)

print("Test Label shape  : ",testLabel.shape)



print("Label examples: ")

print(np.unique(trainLabel))
from sklearn import preprocessing

from sklearn import utils



ltrain = preprocessing.LabelEncoder()

ltest = preprocessing.LabelEncoder()



trainLabel = ltrain.fit_transform(trainLabel)

testLabel  = ltest.fit_transform(testLabel)



print(np.unique(trainLabel))

print(np.unique(testLabel))

print("Train Label shape : ",trainLabel.shape)

print("Test Label shape  : ",testLabel.shape)

print(utils.multiclass.type_of_target(testLabel))
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.utils import shuffle
t0 = time.clock()

# Create the RFE object and compute a cross-validated score.

svc = SVC(kernel="linear")

# The "accuracy" scoring is proportional to the number of correct

# classifications

rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),

              scoring='accuracy')

# Before training the data it is convenient to shuffle the data in training

np.random.seed(1)

print("Labels before Shuffle",testLabel[0:5])

testData,testLabel = shuffle(testData,testLabel)

trainData,trainLabel = shuffle(trainData,trainLabel)

print("Labels after Shuffle",testLabel[0:5])
# train and fit data in the model

rfecv.fit(trainData, trainLabel)



print("Optimal number of features : %d" % rfecv.n_features_)

print("Processing time sec ",time.clock() - t0)

# Plot number of features VS. cross-validation scores

plt.figure(figsize=(32,12))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
print('Accuracy of the SVM model on test data is ', rfecv.score(testData,testLabel) )

print('Ranking of features starting from the best estimated \n',rfecv.ranking_)

# if we mask the features to get only the best we get this

best_features = []

for ix,val in enumerate(rfecv.support_):

    if val==True:

        best_features.append(testData[:,ix])
from pandas.tools.plotting import scatter_matrix

visualize = pd.DataFrame(np.asarray(best_features).T)

print(visualize.shape)

scatter_matrix(visualize.iloc[:,0:5], alpha=0.2, figsize=(16, 16), diagonal='kde')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.preprocessing import LabelBinarizer
train = shuffle(train)

test = shuffle(test)
train_features = train.iloc[:,:562].as_matrix()

test_features = test.iloc[:,:562].as_matrix()
binarizer = LabelBinarizer().fit(train['Activity'])

train_labels = binarizer.transform(train.Activity)

test_labels = binarizer.transform(test.Activity)
def weight_variable(shape):

    

    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))



def bias_variable(shape):

    

    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))



def add_layer(inputs, input_size, output_size, activation=None):

    

    W = weight_variable([input_size, output_size])

    b = bias_variable([output_size])

    wxb = tf.matmul(inputs, W) + b

    if activation:

        

        return activation(wxb)

    

    else:

        

        return wxb
X = tf.placeholder(tf.float32, [None, 562])

layer1 = add_layer(X, 562, 1000, tf.nn.relu)

layer2 = add_layer(layer1, 1000, 300, tf.nn.relu)

layer3 = add_layer(layer2, 300, 50, tf.nn.relu)

output = add_layer(layer3, 50, 6)



y_ = tf.placeholder(tf.float32, [None, 6])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=y_))

optimizer = tf.train.GradientDescentOptimizer(0.001)

train_step = optimizer.minimize(loss)



correct = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))

score = tf.reduce_mean(tf.cast(correct, "float"))
sess = tf.Session()

init = tf.initialize_all_variables()

sess.run(init)
for i in range(10000):

    

    batch = np.random.choice(train_features.shape[0], 100)

    _, cost = sess.run([train_step, loss],  feed_dict = {X:train_features[batch], y_:train_labels[batch]})
print(sess.run(score, feed_dict={X: test_features, y_: test_labels}))
#seperating class label from the dataset



trainLabels= train.Activity.values

trainData=train.drop("Activity",axis=1).values



testLabels= test.Activity.values

testData=test.drop("Activity",axis=1).values



print("Class labels striped off the dataset")
#transforming non-numerical labels to numerical labels using sklearn.preprocessing.LabelEncoder



from sklearn import preprocessing

labelEncoder= preprocessing.LabelEncoder()



labelEncoder.fit(trainLabels)

trainLabelsE=labelEncoder.transform(trainLabels)



labelEncoder.fit(testLabels)

testLabelsE=labelEncoder.transform(testLabels)



print("Labels Transformed and Encoded")
#applying k-nearest neighbours

from sklearn.neighbors import KNeighborsClassifier as knn

import numpy as np



knnScoreDistance=np.zeros(51)

knnScoreUniform=np.zeros(51)



for num in range(5,51):

    knnclf = knn(n_neighbors=num, weights='distance')

    knnModel = knnclf.fit(trainData , trainLabelsE)

    knnScoreDistance[num]=knnModel.score(testData  , testLabelsE )

    print("Testing  set score for KNN_Distance(k=%d): %f" %(num,knnScoreDistance[num]))

    

for num in range(5,51):

    knnclf = knn(n_neighbors=num, weights='uniform')

    knnModel = knnclf.fit(trainData , trainLabelsE)

    knnScoreUniform[num]=knnModel.score(testData  , testLabelsE )

    print("Testing  set score for KNN_Uniform(k=%d): %f" %(num,knnScoreUniform[num]))
import matplotlib.pyplot as plt



x=np.array(range(5,51))



plt.plot(x,knnScoreDistance[5:])

plt.plot(x,knnScoreUniform[5:])

plt.xlabel("No of neighbors (K)")

plt.ylabel("Test Data Mean Accuracy")

plt.legend(['KNN_Distance','KNN_Uniform'])

plt.show
import itertools

import numpy as np

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix



decsnTreeClf= DecisionTreeClassifier(criterion='entropy')

tree=decsnTreeClf.fit(trainData,trainLabelsE)

testPred=tree.predict(testData)



acc= accuracy_score(testLabelsE,testPred)

cfs = confusion_matrix(testLabelsE, testPred)



print("Accuracy: %f" %acc)



plt.figure()

class_names = labelEncoder.classes_

plot_confusion_matrix(cfs, classes=class_names,

                      title='DecisionTree Confusion Matrix, without normalization')
decsnTreeClf= DecisionTreeClassifier()

tree=decsnTreeClf.fit(trainData,trainLabelsE)

testPred=tree.predict(testData)



acc= accuracy_score(testLabelsE,testPred)

cfs = confusion_matrix(testLabelsE, testPred)



print("Accuracy: %f" %acc)



plt.figure()

class_names = labelEncoder.classes_

plot_confusion_matrix(cfs, classes=class_names,

                      title='DecisionTree Confusion Matrix, without normalization')
train_df.shape
X, y = train_df.iloc[:, 0:len(train_df.columns) - 1], train_df.iloc[:, -1]
X_test, y_test = test_df.iloc[:, 0:len(test_df.columns) -1], test_df.iloc[:, -1]
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score # for evaluation
classifiers = [

    DecisionTreeClassifier(),

    KNeighborsClassifier(7), # because there are 6 different labels

    SVC(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]
names = []

scores = []



for clf in classifiers:

    clf = clf.fit(X, y)

    y_pred = clf.predict(X_test)

    

    names.append(clf.__class__.__name__)

    scores.append(accuracy_score(y_pred, y_test))



score_df = pd.DataFrame({'Model': names, 'Score': scores}).set_index('Model')

score_df
import matplotlib.pyplot as plt

%matplotlib inline



ax = score_df.plot.bar()

ax.set_xticklabels(score_df.index, rotation=45, fontsize=10)
from sklearn.model_selection import GridSearchCV



parameters = {

    'kernel': ['linear', 'rbf', 'poly','sigmoid'],

    'C': [100, 50, 20, 1, 0.1]

}



selector = GridSearchCV(SVC(), parameters, scoring='accuracy') # we only care about accuracy here

selector.fit(X, y)



print('Best parameter set found:')

print(selector.best_params_)

print('Detailed grid scores:')

means = selector.cv_results_['mean_test_score']

stds = selector.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, selector.cv_results_['params']):

    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

    print()
clf = SVC(kernel='linear', C=100).fit(X, y)

y_pred = clf.predict(X_test)

print('Accuracy score:', accuracy_score(y_test, y_pred))