# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
from datetime import datetime



from sklearn.metrics import accuracy_score,precision_score,roc_curve,roc_auc_score,classification_report,confusion_matrix

from sklearn.model_selection import GridSearchCV,KFold,train_test_split,learning_curve,cross_val_score,RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler

import seaborn as se

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



import keras

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings('ignore')



from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import LogisticRegression
data = pd.read_csv("/kaggle/input/voicegender/voice.csv") #
data.head()
print("The shape of the DataFrame {}".format(data.shape))
data.info()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(data)
M = data[(data['label'] == 'male')]

B = data[(data['label'] == 'female')]

trace = go.Bar(x = (len(M), len(B)), y = ['male','female'], orientation = 'h', opacity = 0.8, marker=dict(

        color=['red','green'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Count of Target variable')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
sns.pairplot(data)
g = sns.distplot(data['skew'])

g.set_title("Skewness Distribuition", fontsize=18)

g.set_xlabel("")

g.set_ylabel("Probability", fontsize=12)
sns.distplot(data['dfrange'],bins=30)
# Coorelation analysis

plt.figure(figsize=(15,10))

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
data.head()
plt.plot(data['meanfreq'])

plt.xlabel('MeanFreqency')

plt.ylabel('Range')

plt.show()
trace = go.Scatter(x=data['meanfreq'], y=data['sd'],

                    mode='markers',

                    name='markers')



layout = dict(title =  'Plot b/w meanfreq & sd')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
trace = go.Scatter(x=data['minfun'], y=data['maxfun'],

                    mode='markers',

                    name='markers')



layout = dict(title =  'Plot b/w Minfun & Maxfun')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
trace = go.Scatter(x=data['skew'], y=data['kurt'],

                    mode='markers',

                    name='markers')



layout = dict(title =  'Plot b/w skewness & kurtosis')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
trace = go.Scatter(x=data['mindom'], y=data['maxdom'],

                    mode='markers',

                    name='markers')



layout = dict(title =  'Plot b/w mindom & maxdom')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
data.label = data.label.apply(lambda x: 1 if x == 'male' else 0)

X = data.drop('label',1)

y = data.label



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
clfs = []

seed = 3



clfs.append(("LogReg", 

             Pipeline([("Scaler", StandardScaler()),

                       ("LogReg", LogisticRegression())])))



clfs.append(("XGBClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("XGB", XGBClassifier())]))) 

clfs.append(("KNN", 

             Pipeline([("Scaler", StandardScaler()),

                       ("KNN", KNeighborsClassifier())]))) 



clfs.append(("DecisionTreeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("DecisionTrees", DecisionTreeClassifier())]))) 



clfs.append(("RandomForestClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RandomForest", RandomForestClassifier())]))) 



clfs.append(("GradientBoostingClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, 

                                                                       n_estimators=600))]))) 



clfs.append(("RidgeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RidgeClassifier", RidgeClassifier())])))





clfs.append(("ExtraTreesClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreeClassifier())])))



scoring = 'accuracy'

n_folds = 10

msgs = []

results, names  = [], [] 



for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, 

                                 cv=kfold, scoring=scoring, n_jobs=-1)    

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  

                               cv_results.std())

    msgs.append(msg)

    print(msg)
# creating a model

model = RandomForestClassifier()



# feeding the training set into the model

model.fit(X_train, y_train)



# predicting the test set results

y_pred = model.predict(X_test)



# Calculating the accuracies

print("Training accuracy :", model.score(X_train, y_train))

print("Testing accuarcy :", model.score(X_test, y_test))



# classification report

cr = classification_report(y_test, y_pred)

print(cr)



# confusion matrix 

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

se.heatmap(cm, annot = True, cmap = 'winter')

plt.title('Confusion Matrix', fontsize = 20)

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

# Show metrics 

def show_metrics():

    tp = cm[1,1]

    fn = cm[1,0]

    fp = cm[0,1]

    tn = cm[0,0]

    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))

    print('Precision =     {:.3f}'.format(tp/(tp+fp)))

    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))

    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/

                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))
def plot_precision_recall():

    plt.step(recall, precision, color = 'b', alpha = 0.2,

             where = 'post')

    plt.fill_between(recall, precision, step ='post', alpha = 0.2,

                 color = 'b')



    plt.plot(recall, precision, linewidth=2)

    plt.xlim([0.0,1])

    plt.ylim([0.0,1.05])

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('Precision Recall Curve')

    plt.show();
def plot_roc():

    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)

    plt.plot([0,1],[0,1], 'k--', linewidth = 2)

   # plt.xlim([0.0,0.001])

   # plt.ylim([0.0,1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show();
def cross_val_metrics(model) :

    scores = ['accuracy', 'precision', 'recall']

    for sc in scores:

        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)

        print('[%s] : %0.5f (+/- %0.5f)'%(sc, scores.mean(), scores.std()))
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

        

        

rf_cfl = RandomForestClassifier(n_jobs = -1)



# Number of trees in random forest

n_estimators = [100,200,300,400]

# Number of features to consider at every split

max_features = ['auto', 'log2']

# Maximum number of levels in tree

max_depth = [10,20,30,40,50]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

params = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



folds = 5

param_comb = 10



random_search = RandomizedSearchCV(rf_cfl,param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=5, verbose=3, random_state=42)



start_time = timer(None) 

random_search.fit(X, y)

print(random_search.best_params_)

timer(start_time)
random_clf = RandomForestClassifier(n_estimators = 100,min_samples_split=5,min_samples_leaf=1,max_features='auto',max_depth=50,bootstrap=True)



random_clf.fit(X_train,y_train)

y_pred = random_clf.predict(X_test)

train_pred = random_clf.predict(X_train)



# Confusion maxtrix & metrics

cm = confusion_matrix(y_test, y_pred)

class_names = [0,1]
show_metrics()
cross_val_metrics(random_clf)
# ROC curve

fpr, tpr, t = roc_curve(y_test, y_pred)

plot_roc()
model = Sequential()

model.add(Dense(64,input_dim = 20,activation='relu'))

model.add(Dense(32,activation='relu',init = 'uniform'))

model.add(Dense(16,activation='relu',init = 'uniform'))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

history=model.fit(X_train,y_train ,epochs=100,batch_size=128, validation_data=(X_test,y_test))
# evaluate the keras model

_, accuracy_train = model.evaluate(X_train, y_train)

_, accuracy = model.evaluate(X_test, y_test)

print('Accuracy: %.2f' % (accuracy*100))
from matplotlib import pyplot

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()