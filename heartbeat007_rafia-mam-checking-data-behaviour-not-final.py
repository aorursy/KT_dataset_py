import numpy as np

import pandas as pd
df = pd.read_csv("../input/test.csv")
df
import matplotlib.pyplot as plt
## plotting configuration
###### plotting confiduration dont focus on that its just a  configuration

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 50

fig_size[1] = 20

plt.rcParams["figure.figsize"] = fig_size

plt.rcParams.update({'font.size': 50})
## we need 1 vs 1 multioutput classification

## package for multioutput classifier

from sklearn.multioutput import MultiOutputClassifier
## we deine the whole encoding a function to do it easily with other column

from sklearn.preprocessing import LabelEncoder

def encode(df):

    encoder = LabelEncoder()

    target=encoder.fit_transform(df)

    return np.array(target)


df.columns
target1 = encode(df['Rare_Tone'])
target2 = encode(df['Frequent_Tone '])
target2
target1
## save with a another name 

df['target1']  = target1

df['target2'] = target2

df.head()
### cleaning data

df.columns
df = df[['Sample', 'fp1', 'fp2', 'c3', 'c4', 'p7', 'p8', 'o1', 'o2', 'target1', 'target2']]
df.head()
## sample may not be a feature

df=df.drop('Sample',axis=1)
df.head()
df.hist()
df.mean()
df.describe()
df.corr().plot()
## find how the data is related to the target

df.corr()['target1']
df.corr()['target1'].plot()
## find how the data is related to the target

df.corr()['target2']
df.corr()['target2'].plot('bar')
df.corr()['target1'].plot('bar')
import seaborn as sns

df.corr()
sns.heatmap(df.corr())
df['fp1'].plot()
## you can remove the fp1 since all the value are same no impact
df.columns
df[['fp1','fp2', 'c3', 'c4', 'p7', 'p8', 'o1', 'o2']].plot()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold



## first we go individual then both
X = df[['fp1','fp2', 'c3', 'c4', 'p7', 'p8', 'o1', 'o2']]
Y = df[['target1','target2']]
Y.head()
X.head()
def normalize(df):

    return (df-df.min())/(df.max()-df.min())
X = normalize(X)
X = X.drop('fp1',axis=1)
X.head()



## y1 is our main target

y1 = df[['target1','target2']]



# this is a single target for plotting putpose

y2 = df[['target1']]
## Best upon 1 we predict the next one we cant do grid both all at once cause .

# each require different parameter set and we are doing one vs one prediction
x_train1,x_test1,y_train1,y_test1 = train_test_split(X,y1,test_size=.2)  ## for both

x_train,x_test,y_train,y_test = train_test_split(X,y2,test_size=.2)    ## for finding grid
y1.head()
y2.head()




from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier



k_range = list(range(1,31))

weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)

print (param_grid)

knn = KNeighborsClassifier()



grid = GridSearchCV(knn, param_grid, cv = 10,n_jobs=-1,verbose=2 ,scoring = 'accuracy')

grid.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix

knn_final = grid.best_estimator_  ## find the best estimator

tmp_pred_knn = knn_final.fit(x_train,y_train)  ## individual training for individual confusion matrix precision,recall,F1
pr1 = tmp_pred_knn.predict(x_test)



confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)



from sklearn.metrics import precision_score,recall_score,f1_score

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))

print(tmp_pred_knn.score(x_test,y_test))
sns.heatmap(confusion_matrix1)
from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

param_grid = {

 #   'bootstrap': [True],

 #   'max_depth': [80, 90, 100, 110],

 #   'max_features': [2, 3],

 #   'min_samples_leaf': [3, 4, 5],

 #   'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}



# Create a based model

rf = RandomForestClassifier()



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2) ## njobs force the computer to use all their resources
grid_search.fit(x_train, y_train)
### dont forget we are working with just 1 target but later we introduce two of them
rf_final=grid_search.best_estimator_   # find the best estimator

tmp_pred_rf = rf_final.fit(x_train,y_train)       ## individual training 

pr1 = tmp_pred_rf.predict(x_test)   #3individual prediction

confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)

sns.heatmap(confusion_matrix1)

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))
svc_final = SVC()

svc_final.fit(x_train,y_train)

pr1 = svc_final.predict(x_test)

confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)

sns.heatmap(confusion_matrix1)



plt.rcParams["figure.figsize"] = fig_size

plt.rcParams.update({'font.size': 50})

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))
param_grid = {'C':[1,10,100,1000]}

grid3 = GridSearchCV(LinearSVC(),param_grid,refit = True, verbose=2,n_jobs=-1)




grid3.fit(x_train,y_train)



ACP=[]

#plot_cv_results(grid3.cv_results_,'C',)

C = range(1,100)

for item in C:

    model = LinearSVC(C=item)

    model.fit(x_train,y_train)

    ACP.append(model.score(x_test,y_test))
plt.plot(C,ACP)
ACP
## for a single target thats very good
lsvc_final = grid3.best_estimator_



lsvc_final.fit(x_train,y_train)

pr1 = lsvc_final.predict(x_test)

confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)

sns.heatmap(confusion_matrix1)

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))
parameters = {'learning_rate': [0.1, 0.05, 0.02, 0.01],

              'max_depth': [4, 6, 8],

              #'min_samples_leaf': [20, 50,100,150],

              #'max_features': [1.0, 0.3, 0.1] 

              }



grid4 = GridSearchCV(GradientBoostingClassifier(), parameters,verbose=2, cv=10, n_jobs=-1)



grid4.fit(x_train, y_train)

print (grid4.best_estimator_)
gb_final = grid4.best_estimator_



gb_final.fit(x_train,y_train)

pr1 = gb_final.predict(x_test)

confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))

sns.heatmap(confusion_matrix1)
learning_rate1= [0.1, 0.05, 0.02, 0.01]

max_depth1= [4, 6, 8]

min_samples_leaf1= [20, 50,100,150]



ACP1=[]



## plotting based on the most imp charactisitcs learning_rate

for item in learning_rate1:

    model1 = GradientBoostingClassifier(learning_rate=item)

    model1.fit(x_train,y_train)

    ACP1.append(model1.score(x_test,y_test))

    print(item)




plt.plot(learning_rate1,ACP1)



ACP1
## thats very good
learning_rate1= [0.1, 0.05, 0.02, 0.01]

max_depth1= [4, 6, 8]

min_samples_leaf1= [20, 50,100,150]



ACP2=[]

for item in max_depth1:

    model2 = GradientBoostingClassifier(max_depth=item)

    model2.fit(x_train,y_train)

    ACP2.append(model2.score(x_test,y_test))

    print(item)

plt.plot(max_depth1,ACP2)
ACP2 ##depending on the max_depth




# Define the parameter values that should be searched

sample_split_range = list(range(2, 10))



# Create a parameter grid: map the parameter names to the values that should be searched

# Simply a python dictionary

# Key: parameter name

# Value: list of values that should be searched for that parameter

# Single key-value pair for param_grid

param_grid = dict(min_samples_split=sample_split_range)

dtc = DecisionTreeClassifier()

# instantiate the grid

grid5 = GridSearchCV(dtc, param_grid, cv=10,n_jobs=-1,verbose=2, scoring='accuracy')



# fit the grid with data

grid5.fit(x_train, y_train)







dt_final = grid5.best_estimator_

dt_final.fit(x_train,y_train)

pr1 = dt_final.predict(x_test)

confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)

sns.heatmap(confusion_matrix1)

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))



## its not very bad 
## now multioutput classifier




from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

ML=[]

M=['DecisionTreeClassifier','KNeighborsRegressor','SVC','LinearSVC','RandomForestRegressor','GradientBoostingClassifier']

Z=[gb_final,knn_final,svc_final,lsvc_final,rf_final,gb_final]
from sklearn.pipeline import Pipeline

from sklearn.multiclass import OneVsRestClassifier# Using pipeline for applying logistic regression and one vs rest classifier

from sklearn.multioutput import MultiOutputClassifier

for model in Z:

    

    ##new

    models=MultiOutputClassifier(model)

    ##new

    models.fit(x_train1,y_train1)      ## training the model this could take a little time

    accuracy=models.score(x_test1,y_test1)    ## comparing result with the test data set

    ML.append(accuracy) 
d={'Accuracy':ML,'Algorithm':M}

df1=pd.DataFrame(d)
df1
X_temp = df[['fp2', 'c3', 'c4', 'p7', 'p8']]

Y_temp = df[['o1', 'o2']]

Y_temp.head()
X_temp = normalize(X_temp)

X_temp.head()
x_train3,x_test3,y_train3,y_test3 = train_test_split(X_temp,Y_temp,test_size=.2)  ## for both
from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR,LinearSVR

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

ML=[]

M=['DecisionTreeRegressor','KNeighborsRegressor','SVR','LinearSVR','RandomForestRegressor','GradientBoostingRegressor']

dt = DecisionTreeRegressor()

knnr =KNeighborsRegressor()

svr = SVR()

lsvr = LinearSVR()

ran = RandomForestRegressor()

grb = GradientBoostingRegressor()

Z=[dt,knnr,svr,lsvr,ran,grb]
from sklearn.pipeline import Pipeline



from sklearn.multioutput import MultiOutputRegressor

for model in Z:

    

    ##new

    models=MultiOutputRegressor(model)

    ##new

    models.fit(x_train3,y_train3)      ## training the model this could take a little time

    accuracy=models.score(x_test3,y_test3)    ## comparing result with the test data set

    ML.append(accuracy) 
d={'Accuracy':ML,'Algorithm':M}

df2=pd.DataFrame(d)
df2
## linear SVC failed to predict other does very good




from sklearn.ensemble import VotingClassifier

voting_clf1 = MultiOutputClassifier(VotingClassifier(estimators=[('knn',knn_final),('rf',rf_final),('dt',dt_final),('svc',svc_final),('gb',gb_final)],voting='hard'))



voting_clf1.fit(x_train1,y_train1)
predicted=voting_clf1.predict(x_test1)
predicted
cf1_d=[]

cf2_d=[]

for item in predicted:

    cf1_d.append(item[0])

for item in predicted:

    cf2_d.append(item[1])
cf1_d
cf2_d
voting_hybrid_classifier_result=voting_clf1.score(x_test1,y_test1)
voting_hybrid_classifier_result
## very very good
from sklearn.ensemble import VotingRegressor



M=['DecisionTreeRegressor','KNeighborsRegressor','SVR','LinearSVR','RandomForestRegressor','GradientBoostingRegressor']

dt = DecisionTreeRegressor()

knnr =KNeighborsRegressor()

svr = SVR()

lsvr = LinearSVR()

ran = RandomForestRegressor()

grb = GradientBoostingRegressor()





voting_clf2 = MultiOutputRegressor(VotingRegressor(estimators=[('knn',knnr),('dt',dt),('svr',svr),('gbr',grb)]))

voting_clf2.fit(x_train3,y_train3)
voting_hybrid_regressor_result=voting_clf2.score(x_test3,y_test3)
voting_hybrid_regressor_result
## thats also good
from sklearn.model_selection import cross_val_score

## going for 3
print(cross_val_score(voting_clf1, x_test1, y_test1, cv=3))
## cv 3 ahows 96%
## going for 10
print(cross_val_score(voting_clf1, x_test1, y_test1, cv=10))
## for 10 cv it shows 97
print(cross_val_score(voting_clf2, x_test3, y_test3, cv=3))
print(cross_val_score(voting_clf2, x_test3, y_test3, cv=10))
# for cv 10 we got 94% thats very promising
plt.bar(df1["Algorithm"],df1["Accuracy"])
plt.bar(df2["Algorithm"],df2["Accuracy"])
## nural net starts here 

## we go from single to multiple check all of them
x_train1.head()
y_train1.head()
## we start from single classification then multiple classification

## then single classification and regression and with classification with functional API

## then Recurrent NURAL NET
#Dependencies

import keras

from keras.models import Sequential

from keras.layers import Dense# Neural network

n_col = x_train1.shape[1]  ## find th column number
import keras

from keras.models import Sequential

from keras.layers import Dense# Neural network

def model():

    model = Sequential()

    model.add(Dense(7,input_dim = n_col, activation="relu"))

    model.add(Dense(100, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(200, activation="relu"))

    model.add(Dense(2, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model1 = model()
model1.fit(x_train1,y_train1[['target1']],epochs=20)
model1.evaluate(x_test1,y_test1[['target1']])
## thats 98% accuracy for a single label classification
model2 = model()

model2.fit(x_train1,y_train1[['target2']],epochs=20)
model2.evaluate(x_test1,y_test1[['target2']])
predict1 = model2.predict(x_test1)

tmp1=[]

tmp2=[]

for item in np.array(y_test1[['target1']].astype("int")):

    tmp1.append(item[0])

for item in np.array(y_test1[['target1']]).astype("int"):

    tmp2.append(item[0])

print (confusion_matrix(tmp1,tmp2))

sns.heatmap(confusion_matrix(tmp1,tmp2))
## create regression for o1 and o2 

from keras.layers import Input

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Dropout
def neural_net(n_col):

    visible = Input(shape=(n_col,))

    hidden1 = Dense(100, activation='relu')(visible)

    hidden2 = Dense(200, activation='relu')(hidden1)

    dropout3 = Dropout(.2)(hidden2)

    hidden4 = Dense(200, activation='relu')(dropout3)

    dropout5 = Dropout(.2)(hidden4)

    hidden6 = Dense(200, activation='relu')(dropout5)

    dropout7 = Dropout(.2)(hidden6)

    hidden8 = Dense(200, activation='relu')(dropout7)

    dropout9 = Dropout(.2)(hidden8)

    hidden10 = Dense(200, activation='relu')(dropout9)

    dropout11 = Dropout(.2)(hidden10)

    hidden12 = Dense(200, activation='relu')(dropout11)

    dropout13 = Dropout(.2)(hidden12)

    hidden14 = Dense(200, activation='relu')(dropout13)

    dropout15 = Dropout(.2)(hidden14)

    hidden16 = Dense(200, activation='relu')(dropout15)

    hidden17 = Dense(100, activation='relu')(hidden16)

    hidden18 = Dense(100, activation='relu')(hidden17)



    output = Dense(2)(hidden18)

    model = Model(inputs=visible, outputs=output)

    model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])

    return model



model3 = neural_net(5)

xtrain3 = X.drop('o1',axis=1)
xtrain3 = xtrain3.drop('o2',axis=1)
x_train3.head()
y_train3 = X[['o1','o2']]
y_train3
model3.fit(xtrain3,y_train3,epochs=10)
## ths is train evaluation

model3.evaluate(xtrain3,y_train3)
import h2o
from h2o.automl import H2OAutoML
h2o.init()
## without normalization

hdf = h2o.H2OFrame(x_test1)
aml = H2OAutoML()
aml.train(y='o1',training_frame=hdf)
aml.leaderboard
aml2 = H2OAutoML()

aml2.train(y='o2',training_frame=hdf)

aml2.leaderboard
