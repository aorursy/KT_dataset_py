

## data manupulation

import numpy as np

import pandas as pd



## label encoder

from sklearn.preprocessing import LabelEncoder



## for plotting

import matplotlib.pyplot as plt







###### plotting confiduration dont focus on that its just a  configuration

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 50

fig_size[1] = 20

plt.rcParams["figure.figsize"] = fig_size

plt.rcParams.update({'font.size': 50})


## read the data

df = pd.read_csv('../input/data.csv')
df.head()



## init the encoder

encoder = LabelEncoder()
#3 encode the target

target=encoder.fit_transform(df['class'])

## assign the encoded target toa  new column

df['num_class'] = np.array(target)

## we need 1 vs 1 multioutput classification

## package for multioutput classifier

from sklearn.multioutput import MultiOutputClassifier

## we deine the whole encoding a function to do it easily with other column

def encode(df):

    encoder = LabelEncoder()

    target=encoder.fit_transform(df)

    return np.array(target)
## encode the protocol

num_proto = encode(df['protocol_type'])
## save with a another name 

df['num_proto']  = num_proto

service_num = encode(df['service'])

df['service_num'] = service_num

flag_num = encode(df['flag'])

df['flag_num'] = flag_num
df.head()
df.corr()['num_proto'].plot()

df_working = df[['duration','dst_bytes','wrong_fragment','num_failed_logins','logged_in','num_compromised','su_attempted','num_root','num_file_creations','num_shells','num_access_files','is_guest_login','srv_count','same_srv_rate','srv_diff_host_rate','dst_host_same_srv_rate','num_proto','flag_num','num_class']]
## working dataset .the data set we work with

df_working.head()
df_working.corr().plot()
df_working.corr()['num_class']


##dropping my two target

X = df_working.drop(['num_class'], axis=1)

X = X.drop(['num_proto'], axis=1)





## y1 is our main target

y1 = df_working[['num_class','num_proto']]



# this is a single target for plotting putpose

y2 = df_working[['num_class']]





## two working label in necessayy because we want to find the first best param | independently

## why we take another one i mean y2 cause we cant do grid on basis of both er have to choose 1 for grid

X.head()
y1.head() ## two target at once 
y2.head()  ## single first target
##
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold
x_train1,x_test1,y_train1,y_test1 = train_test_split(X,y1,test_size=.2)  ## for both

x_train,x_test,y_train,y_test = train_test_split(X,y2,test_size=.2)    ## for finding grid

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
## for plotting the grid here not the result .and its not part of the main thesis just plotting the grid behaviour

import seaborn as sns

import pandas as pd



def plot_cv_results(cv_results, param_x, param_z, metric='mean_test_score'):

    cv_results = pd.DataFrame(cv_results)

    col_x = 'param_' + param_x

    col_z = 'param_' + param_z

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    sns.pointplot(x=col_x, y=metric, hue=col_z, data=cv_results, ci=99, n_boot=64, ax=ax)

    ax.set_title("CV Grid Search Results")

    ax.set_xlabel(param_x)

    ax.set_ylabel('Accuracy')

    ax.legend(title=param_z)

    return fig
plt.rcParams.update({'font.size': 15})  ## plotting configuration



## plotting grid behaviour

plot_cv_results(grid.cv_results_, 'n_neighbors', 'weights')
## jupyter can render without show command thats why two pic
## plotting with respect to whole
from sklearn.metrics import confusion_matrix

knn_final = grid.best_estimator_  ## find the best estimator

tmp_pred_knn = knn_final.fit(x_train,y_train)  ## individual training for individual confusion matrix precision,recall,F1
import seaborn as sns

pr1 = tmp_pred_knn.predict(x_test)



confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)



from sklearn.metrics import precision_score,recall_score,f1_score

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))
sns.heatmap(confusion_matrix1)
from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}



# Create a based model

rf = RandomForestClassifier()



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2) ## njobs force the computer to use all their resources
grid_search.fit(x_train, y_train)
rf_final=grid_search.best_estimator_   # find the best estimator

tmp_pred_rf = rf_final.fit(x_train,y_train)       ## individual training 

pr1 = tmp_pred_rf.predict(x_test)   #3individual prediction

confusion_matrix1 =confusion_matrix(y_test,pr1)

print (confusion_matrix1)

sns.heatmap(confusion_matrix1)

print ("Precision score "+str(precision_score(y_test,pr1)))

print ("Recall score "+str(recall_score(y_test,pr1)))

print ("F1 score "+str(f1_score(y_test,pr1)))
plot_cv_results(grid_search.cv_results_, 'n_estimators', 'max_depth')

#param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
plot_cv_results(grid_search.cv_results_, 'n_estimators', 'min_samples_split')

#grid2 = GridSearchCV(SVC(),param_grid,n_jobs=-1,refit = True, verbose=2)
plot_cv_results(grid_search.cv_results_,'min_samples_split','max_depth')

#grid2.fit(x_train,y_train)
#svc_final = grid2.best_estimator_
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
## for plotting we can take all the parameter into accountso we take the most important for plotting separately

ACP=[]

#plot_cv_results(grid3.cv_results_,'C',)

C = range(1,100)

for item in C:

    model = LinearSVC(C=item)

    model.fit(x_train,y_train)

    ACP.append(model.score(x_test,y_test))

    
plt.rcParams["figure.figsize"] = fig_size

plt.rcParams.update({'font.size': 50})

plt.plot(C,ACP)
print (ACP)
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

              'min_samples_leaf': [20, 50,100,150],

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
learning_rate1= [0.1, 0.05, 0.02, 0.01]

max_depth1= [4, 6, 8]

min_samples_leaf1= [20, 50,100,150]



ACP3=[]

for item in min_samples_leaf1:

    model3 = GradientBoostingClassifier(min_samples_leaf=item)

    model3.fit(x_train,y_train)

    ACP3.append(model3.score(x_test,y_test))

    print(item)

plt.plot(min_samples_leaf1,ACP3)
print (gb_final)
# Define the parameter values that should be searched

sample_split_range = list(range(2, 50))



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
min_sample_split1=list(range(2, 50))



ACP4=[]

for item in min_sample_split1:

    model4 = DecisionTreeClassifier(min_samples_split=item)

    model4.fit(x_train,y_train)

    ACP4.append(model4.score(x_test,y_test))

    print(item)

plt.plot(min_sample_split1,ACP4)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

ML=[]

M=['DecisionTreeClassifier','KNeighborsRegressor','SVC','LinearSVC','RandomForestRegressor','GradientBoostingClassifier']

Z=[gb_final,knn_final,svc_final,lsvc_final,rf_final,gb_final]

##new

#M=['DecisionTreeClassifier','KNeighborsRegressor','SVC']

#Z=[DecisionTreeClassifier(),KNeighborsClassifier(),SVC()]
print (x_train.shape)

print (y_train.shape)

print (x_test.shape)

print (y_test.shape)



##new

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

ML
d={'Accuracy':ML,'Algorithm':M}

df1=pd.DataFrame(d)
df1
from sklearn.ensemble import VotingClassifier

voting_clf = MultiOutputClassifier(VotingClassifier(estimators=[('knn',knn_final),('rf',rf_final),('dt',dt_final),('svc',svc_final),('gb',gb_final)],voting='hard'))
voting_clf.fit(x_train1,y_train1)
predicted=voting_clf.predict(x_test1)
print (predicted)

#tmp_pred = gb_final.fit(x_train,y_train)

#pr1 = tmp_pred.predict(x_test)

cf1_d=[]

cf2_d=[]

for item in predicted:

    cf1_d.append(item[0])

for item in predicted:

    cf2_d.append(item[1])

    

y_test1
voting_hybrid_classifier=voting_clf.score(x_test1,y_test1)


confusion_matrix1 =confusion_matrix(np.array(y_test1['num_class']),np.array(cf1_d))

#confusion_matrix2 =confusion_matrix(y_test,pr1)



print (confusion_matrix1)

print ("Precision score "+str(precision_score(np.array(y_test1['num_class']),np.array(cf1_d))))

print ("Recall score "+str(recall_score(np.array(y_test1['num_class']),np.array(cf1_d))))

print ("F1 score "+str(f1_score(np.array(y_test1['num_class']),np.array(cf1_d))))

#print (confusion_matrix2)



sns.heatmap(confusion_matrix1)
confusion_matrix2 =confusion_matrix(np.array(y_test1['num_proto']),np.array(cf2_d))

#confusion_matrix2 =confusion_matrix(y_test,pr1)



print (confusion_matrix2)

print ("Precision score "+str(precision_score(np.array(y_test1['num_proto']),np.array(cf2_d),average='micro')))

print ("Recall score "+str(recall_score(np.array(y_test1['num_proto']),np.array(cf2_d),average='micro')))

print ("F1 score "+str(f1_score(np.array(y_test1['num_proto']),np.array(cf2_d),average='micro')))

#print (confusion_matrix2)



sns.heatmap(confusion_matrix2)
## mse

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test1,predicted)
mse
from sklearn.model_selection import cross_val_score
print(cross_val_score(voting_clf, x_test1, y_test1, cv=3))
print(cross_val_score(voting_clf, x_test1, y_test1, cv=10))
voting_hybrid_classifier
#df1=df1.drop('hybrid_voting_Classifier',axis=1)
ML.append(voting_hybrid_classifier)
M=['DecisionTreeClassifier','KNeighborsRegressor','SVC','LinearSVC','RandomForestRegressor','GradientBoostingClassifier','hybrid_voting_classifier']

d={'Accuracy':ML,'Algorithm':M}

df1=pd.DataFrame(d)
df1
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 50

fig_size[1] = 20

plt.rcParams["figure.figsize"] = fig_size



plt.bar(df1['Algorithm'],df1['Accuracy'])