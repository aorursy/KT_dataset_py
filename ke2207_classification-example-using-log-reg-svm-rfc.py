%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report

from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from scipy.stats import randint, uniform

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
import os

print(os.listdir("../input/"))
# load the dataset into a pandas dataframe

dataset = pd.read_csv('../input/nasa-asteroids-classification/nasa.csv')

dataset.head()
# Removal of irrelevant features

dataset = dataset.drop(['Neo Reference ID', 'Name', 'Close Approach Date',

                        'Epoch Date Close Approach', 'Orbit Determination Date'], axis=1)
print(dataset['Orbiting Body'].unique())

print(dataset['Equinox'].unique())
# Removal of features with no discriminative value

dataset = dataset.drop(['Orbiting Body', 'Equinox'], axis=1)
# heatplot

f, ax = plt.subplots(figsize=(20, 20))

corr = dataset.corr("pearson")

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax,annot=True)
#calculate the relative difference between the two features

diff = (dataset['Est Dia in M(max)'] - dataset['Est Dia in M(min)']) / dataset['Est Dia in M(min)']

print("maximum value is: {0}".format(diff.max()))

print("minimum value is: {0}".format(diff.min()))
#removal of correlated (correlation equals 1) columns

dataset = dataset.drop(['Est Dia in KM(max)', 'Est Dia in M(min)',

                        'Est Dia in M(max)', 'Est Dia in Miles(min)','Est Dia in Miles(max)','Est Dia in Feet(min)','Est Dia in Feet(max)'], axis=1)

dataset = dataset.drop(['Relative Velocity km per hr','Miles per hour'], axis=1)

dataset = dataset.drop(['Miss Dist.(lunar)','Miss Dist.(kilometers)','Miss Dist.(miles)'], axis=1)
# heatplot

f, ax = plt.subplots(figsize=(20, 20))

corr = dataset.corr("pearson")

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax,annot=True)
#check for missing values

print(dataset.isnull().sum())
#check dataset for duplicate samples

dataset.duplicated().value_counts()
#print the statistical metrics

dataset.describe()
#plot the boxplots of all features

plt.tight_layout(pad=0.9)

plt.figure(figsize=(35,30)) 

plt.subplots_adjust(wspace = 0.2  )

nbr_columns = 4 

nbr_graphs = len(dataset.columns) 

nbr_rows = int(np.ceil(nbr_graphs/nbr_columns)) 

columns = list(dataset.columns.values) 

for i in range(0,len(columns)-1): 

    plt.subplot(nbr_rows,nbr_columns,i+1) 

    ax1=sns.boxplot(x= columns[i], data= dataset, orient="h") 



plt.show() 
#countplot of labels

print("label balance:")

print(dataset.Hazardous.value_counts())



ax1=sns.countplot(dataset.Hazardous,color="navy")
# one-hot encoding of 'Orbit ID'

dataset = pd.concat([dataset,pd.get_dummies(dataset['Orbit ID'], prefix='Orbit_ID')],axis=1)



dataset.drop(['Orbit ID'],axis=1, inplace=True)

dataset.head()
# Make labels in column "Hazardous" numerical: False / True -> 0 / 1

dataset['Hazardous'] = pd.factorize(dataset['Hazardous'], sort=True)[0]

print(dataset.Hazardous[0:5])
# separating the classlabels from the features  

y = dataset.Hazardous.values

X = dataset.drop(['Hazardous'],axis=1)
# split the featureset into the numerical and one hot encoded columns

X_num = X.loc[:,'Absolute Magnitude':'Mean Motion']

X_One_Hot = X.loc[:,'Orbit_ID_1':].values



# Standardize the numerical features with the Robust Scaler

from sklearn.preprocessing import RobustScaler

RbS = RobustScaler().fit(X_num)

X_num = RbS.transform(X_num)



# merge all features back together in one numpy array

X = np.concatenate((X_num,X_One_Hot),axis=1)
# Split up the dataset into a training and a test set with a 1000 waarden in test set en random_state = 0. Normaliseer de features.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)





#Function to train a model and log the metrics

def train_model(model,X_train, y_train, X_test,y_test):

    

    start_time = time.time()

    model.fit(X_train, y_train)

    

    delta_time = (time.time() - start_time)

    y_predict = model.predict(X_test)

    acc_model = accuracy_score(y_test, y_predict)

    prec_model = precision_score(y_test, y_predict,average= None)

    recall_model = recall_score(y_test, y_predict,average= None)

    log = np.array([[acc_model,prec_model[0],prec_model[1],recall_model[0],recall_model[1],delta_time]])

    

    print("training time: {0}".format(delta_time))

    print("accuracy: {0}".format(acc_model))

    print("\nconfusion matrix: ")

    print("-----------------------")

    print(confusion_matrix(y_test, y_predict))

    target_names = ['Not hazardous', 'Hazardous']

    print("\nclassification report:")

    print("-----------------------")

    print(classification_report(y_test, y_predict,target_names=target_names))

       

    return model, log
# Train and test the logistic regression classifier

Log_reg_model = linear_model.LogisticRegression(C=0.001, solver='lbfgs', multi_class='auto')

Log_reg_model, model_log = (train_model(Log_reg_model,X_train, y_train, X_test,y_test))
# Train a logistic regression model through cross-validation en GridSearch 

#--------------------------------------------------------------------------



parameters =  {'C' : [0.1, 1, 10, 100], 

              'class_weight': [None,'balanced'],

              'solver': ['newton-cg', 'lbfgs', 'liblinear']}

             

Grid_Log_model = GridSearchCV(estimator = LogisticRegression(), 

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 6,

                           n_jobs = -1)



Grid_Log_model,log = train_model(Grid_Log_model,X_train, y_train, X_test,y_test)

model_log= np.append(model_log,log,axis=0)
# Train a Support Vector Machine

from sklearn.svm import SVC



parameters = {'C' : [8,9, 10, 11, 12,13, 14], 

              'kernel' : ['linear', 'rbf'],

              'degree': [1],

              'gamma': (0.01, 0.1,1)}

             

#SVC_model = GridSearchCV(estimator = SVC(), 

#                           param_grid = parameters,

#                           scoring = 'accuracy',

#                           cv = 6,

#                           n_jobs = -1)



SVC_model= SVC(C=20, degree=3, gamma='auto', kernel='rbf')



SVC_model,log = train_model(SVC_model,X_train, y_train, X_test,y_test)

model_log= np.append(model_log,log,axis=0)
# Train a Random Forest Classifier

number_of_trees = 1000

max_number_of_features = 15



RFC_model = RandomForestClassifier(n_estimators=number_of_trees, max_features=max_number_of_features)



RFC_model,log = train_model(RFC_model,X_train, y_train, X_test,y_test)

model_log= np.append(model_log,log,axis=0)
# Boosting

# Boosting met logistic regression



import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)



number_of_estimators = 200

complexity = 10

cart = cart = LogisticRegression(solver='lbfgs', C=complexity)

ADA_model = BaggingClassifier(base_estimator=cart, n_estimators=number_of_estimators)



ADA_model,log = train_model(ADA_model,X_train, y_train, X_test,y_test)

model_log= np.append(model_log,log,axis=0)
model_names = ('Log reg', 'Log reg gridsearch','SVC','Rand Forest', 'Adaboost')

column_names = ('accuracy','precision Not hazardous','precision hazardous','recall Not hazardous','recall hazardous','training time')



plt.figure(figsize=(16,16)) 

colors=[1000,500,1000,900]

colors = [x / max(colors) for x in colors]

my_cmap = plt.cm.get_cmap('GnBu')

colors = my_cmap(colors)

for i in range(0,6): 

    plt.subplot(3,2,i+1) 

    ax=plt.bar(x=model_names,height=model_log[:,i],color=colors)

    plt.title(column_names[i])

plt.show()
pd.DataFrame(data=model_log,index=model_names,columns=column_names)
importances = RFC_model.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(20,7))

plt.title("Feature importances")

ax = plt.bar(range(X.shape[1]), importances[indices],

       color="r", align="center")



plt.show()
#removal of all but 10 most relevant features

least_important_feat_10=importances.argsort()[:-10]

X_red_10 = np.delete(X,least_important_feat_10,1)

print(X_red_10.shape)
# Split up the dataset into a training and a test set

X_train_red_10, X_test_red_10, y_train_red_10, y_test_red_10 = train_test_split(X_red_10, y, test_size=0.25, random_state=0)
# Train a Random Forest Classifier

number_of_trees = 1000

max_number_of_features = 9



RFC_model_red = RandomForestClassifier(n_estimators=number_of_trees, max_features=max_number_of_features)



RFC_model_red,log = train_model(RFC_model_red,X_train_red_10, y_train_red_10, X_test_red_10,y_test_red_10)

model_log= np.append(model_log,log,axis=0)
#visualise the pairplot of the ten most significant features

ds = pd.DataFrame(X_test_red_10)

ds['label']= y_test_red_10

sns.pairplot(ds, hue="label")
#removal of all but the 2 most relevant features

least_important_feat_2=importances.argsort()[:-2]

X_red_2 = np.delete(X,least_important_feat_2,1)

print(X_red_2.shape)
# Split up the dataset into a training and a test set

X_train_red_2, X_test_red_2, y_train_red_2, y_test_red_2 = train_test_split(X_red_2, y, test_size=0.25, random_state=0)
# Train a Random Forest Classifier

number_of_trees = 1000

max_number_of_features = 2



RFC_model_red_2 = RandomForestClassifier(n_estimators=number_of_trees, max_features=max_number_of_features)



RFC_model_red_2,log = train_model(RFC_model_red_2,X_train_red_2, y_train_red_2, X_test_red_2,y_test_red_2)

model_log= np.append(model_log,log,axis=0)
#plotting the two most signifcant features for the testsamples including the RFC boundaries



def plot_2D_boundary(model,X_test):

    h = .01

    plt.figure(figsize=(18,10))



    x_min, x_max = X_test_red_2[:, 0].min() - 0.5, X_test_red_2[:, 0].max() + 0.5

    y_min, y_max = X_test_red_2[:, 1].min() - 0.5, X_test_red_2[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))



    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, m_max]x[y_min, y_max].

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X_test_red_2[:, 0], X_test_red_2[:, 1], c=y_test_red_2, cmap=plt.cm.Paired)



plot_2D_boundary(RFC_model_red_2,X_test_red_2)
#Training of a decision tree using the two most significant features

from sklearn import tree

detr_model = tree.DecisionTreeClassifier()

detr_model,log = train_model(detr_model,X_train_red_2, y_train_red_2, X_test_red_2,y_test_red_2)

model_log= np.append(model_log,log,axis=0)
plot_2D_boundary(detr_model,X_test_red_2)