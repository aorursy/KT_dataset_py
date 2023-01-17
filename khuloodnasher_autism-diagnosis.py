### define our metrics function with plotting confusion matrix

import itertools

from sklearn.metrics import confusion_matrix 



### define function for plotting confusion matrix

def plot_confusion_matrix(y_true, y_preds):

    # Print confusion matrix

    cnf_matrix = confusion_matrix(y_true, y_preds)

    # Create the basic matrix

    plt.imshow(cnf_matrix,  cmap=plt.cm.Blues)

    # Add title and axis labels

    plt.title('Confusion Matrix')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    # Add appropriate axis scales

    class_names = set(y) # Get class labels to add to matrix

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names, rotation=0)

    plt.yticks(tick_marks, class_names)

    # Add labels to each cell

    thresh = cnf_matrix.max() / 2. # Used for text coloring below

    # Here we iterate through the confusion matrix and append labels to our visualization

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):

            plt.text(j, i, cnf_matrix[i, j],

                     horizontalalignment='center',

                     color='white' if cnf_matrix[i, j] > thresh else 'black')

    # Add a legend

    plt.colorbar();

    plt.show();

def metrics(model_name, y_train, y_test, y_hat_train, y_hat_test):

    '''Print out the evaluation metrics for a given models predictions'''

    print(f'Model: {model_name}', )

    print('-'*60)

    plot_confusion_matrix(y_test,y_hat_test)

    print(f'test accuracy: {round(accuracy_score(y_test, y_hat_test),2)}')

    print(f'train accuracy: {round(accuracy_score(y_train, y_hat_train),2)}')

    print('-'*60)

    print('-'*60)

    print('Confusion Matrix:\n', pd.crosstab(y_test, y_hat_test, rownames=['Actual'], colnames=['Predicted'],margins = True))

    print('\ntest report:\n' + classification_report(y_test, y_hat_test))

    print('~'*60)

    print('\ntrain report:\n' + classification_report(y_train, y_hat_train))

    print('-'*60)
def plot_feature_importances(model):

    n_features = X_train.shape[1]

    plt.figure(figsize=(8, 8))

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), X_train.columns.values)

    plt.xlabel('Feature importance')

    plt.ylabel('Feature')
# Importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Supress warnings

import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



# Modelling Helpers :

from sklearn.preprocessing import Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, LabelEncoder

from sklearn.impute import SimpleImputer



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

from sklearn.metrics import accuracy_score,classification_report



# Classification

from sklearn.metrics import recall_score, f1_score, fbeta_score, r2_score, roc_auc_score, roc_curve, auc, cohen_kappa_score





## To display  all the interactive output without using the print function

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import pandas as pd

df=pd.read_csv("../input/autism-screening-for-toddlers/Toddler Autism dataset July 2018.csv")



df.info()

df.head(20)
df.shape
df.describe()
df.columns
df.drop(['Case_No', 'Who completed the test'], axis = 1, inplace = True)

df.columns
# Calculating the percentage of babies shows the symptoms of autisim

yes_autism= df[df['Class/ASD Traits ']=='Yes']

no_autism= df[df['Class/ASD Traits ']=='No']



print("Toddlers:",round(len(yes_autism)/len(df) * 100,2))



print("Toddlers:",round(len(no_autism)/len(df) * 100,2))







# Displaying the content of the target column

df['Class/ASD Traits '].value_counts()
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.set_size_inches(7,7)

plt.pie(df["Class/ASD Traits "].value_counts(),labels=('no_autism','yes_autism'),explode = [0.1,0],autopct ='%1.1f%%' ,

        shadow = True,startangle = 90,labeldistance = 1.1)

plt.axis('equal')



plt.show()
# Checking null data 

df.isnull().sum()
df.dtypes
corr = df.corr( )

plt.figure(figsize = (15,15))

sns.heatmap(data = corr, annot = True, square = True, cbar = True)
# Visualizing Juandice occurance in males and females

plt.figure(figsize = (16,8))



plt.style.use('dark_background')

sns.countplot(x = 'Jaundice', hue = 'Sex', data = yes_autism)
sns.countplot(x = 'Qchat-10-Score', hue = 'Sex', data = df)
#Visualizing  the age distribution of Positive ASD  among Todllers





f, ax = plt.subplots(figsize=(12, 8))

sns.countplot(x="Age_Mons", data=yes_autism, color="r");



plt.style.use('dark_background')

ax.set_xlabel('Toddlers age in months')

ax.set_title('Age distribution of ASD positive')



plt.figure(figsize = (16,8))

sns.countplot(x = 'Ethnicity', data = yes_autism)
#  visualize positive  ASD among Toddlers based on Ethnicity

plt.figure(figsize=(20,6))

sns.countplot(x='Ethnicity',data=yes_autism,order= yes_autism['Ethnicity'].value_counts().index[:11],hue='Sex',palette='Paired')

plt.title('Ethnicity Distribution of Positive ASD among Toddlers')

plt.xlabel('Ethnicity')

plt.tight_layout()

# Displaying number of positive cases of Autisim with Regards Ethnicity

yes_autism['Ethnicity'].value_counts()
#Lets visualize the distribution of autism in family within different ethnicity

f, ax = plt.subplots(figsize=(12, 8))





sns.countplot(x='Family_mem_with_ASD',data=yes_autism,hue='Ethnicity',palette='rainbow',ax=ax)

ax.set_title('Positive ASD Toddler relatives with Autism distribution for different ethnicities')

ax.set_xlabel('Toddler Relatives with ASD')

plt.tight_layout()





# removing 'Qchat-10-Score'

df.drop('Qchat-10-Score', axis = 1, inplace = True)
le = LabelEncoder()

columns = ['Ethnicity', 'Family_mem_with_ASD', 'Class/ASD Traits ', 'Sex', 'Jaundice']

for col in columns:

    df[col] = le.fit_transform(df[col])

df.dtypes

df.head(25)
X = df.drop(['Class/ASD Traits '], axis = 1)

y = df['Class/ASD Traits ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state =42)

X.isnull().sum()

X.info()




models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestRegressor()))

models.append(('XGB', XGBClassifier()))

models.append(('GBR', GradientBoostingRegressor()))

models.append(('ABR', AdaBoostRegressor()))



for name, model in models:

    model.fit(X_train, y_train)

    y_hat_test = model.predict(X_test).astype(int)

    y_hat_train = model.predict(X_train).astype(int)

    print(name, 'Accuracy Score is : ', round(accuracy_score(y_test, y_hat_test)))



    metrics(name, y_train, y_test, y_hat_train, y_hat_test)





    

    
for name, model in models:

    

    y_hat_test = model.predict(X_test).astype(int)

    y_hat_train = model.predict(X_train).astype(int)

    print(name, 'Accuracy Score is : ',round( accuracy_score(y_test, y_hat_test),2))
svc = SVC()



params = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

}



clf = GridSearchCV(svc, param_grid = params, scoring = 'accuracy', cv = 10, verbose = 2)



clf.fit(X_train, y_train)

clf.best_params_
# Re-running model with best parametres

svc1 = SVC(C = 0.8, gamma = 0.1, kernel = 'linear')

svc1.fit(X_train, y_train)

y_hat_test = svc1.predict(X_test)

#print(accuracy_score(y_test, y_hat_test))

metrics(svc1, y_train, y_test, y_hat_train, y_hat_test)

svcgrid_test_acc = round(accuracy_score(y_test, y_hat_test), 2)



svcgrid_test_acc


#Instantiate the pipeline

from sklearn.pipeline import Pipeline



pipe = Pipeline([('classifier', RandomForestClassifier(random_state=123))])

grid = [{'classifier__criterion': ['gini', 'entropy'],

         'classifier__n_estimators':[10, 20, 50, 100],

         'classifier__max_depth': [None, 5, 3, 10],

         'classifier__min_samples_split': [1.0, 6, 10],

         'classifier__min_samples_leaf': [1,  6, 10],

         'classifier__class_weight':['balanced']}]
clf = GridSearchCV(estimator=pipe, param_grid=grid,

                   cv=5, scoring='roc_auc', n_jobs=-1)

clf.fit(X_train, y_train)

y_hat_train = clf.predict(X_train)

y_hat_test = clf.predict(X_test)


metrics(clf, y_train, y_test, y_hat_train, y_hat_test)

print(round(clf.score(X_train, y_train)))

print(round(clf.score(X_test, y_test)))
clf.best_params_
# Research best estimator from grid

best_clf_estimator = clf.best_estimator_

best_clf_estimator.fit(X_train,y_train)
#Predictions

y_hat_train=best_clf_estimator.predict(X_train)

y_hat_test = best_clf_estimator.predict(X_test)

results=metrics(best_clf_estimator, y_train, y_test, y_hat_train, y_hat_test)

rf_gridsearch_test_acc = round(accuracy_score(y_test,y_hat_test), 2)

rf_gridsearch_test_acc
plot_feature_importances(model)
##Applying MLPClassifier Model 

from sklearn.neural_network import MLPClassifier

MLPClassifierModel = MLPClassifier(activation='tanh', # can be also identity , logistic , relu

                                   solver='lbfgs',  # can be also sgd , adam

                                   learning_rate='constant', # can be also invscaling , adaptive

                                   early_stopping= False,

                                   alpha=0.0001 ,hidden_layer_sizes=(100, 3),random_state=33)

MLPClassifierModel.fit(X_train, y_train)

#Calculating Prediction

y_hat_test = MLPClassifierModel.predict(X_test)

y_hat_train = MLPClassifierModel.predict(X_train)



results=metrics(MLPClassifierModel, y_train, y_test, y_hat_train, y_hat_test)



#Calculating Accuracy Score  

nn_sklearn_test_acc = round(accuracy_score(y_test, y_hat_test, normalize=True),2)



print('Accuracy Score is : ', nn_sklearn_test_acc)

# Define simple neural network model

# Keras



from keras.models import Sequential

from keras.layers import *

from keras.optimizers import Adam, RMSprop











model = Sequential()

# Define Input Layer wits 15 features as an input

model.add(Dense(100, input_dim=15, activation='relu'))

#model.add(Dense(1, activation='sigmoid'))

#single output layer with one neuron since we only want to predict two classes either yes autisim =1 or no autisim=zero

model.add(Dense(activation = 'sigmoid', units = 1))





# Compile the Neural network

model.compile(loss='binary_crossentropy', # we use binarray here becuase we just have 2 classes

             optimizer = Adam(lr=0.0001, decay=1e-5), ### learning rate 0.0001

              metrics=['acc'])

# Fit to training data

model.fit(X_train, y_train, epochs=100,  

          validation_data=(X_test, y_test))
# collecting the summary of our neural network paramters

model.summary()

#structure of keras neuralnetwork model

from keras.utils import plot_model



plot_model(model)

# looking at the structure of my neural network
def evaluate_clf(y_true, y_pred):

    """Return confusion matrix, classification report, and accuracy score

    for a classifier.

    

    Parameters

    ----------

    y_true : array-like

        Target class labels

    y_pred : array-like

        Predicted class labels

        

    Returns

    ----------

    Confusion matrix, classification report, accuracy score

    """

    

    test_acc = round(accuracy_score(y_true, y_pred), 2)

    

    print('Confusion Matrix:')

    print(confusion_matrix(y_test, y_pred))

    print('---'*20)

    print('Classification Report:')

    print(classification_report(y_test, y_pred))

    print('---'*20)

    print("kerasNN_test_acc:",round(accuracy_score(y_test,y_hat_test), 2))

    print("kerasNN_train_acc:" ,round(accuracy_score(y_train,y_hat_train), 2))

#Predictions



y_hat_test = model.predict_classes(X_test)

y_hat_train = model.predict_classes(X_train)

# applying the metrics function

evaluate_clf(y_test, y_hat_test)





# Create classifier summary table



LogisticRegression_Accuracy =  1.0

LinearDiscriminantAnalysis_accuracy  =  0.96

KNeighborsClassifier_accuracy =  0.91

DecisionTreeClassifier_accuracy =  0.91

GaussianNB_accuracy =  0.94

SVC_beforegrid_accuracy =  0.78

RandomForest_beforegrid_accuracy =  0.64

XGBClassifier_accuracy=  0.99

GradientBoosting_accuracy =  0.64,

AdaBoosting_accuracy = 0.49

SVC_aftergrid_accuracy = 1.0

RandomForest_aftergrid_accuracy =0.96

Neuralnetwork_SKLearn_accuracy= 0.99

Neuralnetwork_Keras_accuracy = 0.95







models=['LogisticRegression','LinearDiscriminantAnalysis',

       'KNeighborsClassifier','DecisionTreeClassifier',

        'GaussianNB','SVC_beforegrid',

        'RandomForest_beforegrid','XGBClassifier',

        'GradientBoosting','AdaBoosting','SVC_aftergrid','RandomForest_aftergrid','Neuralnetwork_SKLearn Accuracy','Neuralnetwork_Keras']





test_Accuracy=[1.0,0.96,0.91,0.91,0.94, 0.78,0.64, 0.99,0.64,0.49,1.0,0.96,0.99,0.95]









accuracy_summary = pd.DataFrame([models, test_Accuracy]).T

accuracy_summary.columns = ['Classifier', 'test_Accuracy']

accuracy_summary