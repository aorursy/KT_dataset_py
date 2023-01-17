import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

train = shuffle(pd.read_csv("../input/train.csv"))

test = shuffle(pd.read_csv("../input/test.csv"))

train_outcome = pd.crosstab(index=train["Activity"], columns="count")

train_outcome

temp = train["Activity"].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })



#df.plot(kind='pie',labels='labels',values='values', title='Activity Ditribution',subplots= "True")



labels = df['labels']

sizes = df['values']

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
# Seperating Predictors and Outcome values from train and test sets

X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))

Y_train_label = train.Activity.values.astype(object)

X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))

Y_test_label = test.Activity.values.astype(object)



# Dimension of Train and Test set 

#print("Dimension of Train set",X_train.shape)

#print("Dimension of Test set",X_test.shape,"\n")



# Transforming non numerical labels into numerical labels

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()



# encoding train labels 

encoder.fit(Y_train_label)

Y_train = encoder.transform(Y_train_label)



# encoding test labels 

encoder.fit(Y_test_label)

Y_test = encoder.transform(Y_test_label)



#Total Number of Continous and Categorical features in the training set

num_cols = X_train._get_numeric_data().columns

print("Number of numeric features:",num_cols.size)

#list(set(X_train.columns) - set(num_cols))





names_of_predictors = list(X_train.columns.values)



# Scaling the Train and Test feature set 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
#Libraries to Build Ensemble Model : Random Forest Classifier 

# Create the parameter grid based on the results of random search 

params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Performing CV to tune parameters for best SVM fit 

svm_model = GridSearchCV(SVC(), params_grid, cv=5)

svm_model.fit(X_train_scaled, Y_train)
final_model = svm_model.best_estimator_

Y_pred = final_model.predict(X_test_scaled)

Y_pred_label = list(encoder.inverse_transform(Y_pred))
print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))

print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))



svm_model.score