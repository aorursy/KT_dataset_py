# Libraries

import numpy as np

import pandas as pd

import seaborn as sns
# store the data into a variable

df = pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv", sep = ';')



# Print the head of data

df.head()
# check for any null or missing values

df.isnull().values.any()
# View some basic statistics

df.describe().T
# Get a count of the num of patients with a cardiovascular disease and without



df['cardio'].value_counts()
# Visualize the count

sns.countplot(df['cardio'])
# create a years column

df['years'] = (df['age']/365).round(0)

df['years'] = pd.to_numeric( df['years'], downcast = 'integer')



# Visualize the data

sns.countplot(x='years', hue = 'cardio', data = df, palette = 'colorblind', edgecolor = sns.color_palette('dark', n_colors = 1));
# orrelation table

import matplotlib.pyplot as plt



plt.figure(figsize =(7,7))

sns.heatmap(df.corr(), annot=True, fmt = '.0%');

# with that heatmap we can see easily correlations
# drop the years column

df = df.drop('years', axis = 1) # axis = 1 means column

# drop the id column

df = df.drop('id', axis = 1)
# split the data into deature data and target data

X = df.iloc[:, :-1].values

Y = df.iloc[:, -1].values
# Split the data into 67% training - 33% testing



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
# Feature Scaling

# Scale the values in the data to be values btw 0 and 1 inclusive



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



# Use Random farest classifier

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)

forest.fit(X_train, y_train)
# Test our model's accuracy on the training data set



model = forest

model.score(X_train, y_train)

#0.98 is a not bad score but I tried tune the model, in next step

# Test the model's accuracy on the test data set 



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, model.predict(X_test))



TN = cm[0][0]

TP = cm[1][1]

FN = cm[1][0]

FP = cm[0][1]



# print the confusion matrix

print(cm)



# Print the models accuracy on the test data

print('Model Test Accuracy = {}'.format((TP+TN)/ (TP+TN+FN+FP)))
# Tune our model

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



rf_params = {"max_depth": [2,5,8,10],

             "max_features": [2,5,8],

             "n_estimators": [10,500,1000],

             "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params,

                           cv = 10,

                           n_jobs = -1,

                           verbose = 2)
rf_cv_model.fit(X_train, y_train)
print("Best rf parameters: " + str(rf_cv_model.best_params_))
rf_model = RandomForestClassifier(max_depth = 8,

                                  max_features = 5 ,

                                  min_samples_split = 10 ,

                                  n_estimators = 500)

                                

rf_model.fit(X_train, y_train)
# Test our model's accuracy on the training data set



tuned_model = rf_model

tuned_model.score(X_train, y_train)
# Test the model's accuracy on the test data set 



cm_tuned = confusion_matrix(y_test, tuned_model.predict(X_test))



TN = cm_tuned[0][0]

TP = cm_tuned[1][1]

FN = cm_tuned[1][0]

FP = cm_tuned[0][1]



# print the confusion matrix

print(cm_tuned)



# Print the models accuracy on the test data

print('Tuned Model Test Accuracy = {}'.format((TP+TN)/ (TP+TN+FN+FP))) 