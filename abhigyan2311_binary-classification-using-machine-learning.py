# Importing Libraries



import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from xgboost import XGBClassifier



from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
# Setting the numpy random seed



np.random.seed(37)
# Loading Dataset



df = pd.read_csv('../input/attrition_data.csv')

print('Dataframe shape: ', df.shape)
df.head()
# Checking for missing values



df.isnull().sum()
# Dropping irrelevant columns



df.drop(['EMP_ID', 'JOBCODE', 'TERMINATION_YEAR'], axis=1, inplace=True)

df.drop(df.iloc[:, -5:], axis=1, inplace=True)
df['REFERRAL_SOURCE'].fillna(df['REFERRAL_SOURCE'].mode()[0], inplace=True)
df.head()
sns.set(style="darkgrid")

ax = sns.countplot(x="STATUS", data=df, palette=sns.xkcd_palette(["azure", "light red"]))

plt.xlabel('Status')

plt.ylabel('Count')

# plt.savefig('./plots/status_count.png')

plt.show()
fig=plt.figure(figsize=(8,4))

for x in ['T','A']:

    df['AGE'][df['STATUS']==x].plot(kind='kde')

    

plt.title('Status V/S Age Density Distribution')

plt.legend(('T','A'))

plt.xlabel('Age')

# plt.savefig('./plots/status_age_distribution.png')

plt.show()
sns.countplot(x='PERFORMANCE_RATING', data=df, hue='STATUS', palette=sns.xkcd_palette(["azure", "light red"]))

plt.title("Performance Rating Count Plot")

plt.xlabel('Performance Rating')

plt.ylabel('Count')

# plt.savefig('./plots/performance_count.png')

plt.show()
sns.countplot(x='JOB_SATISFACTION', data=df, hue='STATUS', palette=sns.xkcd_palette(["aqua", "periwinkle"]))

plt.title("Job Satisfaction Count Plot")

plt.xlabel('Job Satisfaction')

plt.ylabel('Count')

# plt.savefig('./plots/satisfaction_count.png')

plt.show()
sns.boxplot(x='JOB_SATISFACTION',data=df,hue='STATUS',y='AGE', palette=sns.xkcd_palette(["pastel purple", "pastel yellow"]))

plt.title("Job Satisfaction and Age Boxplot")

plt.xlabel('Job Satisfaction')

plt.ylabel('Age')

# plt.savefig('./plots/age_satisfaction_box.png')

plt.show()
# Label Encoding categorical features



le = LabelEncoder()

df['NUMBER_OF_TEAM_CHANGED'] = le.fit_transform(df['NUMBER_OF_TEAM_CHANGED'])

df['REHIRE'] = le.fit_transform(df['REHIRE'])

df['IS_FIRST_JOB'] = le.fit_transform(df['IS_FIRST_JOB'])

df['TRAVELLED_REQUIRED'] = le.fit_transform(df['TRAVELLED_REQUIRED'])

df['DISABLED_EMP'] = le.fit_transform(df['DISABLED_EMP'])

df['DISABLED_VET'] = le.fit_transform(df['DISABLED_VET'])

df['EDUCATION_LEVEL'] = le.fit_transform(df['EDUCATION_LEVEL'])

df['STATUS'] = le.fit_transform(df['STATUS'])
# Correlation Heatmap



fig, ax = plt.subplots(figsize=(15,10))

sns.heatmap(df.corr(), annot = True, ax=ax)

# plt.savefig('./plots/correlation_heatmap.png')

plt.show()
df.drop(['HRLY_RATE'], axis=1, inplace=True)
# One-Hot Encoding categorical features



df['HIRE_MONTH'] = df['HIRE_MONTH'].astype('category')

df['JOB_GROUP'] = df['JOB_GROUP'].astype('category')

df['REFERRAL_SOURCE'] = df['REFERRAL_SOURCE'].astype('category')

df['ETHNICITY'] = df['ETHNICITY'].astype('category')

df['SEX'] = df['SEX'].astype('category')

df['MARITAL_STATUS'] = df['MARITAL_STATUS'].astype('category')

df = pd.get_dummies(df, columns=['HIRE_MONTH', 'JOB_GROUP', 'REFERRAL_SOURCE', 'SEX', 'MARITAL_STATUS', 'ETHNICITY'])
# X = features & y = Target class



X = df.drop(['STATUS'], axis=1)

y = df['STATUS']
# Normalizing the all the features



scaler = StandardScaler()



X = scaler.fit_transform(X)
# Splitting dataset into training and testing split with 70-30% ratio



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# K-fold splits



cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'penalty': ['l1','l2'],

    'C': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]

}



# Building model

logreg = LogisticRegression(solver='liblinear')



# Parameter estimating using GridSearch

grid = GridSearchCV(logreg, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
logreg_grid_val_score = grid.best_score_

print('Best Score:', logreg_grid_val_score)

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



logreg_grid = grid.best_estimator_

y_pred = logreg_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



logreg_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', logreg_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'n_neighbors': [3,5,11,19],

    'weights': ['uniform','distance']

}



# Building model

knn = KNeighborsClassifier()



# Parameter estimating using GridSearch

grid = GridSearchCV(knn, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
knn_grid_val_score = grid.best_score_

print('Best Score:', knn_grid_val_score)

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



knn_grid= grid.best_estimator_

y_pred = knn_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



knn_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', knn_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Building our model with K-fold validation and GridSearch to find the best parameters



# No such parameters for Gaussian Naive Bayes

params = {}



# Building model

gb = GaussianNB()



# Parameter estimating using GridSearch

grid = GridSearchCV(gb, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
gb_grid_val_score = grid.best_score_

print('Best Score:', gb_grid_val_score)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



gb_grid= grid.best_estimator_

y_pred = gb_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



gb_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', gb_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'C': [0.001, 0.01, 0.1, 1, 10], 

    'gamma' : [0.001,0.001, 0.01, 0.1, 1]

}



# Building model

svc = SVC(kernel='rbf', probability=True) ## 'rbf' stands for gaussian kernel



# Parameter estimating using GridSearch

grid = GridSearchCV(svc, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
svm_grid_val_score = grid.best_score_

print('Best Score:', svm_grid_val_score)

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



svm_grid= grid.best_estimator_

y_pred = svm_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



svm_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', svm_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'max_features': [1, 3, 10],

    'min_samples_split': [2, 3, 10],

    'min_samples_leaf': [1, 3, 10],

    'criterion': ["entropy", "gini"]

}



# Building model

dtc = DecisionTreeClassifier()



# Parameter estimating using GridSearch

grid = GridSearchCV(dtc, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
dtc_grid_val_score = grid.best_score_

print('Best Score:', dtc_grid_val_score)

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



dtc_grid= grid.best_estimator_

y_pred = dtc_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



dtc_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', dtc_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'max_features': [1, 3, 10],

    'min_samples_split': [2, 3, 10],

    'min_samples_leaf': [1, 3, 10],

    'bootstrap': [False],

    'n_estimators' :[100,300],

    'criterion': ["entropy", "gini"]

}



# Building model

rfc = RandomForestClassifier()



# Parameter estimating using GridSearch

grid = GridSearchCV(rfc, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
rfc_grid_val_score = grid.best_score_

print('Best Score:', rfc_grid_val_score)

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



rfc_grid= grid.best_estimator_

y_pred = rfc_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



rfc_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', rfc_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Defining our neural network model



def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(8, activation='relu'))

    model.add(Dense(4, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'optimizer': ['rmsprop', 'adam'],

    'epochs': [100, 200, 400],

    'batch_size': [5, 10, 20]

}



# Building model

nn = KerasClassifier(build_fn=create_model)



# Parameter estimating using GridSearch

grid = GridSearchCV(nn, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
nn_grid_val_score = grid.best_score_

print('Best Score:', nn_grid_val_score) 

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



nn_grid= grid.best_estimator_

y_pred = nn_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



nn_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', nn_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
# Building our model with K-fold validation and GridSearch to find the best parameters



# Defining all the parameters

params = {

    'max_depth': range (2, 10, 1),

    'n_estimators': range(60, 220, 40),

    'learning_rate': [0.1, 0.01, 0.05]

}



# Building model

xgb = XGBClassifier(objective='binary:logistic')



# Parameter estimating using GridSearch

grid = GridSearchCV(xgb, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)



# Fitting the model

grid.fit(X_train, y_train)
xgb_grid_val_score = grid.best_score_

print('Best Score:', xgb_grid_val_score) 

print('Best Params:', grid.best_params_)

print('Best Estimator:', grid.best_estimator_)
# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)



xgb_grid= grid.best_estimator_

y_pred = xgb_grid.predict(X_test)
# Confusion matrix



pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )
# Calculating metrics



xgb_grid_score = accuracy_score(y_test, y_pred)

print('Model Accuracy:', xgb_grid_score)

print('Classification Report:\n', classification_report(y_test, y_pred))
score_df = pd.DataFrame(

    [

        ['Logistic Regression', logreg_grid_score, logreg_grid_val_score],

        ['K-Nearest Neighbors', knn_grid_score, knn_grid_val_score],

        ['Gaussian Naïve Bayes', gb_grid_score, gb_grid_val_score],

        ['Support Vector Machines', svm_grid_score, svm_grid_val_score],

        ['Decision Tree Classifier', dtc_grid_score, dtc_grid_val_score],

        ['Random Forest Tree Classifier', rfc_grid_score, rfc_grid_val_score],

        ['Artificial Neural Networks', nn_grid_score, nn_grid_val_score],

        ['GBM - XGBoost', xgb_grid_score, xgb_grid_val_score], 

    ],

    columns= ['Model', 'Test Score', 'Validation Score']

)

score_df['Test Score'] = score_df['Test Score']*100

score_df['Validation Score'] = score_df['Validation Score']*100
score_df
fig, ax1 = plt.subplots(figsize=(10, 5))

tidy = score_df.melt(id_vars='Model').rename(columns=str.title)

sns.barplot(x='Model', y='Value', hue='Variable', data=tidy, ax=ax1, palette=sns.xkcd_palette(["azure", "light red"]))

plt.ylim(20, 90)

plt.xticks(rotation=45, horizontalalignment="right")

# plt.savefig('./plots/result.png')

sns.despine(fig)
time_df = pd.DataFrame(

    [

        ['Logistic Regression', 1.2],

        ['K-Nearest Neighbors', 1.0],

        ['Gaussian Naïve Bayes', 0.0034],

        ['Support Vector Machines', 51.7],

        ['Decision Tree Classifier', 0.068],

        ['Random Forest Tree Classifier', 15.1],

        ['Artificial Neural Networks', 454.2],

        ['GBM - XGBoost', 40.8], 

    ],

    columns= ['Model', 'Training Time']

)
fig, ax1 = plt.subplots(figsize=(10, 5))

sns.barplot(data=time_df, x='Model', y='Training Time', palette=sns.color_palette('husl'))

plt.xticks(rotation=45, horizontalalignment="right")

plt.ylabel('Training Time(in mins)')

# plt.savefig('./plots/training_time.png')

sns.despine(fig)