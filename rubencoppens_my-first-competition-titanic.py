# Loading relevant libraries

import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import math
# Reading the dataset

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.set_index('PassengerId', inplace=True)

print("Shape : ", df.shape)

print(df.dtypes)

df.head()
df =  df.drop(["Name","Ticket"], axis=1) # Remove non-relevant values
df.describe()
missing_columns = df.columns[df.isnull().any()] # Which columns have missing values?

df.isnull().sum(axis=0).loc[missing_columns] # Age: average imputation, Cabin: drop column due to too much missing values, impute Embarked 
df = df.drop(["Cabin"], axis=1)
def plot_survival_rate(df, variable, i): # Helper function

    obs = df.shape[0]

    plot_df = df.groupby(variable)["Survived"].agg(["sum","count"]).rename(columns = { "sum":"survived", "count":"exposure"}).reset_index(drop=False)

    plot_df["survived"] = plot_df["survived"]/plot_df["exposure"]

    plot_df["exposure"] = plot_df["exposure"]/obs



    ax = fig.add_subplot(2, 3, i)

    sns.lineplot(x=plot_df[variable],y=plot_df["survived"],ax=ax, ci=None) # How does the respons behave?

    sns.barplot(x=plot_df[variable],y=plot_df["exposure"],ax=ax, ci=None) # How is the explanatory variable distributed?

fig = plt.figure(figsize=(15,8))

plot_survival_rate(df, "Sex",1)

plot_survival_rate(df, "Pclass",2)

plot_survival_rate(df, "SibSp",3)

plot_survival_rate(df, "Parch",4)

plot_survival_rate(df, "Embarked",5)

fig.show()
df_temp = df.copy()

df_temp["binAge"] = np.round(df_temp["Age"])

df_temp["binFare"] = np.minimum(np.round(df_temp["Fare"]),50)



fig = plt.figure(figsize=(20,10))

plot_survival_rate(df_temp, "binAge",1)

plot_survival_rate(df_temp, "binFare",2)

fig.show()
df.columns

df.dtypes
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



X_test = test_data.copy()

X_test = X_test.drop(["Name","Ticket","Cabin","PassengerId"], axis=1) # Remove non-relevant values

X_test.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler





# Separate target from predictors

y = df.Survived

X = df.drop(['Survived'], axis=1)





# Select numerical and categorical columns

numerical_cols = [cname for cname in X if X[cname].dtype in ['int64', 'float64']]

categorical_cols = [cname for cname in X if X[cname].dtype == "object"]



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



# Bundle preprocessing for numerical and categorical data

preprocessor_temp = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

preprocessor = Pipeline(steps=[

    ('preprocessor_temp', preprocessor_temp),

    ('scaler', StandardScaler())])



preprocessor.fit(X)
X.iloc[0,:]
dim_trans = preprocessor.transform(X).shape[1]

preprocessor.transform(X)[0,:]
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



params_knn = {"kNN__n_neighbors": np.arange(3,10), "kNN__weights":["uniform", "distance"]}



kNN = KNeighborsClassifier()

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('kNN', kNN)])

knnpipe = GridSearchCV(pipe, params_knn, n_jobs=-1)

knnpipe.fit(X, y)

print("Best parameter (CV score=%0.3f):" % knnpipe.best_score_)

print(knnpipe.best_params_)

print()

print("Accuracy training set %0.3f" % accuracy_score(y, knnpipe.predict(X)))
from sklearn.linear_model import LogisticRegression



params_logit = {"logit__penalty": ["l1","l2"], "logit__C":np.arange(0.5,1.5,0.1)}



logit = LogisticRegression()

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('logit', logit)])

logitpipe = GridSearchCV(pipe, params_logit, n_jobs=-1)

logitpipe.fit(X, y)

print("Best parameter (CV score=%0.3f):" % logitpipe.best_score_)

print(logitpipe.best_params_)

print()

print("Accuracy training set %0.3f" % accuracy_score(y, logitpipe.predict(X)))
from sklearn.svm import SVC



params_svm = {"svm__kernel": ["linear","poly","rbf"], "svm__C":np.arange(0.5,1.5,0.1)}



svm = SVC()

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm)])

svmpipe = GridSearchCV(pipe, params_svm, n_jobs=-1)

svmpipe.fit(X, y)

print("Best parameter (CV score=%0.3f):" % svmpipe.best_score_)

print(svmpipe.best_params_)

print()

print("Accuracy training set %0.3f" % accuracy_score(y, svmpipe.predict(X)))
from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from keras.models import load_model



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state = 1)



preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)

X_valid = preprocessor.transform(X_valid)



network = Sequential([

  Dense(16, activation='relu', input_shape=(dim_trans,)),

  Dense(16, activation='relu'),

  Dense(1, activation='sigmoid'),

])



network.compile(loss='binary_crossentropy', # Cross-entropy

                optimizer='rmsprop', # Root Mean Square Propagation

                metrics=['accuracy']) # Accuracy performance metric



es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



# Train neural network

history = network.fit(X_train, # Features

                      y_train, # Target vector

                      epochs=1000, # Number of epochs

                      verbose=1, # Print description after each epoch

                     validation_data=(X_valid,y_valid),# Data for evaluation

                     callbacks=[es, mc]) 



saved_model = load_model('best_model.h5')

print("Accuracy training set %0.3f" % accuracy_score(y, saved_model.predict(preprocessor.transform(X))>0.5))
plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()
from xgboost import XGBClassifier

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state = 1)



preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)

X_valid = preprocessor.transform(X_valid)



xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=-1)

xgb.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=True)



print("Accuracy training set %0.3f" % accuracy_score(y, xgb.predict(preprocessor.transform(X),ntree_limit=xgb.best_ntree_limit)))
predictions = xgb.predict(preprocessor.transform(X_test),ntree_limit=xgb.best_ntree_limit)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")