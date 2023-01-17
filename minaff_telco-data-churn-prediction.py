import pandas as pd

import numpy as np

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import scale

from keras import Sequential, Input

from keras.layers import Dense, Dropout

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.dtypes
df.loc[df['TotalCharges']==' ','TotalCharges'] = np.nan

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df.loc[df['TotalCharges'].isnull(),'TotalCharges'] = df['TotalCharges'].mean()
df.describe()
df_dummy = pd.get_dummies(df.drop(columns=['customerID']), prefix=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod', 'Churn'])
df_dummy.columns
df_dummy=df_dummy.drop(columns=[x for x in df_dummy.columns if x[-3:]=='_No'])
df_dummy['Churn_Yes'].value_counts()
df_dummy['MonthlyCharges'] = scale(df_dummy['MonthlyCharges'])

df_dummy['TotalCharges'] = scale(df_dummy['TotalCharges'])

df_dummy['tenure'] = scale(df_dummy['tenure'])
df_dummy[['MonthlyCharges','TotalCharges','tenure']].describe()
clf = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(df_dummy.drop(columns=['Churn_Yes']), df_dummy['Churn_Yes'], test_size=0.3)

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
model = Sequential(

    [  

        Dense(16, activation="relu", input_dim=df_dummy.shape[1]-1),

        Dense(8, activation="relu"),

        Dense(1,  activation='sigmoid')

    ]

)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=10)
pred = model.predict(x_test)

pred = [1 if x[0]>=0.5 else 0 for x in pred]

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
dropout_rate = [0.0, 0.2,0.4, 0.6, 0.8]

n_neurons = [(32,16,8), (10,5), (16,8,4)]

batch_size = [20, 60, 100]

epochs = [10, 50, 100]
def create_model(dropout_rate, n_neurons):

    model = Sequential()

    model.add(Dense(n_neurons[0], activation="relu", input_dim=df_dummy.shape[1]-1))

    for i in range(1,len(n_neurons)):

        print(n_neurons[i])

        model.add(Dense(n_neurons[i], activation="relu"))

    model.add(Dropout(dropout_rate))

    model.add(Dense(1,  activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# reference for this part https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

model = KerasClassifier(build_fn=create_model, verbose=0)

param_grid = dict(dropout_rate=dropout_rate, n_neurons=n_neurons, batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(x_train, y_train)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))