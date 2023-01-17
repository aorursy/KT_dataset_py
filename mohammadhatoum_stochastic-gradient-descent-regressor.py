import numpy as np # linear algebra

import pandas as pd # data processing

df = pd.read_csv("../input/data.csv")
print(df.columns.values)
print(df.info())
df.head()
print(df['date'].unique())
print(df['timestamp'].unique)
df = df.drop(['date','timestamp'],axis=1)
# Extract the training and test data

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

from sklearn.model_selection import train_test_split

X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# View the shape (structure) of the data

print(f"Training features shape: {X_train_original.shape}")

print(f"Testing features shape: {X_test_original.shape}")

print(f"Training label shape: {y_train.shape}")

print(f"Testing label shape: {y_test.shape}")
from sklearn.preprocessing import StandardScaler

# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train_original)

X_train = scaler.transform(X_train_original)

X_test = scaler.transform(X_test_original)
# Import library

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

matplotlib.style.use('ggplot')



# Specify number of plot to be displayed and the figure size

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))



# Set a title and plot the data

ax1.set_title('Before Scaling')

ax1.plot(X_train_original[:60,3])



ax2.set_title('After Standard Scaler')

ax2.plot(X_train[:60,3])



# Display the graph

plt.show()
# Establish a model

from sklearn.linear_model import SGDRegressor

sgd_huber=SGDRegressor(alpha=0.01, learning_rate='optimal', loss='huber', 

    penalty='elasticnet')  
sgd_huber.fit(X_train, y_train)
y_pred_lr = sgd_huber.predict(X_test)  # Predict labels
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# The mean squared error

print(f"Mean squared error: {round( mean_squared_error(y_test, y_pred_lr),3)}")

# Explained variance score: 1 is perfect prediction

print(f"Variance score: {round(r2_score(y_test, y_pred_lr),3)}")

# Mean Absolute Error

print(f"Mean squared error: { round(mean_absolute_error(y_test, y_pred_lr),3)}")
# Try different parameters

sgd_l2 = SGDRegressor(alpha=0.01,learning_rate='optimal', loss='squared_loss',

             penalty='l2')



sgd_l2.fit(X_train, y_train)

print(f"Score on training set {round(sgd_l2.score(X_train, y_train),3)}")



y_pred_lr = sgd_l2.predict(X_test)  # Predict labels



from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# The mean squared error

print(f"Mean squared error: {round( mean_squared_error(y_test, y_pred_lr),3)}")

# Explained variance score: 1 is perfect prediction

print(f"Variance score: {round(r2_score(y_test, y_pred_lr),3)}")

# Mean Absolute Error

print(f"Mean squared error: { round(mean_absolute_error(y_test, y_pred_lr),3)}")
# Establish a model

model = SGDRegressor(learning_rate='optimal',penalty='l2')

from sklearn.model_selection import GridSearchCV

# Grid search - this will take about 1 minute.

param_grid = {

    'alpha': 10.0 ** -np.arange(1, 7),

    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],

}

clf = GridSearchCV(model, param_grid,cv=5)

clf.fit(X_train, y_train)

print(f"Best Score: {round(clf.best_score_,3)}" )

print(f"Best Estimator: {clf.best_estimator_}" )

print(f"Best Params: {clf.best_params_}" )