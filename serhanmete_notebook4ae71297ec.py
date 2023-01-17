# Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
%matplotlib inline
sns.set_palette('colorblind')
sns.set_style('whitegrid')
plt.rc('font', size = 15)
# Filter warning - use this with caution!
import warnings
warnings.filterwarnings('ignore')
# Let's take a look at the training data
training_df = pd.read_csv('../input/train.csv')
# Here is some information
training_df.info()
# Here is a quick summary
training_df.describe()
# Let's make a count plot where we see the survival for each sex
fig, ax = plt.subplots(1, 1, figsize = (15,5))
sns.countplot(x = 'Sex', hue = 'Survived', data = training_df, ax = ax)
# Now let's make a cluster map
sns.clustermap(training_df.corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')
# Let's make a count plot where we see the survival for each sex
fig, ax = plt.subplots(1, 1, figsize = (15,5))
sns.countplot(x = 'Embarked', hue = 'Survived', data = training_df, ax = ax)
val1 = training_df[(training_df['Embarked']=='C')&(training_df['Survived']==0)]['PassengerId'].count()
std1 = math.sqrt(val1) # Assuming Poisson - see below
val2 = training_df[(training_df['Embarked']=='C')&(training_df['Survived']==1)]['PassengerId'].count()
std2 = math.sqrt(val2) # Assuming Poisson - see below
print('A-B = %.2f +/- %.2f'%(val1-val2,math.sqrt(std1**2+std2**2))) # Simple uncorrelated error propagation for A-B
# Now refine the dataframe
training_df_refined = training_df.drop(['PassengerId','Name','Ticket','Age','Cabin'], axis = 1)
# Deal w/ categorical data
features = ['Sex','Embarked']
training_df_final = pd.get_dummies(training_df_refined, columns = features, drop_first = True)
training_df_final.head()
# Let's load the necessary libraries
from sklearn.preprocessing import StandardScaler
# Define a scaler, fit, and transform
sc = StandardScaler() # use the default configuration
sc.fit(X = training_df_final.drop('Survived', axis = 1))
scaled_data = sc.transform(X = training_df_final.drop('Survived', axis = 1))
# Put the scaled data into a new dataframe
training_df_final_scaled = pd.DataFrame(data = scaled_data, columns = training_df_final.columns[1:] ) 
training_df_final_scaled.head()
# Split our datasets
from sklearn.model_selection import train_test_split
# Now prepare the data
X = training_df_final.drop('Survived', axis = 1).values
y = training_df_final['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 101)
# Let's load Keras and build a sequential dense model
from keras.models import Sequential
from keras.layers import Dense
# A sequential model where we stack layers on top of each other
model = Sequential()
model.add(Dense(units = 10, activation='relu', input_dim = 7))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(units = 1, activation='sigmoid'))
# Now compile the method.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Now fit the model
model.fit(X_train, y_train, epochs = 30, batch_size = 20, verbose = 0)
# Load some useful functions
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_predict = model.predict_classes(X_test) # Here we predict the classes
# Let's look at the accuracy score
print(accuracy_score(y_test,y_predict))
# Let's look at the confusion matrix
print(confusion_matrix(y_test,y_predict))
# Let's look at the classification report
print(classification_report(y_test,y_predict))