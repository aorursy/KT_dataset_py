# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



# set seed for reproducibility

np.random.seed(0)



#Data

tumor_data = pd.read_csv('../input/data.csv')

tumor_data.sample(5)
print("The dataset has %d rows and %d columns" % (tumor_data.shape[0], tumor_data.shape[1]))
tumor_data.describe(include='all')
tumor_data.dtypes
tumor_data['Unnamed: 32'].sample(8)
missing_values = tumor_data['Unnamed: 32'].isnull().sum()

number_of_rows = tumor_data['Unnamed: 32'].shape[0]

if missing_values == number_of_rows:

    print('The whole \'Unnamed: 32\' column has empty values.')

else:

    print('There are non-empty values in the \'Unnamed: 32\' column.')
tumor_data.drop(['Unnamed: 32'], axis= 1, inplace = True)

tumor_data.columns
tumor_data.isna().sum()
tumor_data['diagnosis'].value_counts()
tumor_data['target'] = tumor_data['diagnosis'].replace({'B': 1, 'M': 0})

# Let's show if the convertion was rightly done

tumor_data[['id', 'diagnosis', 'target']].sample(5)
tumor_data.drop(['diagnosis'], axis = 1, inplace = True)

tumor_data.columns
tumor_data.drop(['id'], axis = 1, inplace=True)

tumor_data.columns
sns.pairplot(

    tumor_data,

    vars = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean'],

    hue = 'target'

)
# First, we make sure that the graphic is crearly visible

plt.figure(figsize = (30, 20))

# And now, draw the heatmap

sns.heatmap(tumor_data.corr(), cmap = "RdBu_r")
# Independent variables

X = tumor_data.drop(['target'], axis = 1)

# Dependent variable

Y = tumor_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(X_train, Y_train)
Y_predicted = svc_model.predict(X_test)

Y_predicted
cm = confusion_matrix(Y_test, Y_predicted)

sns.heatmap(cm, annot = True, cmap="Blues")
print(classification_report(Y_test, Y_predicted))
tumor_data.describe()
min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train)/range_train

X_train_scaled.describe()
fig = plt.figure(figsize = (20, 5))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

ax1.set_title('Values without normalization')

ax2.set_title('Values with normalization')

sns.scatterplot(x = X_train['texture_mean'], y = X_train['area_mean'], hue = Y_train, ax = ax1)

sns.scatterplot(x = X_train_scaled['texture_mean'], y = X_train_scaled['area_mean'], hue = Y_train, ax = ax2)
# Train the model again

svc_model.fit(X_train_scaled, Y_train)

# Create scaled test data

X_test_scaled = (X_test - X_test.min())/(X_test - X_test.min()).max()

# Calculate new predictions

Y_predicted = svc_model.predict(X_test_scaled)

# Draw confusion matrix

cm = confusion_matrix(Y_test, Y_predicted)

sns.heatmap(cm, annot = True, cmap='Blues')
print(classification_report(Y_test, Y_predicted))
# We can automate the refinement of C and gamma using the GridSearchCV library

from sklearn.model_selection import GridSearchCV

C_values =  [0.1, 1, 10, 100, 1000]

gamma_values = [1, 0.1, 0.01, 0.001]

grid = GridSearchCV(SVC(), {'C': C_values, 'gamma': gamma_values, 'kernel': ['rbf']}, refit = True, verbose = 4)

# Find best pair of C and gamma values

grid.fit(X_train_scaled, Y_train)

grid.best_params_
# We can use the optimized grid object directly to get predictions

grid_predicted = grid.predict(X_test_scaled)



cm = confusion_matrix(Y_test, grid_predicted)

sns.heatmap(cm, annot = True, cmap = 'Blues')

print(classification_report(Y_test, grid_predicted))