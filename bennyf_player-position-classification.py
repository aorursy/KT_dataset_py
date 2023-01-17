# Import required packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# import pydotplus

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  
# Import the dataset as a dataframe

# Dataset can be found at https://www.kaggle.com/karangadiya/fifa19

# Dataset is also included in assignment zip file

df = pd.read_csv('/kaggle/input/fifa19/data.csv', sep=',')
# Change settings to display all columns

pd.set_option('display.max_columns', None)



# Preview the dataframe

df.head()
# For this application, we only want the players name and ID, along with their skill scores and position.

# Drop unwanted columns for the purpose of this application

# Note we dropped the Goalkeeping skills attributes, this would make it too easy for our model. 

# Plus in a real world scenario, these skills would be hard to measure for ALL players.

df = df.drop(['Unnamed: 0', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential', 'Club', 'Club Logo', 'Value',

              'Wage','Special', 'Preferred Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate',

              'Body Type', 'Real Face', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height',

              'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',

              'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'GKDiving', 'GKHandling', 

              'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'], axis = 1)

# Check the new dataframe

df.head()
# Check how many players in the dataset

df.shape
# Let's see how many Nulls we're dealing with

df.isnull().sum()
# We can see there is about 60 possible players with NA value

# Because we already have over 18,000 players in the dataset, we will just delete them

df = df.dropna()

df.isnull().sum()
# Investigate the Position attribute

df.Position.value_counts()
# Convert the current 27 positions into 4 main position disciplines, Attacker, Midfielder, Defender, Goalkeeper

att = dict.fromkeys(['ST', 'LW', 'RW', 'LS', 'RS', 'CF', 'RF', 'LF'], 'Attacker')

mid = dict.fromkeys(['CM', 'RM', 'LM', 'CAM', 'CDM', 'LCM', 'RCM', 'RDM', 'LDM', 'RAM', 'LAM'], 'Midfielder')

dfnc = dict.fromkeys(['CB', 'LB', 'RB', 'RCB', 'LCB', 'RWB', 'LWB' ], 'Defender')

df.Position.replace('GK', 'Goalkeeper', inplace=True)

df.Position.replace(att, inplace=True)

df.Position.replace(mid, inplace=True)

df.Position.replace(dfnc, inplace=True)



# Confirm the changes

df.Position.value_counts()
# Check for duplicate players using the players unique player ID

df.duplicated('ID').sum()
# Create a pie chart showing the percentage of each position represented in the dataset

df.Position.value_counts().plot(kind = 'pie',

                                autopct = '%0.1f%%',

                                shadow = True,

                                cmap = 'Set3'

                                )

plt.title('Position Representation\n', fontsize = 16 )

plt.xlabel('')

plt.ylabel('')

plt.axis('equal')

plt.show()
# Create a histogram for each column and also return min, max, mean and median.

# Each of these columns should range between 0 and 100.

# The histogram and the min, max, mean values should provide us a quick way to validate the data.

# Use a for loop to cycle through the attributes

for column in df.iloc[:,3:]:

    sns.distplot(df[column], kde = False, bins = 20)

    plt.title(str(df[column].name) + " Histogram", fontsize = 14)

    plt.ylabel("Count")

    plt.show()

    print (df[column].name) 

    print (df[column].describe())

    print ("\n\n")
# Create a scatterplot

sns.scatterplot(x = "Reactions", 

                y = "Balance", 

                hue = "Position", 

                data = df,

                )

plt.title('Balance Vs Reactions by Position', fontsize = 14)

plt.show()
# Create a scatterplot

sns.scatterplot(x = "Strength", 

                y = "Jumping", 

                hue = "Position", 

                data = df,

                )

plt.title('Jumping Vs Strength by Position', fontsize = 14)

plt.show()
# Create a scatterplot

sns.scatterplot(x = "Finishing", 

                y = "HeadingAccuracy", 

                hue = "Position", 

                data = df,

                )

plt.title('HeadingAccuracy Vs Finishing by Position', fontsize = 14)

plt.show()
# Create a scatterplot

sns.scatterplot(x = "Interceptions", 

                y = "Marking", 

                hue = "Position", 

                data = df,

                )

plt.title('Interceptions Vs Marking by Position', fontsize = 14)

plt.show()
# Create a scatterplot

sns.scatterplot(x = "SlidingTackle", 

                y = "StandingTackle", 

                hue = "Position", 

                data = df,

                )

plt.title('StandingTackle Vs SlidingTackle by Position', fontsize = 14)

plt.show()
# Create a scatterplot

sns.scatterplot(x = "SlidingTackle", 

                y = "Finishing", 

                hue = "Position", 

                data = df,

                )

plt.title('Finishing Vs SlidingTackle by Position', fontsize = 14)

plt.show()
# Create box plots for 2 key skill ratings

# Outliers removed for cleaner plots



sns.boxplot(x = 'Position',

            y = 'Finishing', 

            data = df,

            width = 0.2,

            fliersize = 0

            )

plt.title("Finishing by Position", fontsize = 14)

plt.ylabel("Rating")

plt.show()

    

sns.boxplot(x = 'Position',

            y = 'SlidingTackle', 

            data = df,

            width = 0.2,

            fliersize = 0

            )

plt.title("SlidingTackle by Position", fontsize = 14)

plt.ylabel("Rating")

plt.show()
# Create a subset based on 4 specialised skills

# Plot the 4 skills in a kde matrix to try an identify clusters

special_skills = df[['Finishing', 'HeadingAccuracy','SlidingTackle', 'StandingTackle']]

g = sns.PairGrid(special_skills)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels = 20)

plt.show()
# Create a subset based on 4 general skills

# Plot the 4 skills in a kde matrix to try an identify clusters

general_skills = df[['Balance', 'Reactions', 'Jumping', 'Strength']]

g = sns.PairGrid(general_skills)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels = 20)

plt.show()
# Check that our features ie skill columns are ints or floats

df.dtypes
# Nominate the features

feature_cols = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',

                'LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower',

                'Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision',

                'Penalties','Composure','Marking','StandingTackle','SlidingTackle']



# Assign the feature data

X = df[feature_cols]



# Assign the outcomes

y = df.Position



# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1984)
# Identify the best value for max_depth

error_rate = []



# Set the range of potential max_depth

# Run clf for each max_depth in the range

for i in range(1,6):

    clf = DecisionTreeClassifier(criterion = 'gini', # Default gini

                                 max_features = None, # Default None

                                 max_depth = i, # Default None

                                 min_samples_split = 2, # Default 2

                                 min_samples_leaf = 1 # Default 1 

                                 )

    fit = clf.fit(X_train, y_train)

    pred_i = fit.predict(X_test)



    # Record the error value 

    error_rate.append(np.mean(pred_i != y_test))

    

# Plot the error rates and choose a max_depth value

plt.plot(range(1,6), 

         error_rate,

         color = 'blue', 

         linestyle = 'dashed',

         markerfacecolor = 'red',

         marker = 'o',

         markersize = 5)



plt.title('Error Rate vs. max_depth')

plt.xlabel('max_depth')

plt.ylabel('Error Rate')

plt.show()
# Identify the best value for min_samples_split

error_rate = []



# Set the range of potential min_samples_split

# Run clf for each mmin_samples_split in the range

for i in range(10,210,10):

    clf = DecisionTreeClassifier(criterion = 'gini', # Default gini

                                 max_features = None, # Default None

                                 max_depth = 5, # Default None

                                 min_samples_split = i, # Default 2

                                 min_samples_leaf = 1 # Default 1 

                                 )

    fit = clf.fit(X_train, y_train)

    pred_i = fit.predict(X_test)



    # Record the error value 

    error_rate.append(np.mean(pred_i != y_test))

    

# Plot the error rates and choose a min_samples_split value

plt.plot(range(10,210,10), 

         error_rate,

         color = 'blue', 

         linestyle = 'dashed',

         markerfacecolor = 'red',

         marker = 'o',

         markersize = 5)



plt.title('Error Rate vs. min_samples_split')

plt.xlabel('min_samples_split (x10)')

plt.ylabel('Error Rate')

plt.show()
# Identify the best value for min_samples_leaf

error_rate = []



# Set the range of potential min_samples_leaf

# Run clf for each min_samples_leaf in the range

for i in range(1,51):

    clf = DecisionTreeClassifier(criterion = 'gini', # Default gini

                                 max_features = None, # Default None

                                 max_depth = 5, # Default None

                                 min_samples_split = 20, # Default 2

                                 min_samples_leaf = i # Default 1 

                                 )

    fit = clf.fit(X_train, y_train)

    pred_i = fit.predict(X_test)



    # Record the error value 

    error_rate.append(np.mean(pred_i != y_test))

    

# Plot the error rates and choose a min_samples_leaf value

plt.plot(range(1,51), 

         error_rate,

         color = 'blue', 

         linestyle = 'dashed',

         markerfacecolor = 'red',

         marker = 'o',

         markersize = 5)



plt.title('Error Rate vs. min_samples_leaf')

plt.xlabel('min_samples_leaf')

plt.ylabel('Error Rate')

plt.show()
# Identify the best value for max_features

error_rate = []



# Set the range of potential max_features

# Run clf for each max_features in the range

for i in range(1,30):

    clf = DecisionTreeClassifier(criterion = 'gini', # Default gini

                                 max_features = i, # Default None

                                 max_depth = 5, # Default None

                                 min_samples_split = 20, # Default 2

                                 min_samples_leaf = 5 # Default 1 

                                 )

    fit = clf.fit(X_train, y_train)

    pred_i = fit.predict(X_test)



    # Record the error value 

    error_rate.append(np.mean(pred_i != y_test))

    

# Plot the error rates and choose a max_features value

plt.plot(range(1,30), 

         error_rate,

         color = 'blue', 

         linestyle = 'dashed',

         markerfacecolor = 'red',

         marker = 'o',

         markersize = 5)



plt.title('Error Rate vs. max_features')

plt.xlabel('max_features')

plt.ylabel('Error Rate')

plt.show()
# Build the Decision Tree Model

clf = DecisionTreeClassifier(criterion = 'gini', # Default gini

                             max_features = None, # Default None

                             max_depth = None, # Default None

                             min_samples_split = 2, # Default 2

                             min_samples_leaf = 1 # Default 1 

                             )



# Train the model on the training set

fit = clf.fit(X_train, y_train)



# Test the model on the test set

y_pred = fit.predict(X_test)



# Build a confusion matrix

cm = confusion_matrix(y_test, y_pred)



# Cross Validate the model with 10 folds

cv_scores = cross_val_score(clf, X, y, cv=10)



# Set the accuracy to a variable

acc = metrics.accuracy_score(y_test, y_pred)



# Set the error rate to a variable

err = np.mean(y_pred != y_test)



# Set the cross validation mean score to a variable

cvm = np.mean(cv_scores)



# Calculate the differece between the Accuracy and CV Accuracy

dif = acc-cvm



# Determine if model is underfitted or overfitted

if dif < 0:

    underover = "Underfitted"

else:

    underover = "Overfitted"



# Set the model node count to a variable

nds = clf.tree_.node_count



print ("Model Accuracy: ",str(round(acc*100,2))+"%")

print ("Cross Validation Accuracy: ",str(round(cvm*100,2))+"%")

print ("Model Fitting: ",str(round(dif*100,2))+"%",underover)

print ("Number of Nodes in Model: ",nds)

print ("Error Rate: ",str(round(err*100, 2))+"%")

print ("\n\n\nConfusion Matrix: \n\n",cm)

print ("\n\n\nClassification Report:\n\n",classification_report(y_test, y_pred))
# Build the Decision Tree Model

clf = DecisionTreeClassifier(criterion = 'gini', # Default gini

                             max_features = 23, # Default None

                             max_depth = 5, # Default None

                             min_samples_split = 20, # Default 2

                             min_samples_leaf = 5 # Default 1 

                             )



# Train the model on the training set

fit = clf.fit(X_train, y_train)



# Test the model on the test set

y_pred = fit.predict(X_test)



# Build a confusion matrix

cm = confusion_matrix(y_test, y_pred)



# Cross Validate the model with 10 folds

cv_scores = cross_val_score(clf, X, y, cv=10)



# Set the accuracy to a variable

acc = metrics.accuracy_score(y_test, y_pred)



# Set the error rate to a variable

err = np.mean(y_pred != y_test)



# Set the cross validation mean score to a variable

cvm = np.mean(cv_scores)



# Calculate the differece between the Accuracy and CV Accuracy

dif = acc-cvm



# Determine if model is underfitted or overfitted

if dif < 0:

    underover = "Underfitted"

else:

    underover = "Overfitted"



# Set the model node count to a variable

nds = clf.tree_.node_count



print ("Model Accuracy: ",str(round(acc*100,2))+"%")

print ("Cross Validation Accuracy: ",str(round(cvm*100,2))+"%")

print ("Model Fitting: ",str(round(dif*100,2))+"%",underover)

print ("Number of Nodes in Model: ",nds)

print ("Error Rate: ",str(round(err*100, 2))+"%")

print ("\n\n\nConfusion Matrix: \n\n",cm)

print ("\n\n\nClassification Report:\n\n",classification_report(y_test, y_pred))
# # Draw the model

# dot_data = StringIO()

# export_graphviz(fit, 

#                 out_file = dot_data, 

#                 filled = True, 

#                 rounded = True,

#                 special_characters = True,

#                 feature_names = feature_cols, 

#                 class_names = ['Attacker', 'Defender', 'Goalkeeper', 'Midfielder']

#                ) 

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# graph.write_png('fifa-clf.png')

# Image(graph.create_png())
# Nominate the features

feature_cols = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',

                'LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower',

                'Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision',

                'Penalties','Composure','Marking','StandingTackle','SlidingTackle']



# Assign the feature data

X = df[feature_cols]



# Assign the outcomes

y = df.Position



# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1984)
# Identify the best value for K using the elbow method

# The elbow method plots the error rate of a range of K values

error_rate = []



# Set the range of potential K values

# Run KNN for each K in the range

for i in range(1,32,2):

   

    knn = KNeighborsClassifier(n_neighbors = i)

    fit = knn.fit(X_train, y_train)

    pred_i = fit.predict(X_test)

    # Record the error value 

    error_rate.append(np.mean(pred_i != y_test))



    

# Plot the error rates and choose a K value

plt.plot(range(1,32,2), 

         error_rate,

         color = 'blue', 

         linestyle = 'dashed',

         markerfacecolor = 'red',

         marker = 'o',

         markersize = 5)



plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.show()
error_rate = []



# Set the range of potential P values

# Run KNN for each P in the range

for i in range(1,3):

   

    knn = KNeighborsClassifier(n_neighbors = 7, p = i)

    fit = knn.fit(X_train, y_train)

    pred_i = fit.predict(X_test)

    # Record the error value 

    error_rate.append(np.mean(pred_i != y_test))



    

# Plot the error rates and choose a P value

plt.plot(range(1,3), 

         error_rate,

         color = 'blue', 

         linestyle = 'dashed',

         markerfacecolor = 'red',

         marker = 'o',

         markersize = 5)



plt.title('Error Rate vs. P Value')

plt.xlabel('P')

plt.ylabel('Error Rate')

plt.show()
# Build the Decision Tree Model

knn = KNeighborsClassifier(n_neighbors = 5, # Default 5

                           weights = 'uniform', # Default uniform

                           metric = 'minkowski', # Default minkowski

                           p = 2 # Default 2

                          )



# Train the model on the training set

fit = knn.fit(X_train, y_train)



# Test the model on the test set

y_pred = fit.predict(X_test)



# Build a confusion matrix

cm = confusion_matrix(y_test, y_pred)



# Cross Validate the model with 10 folds

cv_scores = cross_val_score(knn, X, y, cv=10)



# Set the accuracy to a variable

acc = metrics.accuracy_score(y_test, y_pred)



# Set the error rate to a variable

err = np.mean(y_pred != y_test)



# Set the cross validation mean score to a variable

cvm = np.mean(cv_scores)



# Calculate the differece between the Accuracy and CV Accuracy

dif = acc-cvm



# Determine if model is underfitted or overfitted

if dif < 0:

    underover = "Underfitted"

else:

    underover = "Overfitted"



print ("Model Accuracy: ",str(round(acc*100,2))+"%")

print ("Cross Validation Accuracy: ",str(round(cvm*100,2))+"%")

print ("Model Fitting: ",str(round(dif*100,2))+"%",underover)

print ("Error Rate: ",str(round(err*100, 2))+"%")

print ("\n\n\nConfusion Matrix: \n\n",cm)

print ("\n\n\nClassification Report:\n\n",classification_report(y_test, y_pred))
# Build the Decision Tree Model

knn = KNeighborsClassifier(n_neighbors = 7, # Default 5

                           weights = 'uniform', # Default uniform

                           metric = 'minkowski', # Default minkowski

                           p = 2 # Default 2

                          )



# Train the model on the training set

fit = knn.fit(X_train, y_train)



# Test the model on the test set

y_pred = fit.predict(X_test)



# Build a confusion matrix

cm = confusion_matrix(y_test, y_pred)



# Cross Validate the model with 10 folds

cv_scores = cross_val_score(knn, X, y, cv=10)



# Set the accuracy to a variable

acc = metrics.accuracy_score(y_test, y_pred)



# Set the error rate to a variable

err = np.mean(y_pred != y_test)



# Set the cross validation mean score to a variable

cvm = np.mean(cv_scores)



# Calculate the differece between the Accuracy and CV Accuracy

dif = acc-cvm



# Determine if model is underfitted or overfitted

if dif < 0:

    underover = "Underfitted"

else:

    underover = "Overfitted"



print ("Model Accuracy: ",str(round(acc*100,2))+"%")

print ("Cross Validation Accuracy: ",str(round(cvm*100,2))+"%")

print ("Model Fitting: ",str(round(dif*100,2))+"%",underover)

print ("Error Rate: ",str(round(err*100, 2))+"%")

print ("\n\n\nConfusion Matrix: \n\n",cm)

print ("\n\n\nClassification Report:\n\n",classification_report(y_test, y_pred))