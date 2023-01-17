# Importing all the necessary packages, modules, tools, etc.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use("ggplot")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# Reading the files with the training data
train = pd.read_csv('../input/capstone-project-training-data/train_values.csv')
tlabel = pd.read_csv('../input/capstone-project-training-labels/train_labels.csv')
# Summary stats of the training data
print(train.describe())
print(tlabel.describe())
# A histogram of damage_grade, the label of this dataset
fig = plt.figure(figsize=(9, 6))
ax = fig.gca()
plt.hist(tlabel['damage_grade'],)
ax.set_title('Histogram of damage_grade')
ax.set_xlabel('damage_grade')
ax.set_ylabel('Number of buildings')
plt.show()
# Listing the numeric, non-binary features and getting their summary stats
numeric_cols = ['count_floors_pre_eq', 'age', 'area', 'height', 'count_families']
print(train[numeric_cols].describe())
for col in numeric_cols:
    print(train[col].nunique())

# Joining the label and features
train2 = train = train.set_index('building_id').join(tlabel.set_index('building_id')).reset_index()
#Getting a general outline of the joined dataset
train2.head()
train2.dtypes
# Getting the correlation values between the label and the numeric features
numeric_cols2 = ['count_floors_pre_eq', 'age', 'area', 'height', 'count_families', 'damage_grade']
train2[numeric_cols2].corr()
# Histograms of categorical features
cols = train2.columns.tolist()[:-1]
for col in cols:
    if(train2[col].dtype not in [np.int64, np.int32, np.float64]):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        plt.hist(train2[col])
        ax.set_title('Number of building types by ' + str(col))
        ax.set_xlabel(col)
        ax.set_ylabel('Number of buildings')
        plt.show()
# Histograms of numeric features
cols = train2.columns.tolist()[:-1]
for col in cols:
    if(train2[col].dtype in [np.int64, np.int32, np.float64]):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        plt.hist(train2[col])
        ax.set_title('Histogram of ' + str(col))
        ax.set_xlabel(col)
        ax.set_ylabel('Number of buildings')
        plt.show()
# Creating a function to draw histograms of subsets of the data, which are based on the superstructure of the buildings
def damage_grade_hist(df, superstructure):
    print(df.head())
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    plt.hist(df['damage_grade'],)
    ax.set_title('Histogram of damage_grade for buildings that have ' + superstructure)
    ax.set_xlabel('damage_grade')
    ax.set_ylabel('Number of buildings')
    plt.show()
# Creating a subset of buildings with cement mortar and brick superstructure
cemented_buildings = train2[train2['has_superstructure_cement_mortar_brick'] == 1]
cemented_buildings = pd.DataFrame(cemented_buildings)
# Calling the damage_grade_hst function on subset No.1
damage_grade_hist(cemented_buildings, 'cement mortar and brick superstructure')
# Creating a subset of buildings with cement mortar and stone superstructure
cemented_buildings2 = train2[train2['has_superstructure_cement_mortar_stone'] == 1]
cemented_buildings2 = pd.DataFrame(cemented_buildings2)
# Calling the damage_grade_hist function on subset No.2
damage_grade_hist(cemented_buildings2, 'cement mortar and stone superstructure')
# Creating a subset of buildings with non_engineered reinforced concrete superstructure
non_engineered_rc_buildings = train2[train2['has_superstructure_rc_non_engineered'] == 1]
non_engineered_rc_buildings = pd.DataFrame(non_engineered_rc_buildings)
# Calling the damage_grade_hist function on subset No.3
damage_grade_hist(non_engineered_rc_buildings, 'non_engineered reinforced concrete superstructure')
# Creating a subset of buildings with engineered reinforced concrete superstructure
engineered_rc_buildings = train2[train2['has_superstructure_rc_engineered'] == 1]
engineered_rc_buildings = pd.DataFrame(engineered_rc_buildings)
# Calling the damage_grade_hist function on subset No.4
damage_grade_hist(engineered_rc_buildings, 'engineered reinforced concrete superstructure')
# Creating boxplots of damage_grade by each categorical feature
cols = train2.columns.tolist()[:-1]
for col in cols:
    if(train2[col].dtype not in [np.int64, np.int32, np.float64]):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        train2.boxplot( column = 'damage_grade', ax = ax, by = col)
        ax.set_xlabel(col)
        ax.set_ylabel('damage_grade')
# Creating boxplots of each numeric, non_binary feature by damage_grade
for col in numeric_cols:
    fig = plt.figure(figsize = (6,6))
    fig.clf()
    ax = fig.gca() 
    train2.boxplot(column = [col], ax = ax, by = 'damage_grade')
    ax.set_ylabel(col)
# Defining a function that creates conditioned histograms
def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ## Loop over the list of columns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
    return grid_col
#Defining a function that creates barplots of each categorical feature by damage_grade
def damage_barplot(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    cols = df.columns.tolist()[:-1]
    for col in cols:
        temp2 = df.ix[df['damage_grade'] == 3, col].value_counts()
        temp1 = df.ix[df['damage_grade'] == 2, col].value_counts()
        temp0 = df.ix[df['damage_grade'] == 1, col].value_counts() 
        if(df.ix[:, col].dtype not in [np.int64, np.int32, np.float64]):    
            ylim = [0, max(max(temp2), max(temp1), max(temp0))]
            fig = plt.figure(figsize = (12,6))
            fig.clf()
            ax2 = fig.add_subplot(1, 3, 1)
            ax1 = fig.add_subplot(1, 3, 2)
            ax0 = fig.add_subplot(1, 3, 3)
            temp2.plot(kind = 'bar', ax = ax2, ylim = ylim)
            ax2.set_title('Values of ' + col + '\n for damage grade 3')
            temp1.plot(kind = 'bar', ax = ax1, ylim = ylim)
            ax1.set_title('Values of ' + col + '\n for damage grade 2')
            temp0.plot(kind = 'bar', ax = ax0, ylim = ylim)
            ax0.set_title('Values of ' + col + '\n for damage grade 1')
    return('Done')
# Calling the damage_barplot function on the training dataset
damage_barplot(train2)
plt.show()
# Creating conditioned histograms of numeric features
cols = train2.columns.tolist()[:-1]
plot_cols = pd.DataFrame()
for col in cols:
        if(train2[col].dtype in [np.int64, np.int32, np.float64]): 
            plot_cols = pd.concat([plot_cols, train2[col]], axis = 1)
cond_hists(train2, plot_cols, 'damage_grade')
plt.show()
# Encoding categorical features
train2 = pd.get_dummies(train2)
# Listing columns to see changes in the dataset
print(train2.columns)
# Separating the label and the features for machine learning
y = train2['damage_grade'].copy()
X = train2.drop(['damage_grade'], axis = 1)
# Checking for issues
print(y.describe())
print(X.describe())
# Train-test split of the training data
# The test set does not have labels assigned becaue the goal of the competition was to predict labels
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)
skf = StratifiedKFold(n_splits=10, random_state = 7)
# Creating machine learning model and fitting to the training set
model = OneVsRestClassifier(estimator = RandomForestClassifier(n_estimators = 137, criterion = 'entropy', max_features = 54, min_samples_split = 2, min_samples_leaf = 1, random_state = 7))
model.fit(X_train, y_train)
# Predicting for the test set
y_pred = model.predict(X_test)
# Scoring the results using micro-averaged F1-score, which was the target metric for the competition
print(f1_score(y_test, y_pred, average = 'micro'))
# Cross-validating results for more clarity
y_pred2 = cross_val_predict(model, X_test, y_test)
# Using more merics to assess the results
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))
# Fitting the data for the whole training dataset and scoring it
model.fit(X, y)
scores = cross_val_score(model, X, y, cv = skf)
print(scores.mean())
# Importing the test set and preparing it
test = pd.read_csv('../input/capstone-project-test-data/test_values.csv')
test = pd.get_dummies(test)
# Predicting for the test set and checking it for any errors
prediction = model.predict(test)
print(prediction)
# Converting the predictions into a pandas series
prediction = pd.Series(prediction)
# Creating the submission in the predefined format
submission = pd.DataFrame({'building_id' : test3['building_id'], 'damage_grade': prediction})
# Getting a feel for the results, checking to see if it is similar to the label of the training set
print(submission.head())
plt.hist(prediction)
plt.show()
# Creating the submission file
submission.to_csv('submission.csv', index = False)