from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
from IPython.display import Image
Image(filename='../input/titanic-images/titanic.png')
Image(filename='../input/titanic-images/thecause.png')
Image(filename='../input/titanic-images/timeline.png')
Image(filename='../input/titanic-images/passengers.png')
# This first set of packages include Pandas, for data manipulation, numpy for mathematical computation
# and matplotlib & seaborn, for visualisation.
import pandas as pd
import numpy as np
from numpy import sort
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')
print('Data Manipulation, Mathematical Computation and Visualisation packages imported!')

# Next, these packages are from the scikit-learn library, including the algorithms I plan to use.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
print('Algorithm packages imported!')

# These packages are also from scikit-learn, but these will be used for model selection and cross validation.
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
print('Model Selection packages imported!')

# Once again, these are from scikit-learn, but these will be used to assist me during feature reduction.
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
print('Feature Reduction packages imported!')

# Finally, a metrics package from scikit-learn to be used when comparing the results of the cross validated models. 
from sklearn import metrics
print('Cross Validation Metrics packages imported!')

# Set visualisation colours
mycols = ["#76CDE9", "#FFE178", "#9CE995", "#E97A70"]
sns.set_palette(palette = mycols, n_colors = 4)
print('My colours are ready! :)')

# Ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('Future warning will be ignored!')
# Read the raw training and test data into DataFrames.
raw_train = pd.read_csv('../input/titanic/train.csv')
raw_test = pd.read_csv('../input/titanic/test.csv')

# Append the test dataset onto train to allow for easier feature engineering across both datasets.
full = raw_train.append(raw_test, ignore_index=True)

print('Datasets \nfull: ', full.shape, '\ntrain: ', raw_train.shape, '\ntest: ', raw_test.shape)
full.head(3)
full.isnull().sum()
full.describe(include='all')
# Correlation matrix between numerical values (SibSp, Parch, Age and Fare) with Survived 
plt.subplots(figsize=(20, 15))
g = sns.heatmap(full[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "YlGnBu", linewidths = 1)
# Creates a list of all the rows with missing values for age, ordered by Index
missing_age = list(full["Age"][full["Age"].isnull()].index)

# Using a for loop to iterate through the list of missing values, and filling each entry with the imputed value
for i in missing_age:
    age_med = full["Age"].median() # median age for entire dataset
    age_pred = full["Age"][((full['SibSp'] == full.iloc[i]["SibSp"]) & (full['Parch'] == full.iloc[i]["Parch"]) & (full['Pclass'] == full.iloc[i]["Pclass"]))].median() # median age of similar passengers
    if not np.isnan(age_pred):
        full['Age'].iloc[i] = age_pred # if there are similar passengers, fill with predicted median
    else:
        full['Age'].iloc[i] = age_med # otherwise, fill with median of entire dataset

print('Number of missing values for Age: ', full['Age'].isnull().sum())
plt.subplots(figsize=(15, 10))
g = sns.distplot(full["Age"], color="#76CDE9", label="Skewness : %.2f"%(full["Age"].skew()))
g = g.legend(loc="best");
# Crosstab to show the number of unique categories within the Cabin feature
pd.crosstab(full['Sex'], full['Cabin'])
# Cabin
full["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in full['Cabin'] ])
full.head()
#  Filling the missing values in Embarked with 'S'
full['Embarked'].fillna("S", inplace = True)

# full = pd.get_dummies(full, columns = ["Cabin"], prefix="Cabin")
print('Number of missing values for Embarked: ', full['Embarked'].isnull().sum())
# Fill Fare missing values with the median value
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# full = pd.get_dummies(full, columns = ["Cabin"], prefix="Cabin")
print('Number of missing values for Fare: ', full['Fare'].isnull().sum())
plt.subplots(figsize=(15, 10))
g = sns.distplot(full["Fare"], color="#76CDE9", label="Skewness : %.2f"%(full["Fare"].skew()))
g = g.legend(loc="best");
# Apply log to Fare to reduce skewness distribution
full["Fare"] = full["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

plt.subplots(figsize=(15, 10))
g = sns.distplot(full["Fare"], color="#76CDE9", label="Skewness : %.2f"%(full["Fare"].skew()))
g = g.legend(loc="best")
# Age distibution vs Survived
plt.subplots(figsize=(15, 10))
g = sns.kdeplot(full["Age"][(full["Survived"] == 0)], color="#76CDE9",  shade = True)
g = sns.kdeplot(full["Age"][(full["Survived"] == 1)], ax = g, color="#FFDA50", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
# Age distibution vs Survived
plt.subplots(figsize=(15, 10))
g = sns.kdeplot(full["Fare"][(full["Survived"] == 0)], color="#76CDE9",  shade = True)
g = sns.kdeplot(full["Fare"][(full["Survived"] == 1)], ax = g, color="#FFDA50", shade= True)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
# Explore Parch feature vs Survived
g  = sns.factorplot(x = "Parch", y = "Survived", data = full, kind = "bar", palette = mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
# Explore Pclass feature vs Survived
g  = sns.factorplot(x = "Pclass", y = "Survived", data = full, kind = "bar", palette = mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
# Explore Pclass vs Survived by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
# Explore SibSp vs Survived
g = sns.factorplot(x="SibSp", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
plt.subplots(figsize=(15, 10))
g = sns.countplot(x = "Cabin", data = full, palette = mycols)
g = sns.factorplot(y = "Survived", x ="Cabin", data = full, kind = "bar", size = 10, aspect = 1.5,
                   order=['A','B','C','D','E','F','G','T','X'], palette = mycols)
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
# Explore Embarked vs Survived
g = sns.factorplot(x="Embarked", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
htmp = pd.crosstab(full['Embarked'], full['Pclass'])

plt.subplots(figsize=(15, 10))
g = sns.heatmap(htmp, cmap="YlGnBu", vmin = 0, vmax = 500, linewidths = 1, annot = True, fmt = "d");
# Explore Sex vs Survived
g = sns.factorplot(x="Sex", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
sns.factorplot(x="Sex", col = "Pclass", hue = "Survived", data=full, kind="count");
sns.factorplot(x="Sex", y = "Age", hue = "Survived", data=full, kind="violin",
               split = True, size = 10, aspect = 1.5);
sns.factorplot(x="Sex", y="Fare", hue="Survived", col = "Pclass", row="Embarked", data=full, kind="bar");
full.head()
# Creating a new feature by cutting the Age column into 5 bins
full['Age Band'] = pd.cut(full['Age'], 5)

# Now lets see the age bands that pandas has created and how survival is affected by each
full[['Age Band', 'Survived']].groupby(['Age Band'], as_index=False).mean().sort_values(by='Age Band', 
                                                                                          ascending=True)
# Locate and replace values
full.loc[full['Age'] <= 16.136, 'Age'] = 1
full.loc[(full['Age'] > 16.136) & (full['Age'] <= 32.102), 'Age'] = 2
full.loc[(full['Age'] > 32.102) & (full['Age'] <= 48.068), 'Age']   = 3
full.loc[(full['Age'] > 48.068) & (full['Age'] <= 64.034), 'Age']   = 4
full.loc[ full['Age'] > 63.034, 'Age'] = 5
full['Fare'] = full['Fare'].astype(int)

# Replace with categorical values
full['Age'] = full['Age'].replace(1, '0-16')
full['Age'] = full['Age'].replace(2, '16-32')
full['Age'] = full['Age'].replace(3, '32-48')
full['Age'] = full['Age'].replace(4, '48-64')
full['Age'] = full['Age'].replace(5, '64+')


# Creating dummy columns for each new category, with a 1/0
full = pd.get_dummies(full, columns = ["Age"], prefix="Age")

# Deleting the Age and Age Band column as they are no longer needed
drop = ['Age Band']
full.drop(drop, axis = 1, inplace = True)

full.head(n=3)
full = pd.get_dummies(full, columns = ["Cabin"], prefix="Cabin")
full.head(3)
full = pd.get_dummies(full, columns = ["Embarked"], prefix="Embarked")
full.head(3)
full['Fare Band'] = pd.cut(full['Fare'], 4)
full[['Fare Band', 'Survived']].groupby(['Fare Band'], as_index=False).mean().sort_values(by='Fare Band', 
                                                                                          ascending=True)
# Locate and replace values
full.loc[ full['Fare'] <= 1.56, 'Fare'] = 1
full.loc[(full['Fare'] > 1.56) & (full['Fare'] <= 3.119), 'Fare'] = 2
full.loc[(full['Fare'] > 3.119) & (full['Fare'] <= 4.679), 'Fare']   = 3
full.loc[ full['Fare'] > 4.679, 'Fare'] = 4
full['Fare'] = full['Fare'].astype(int)

# Replace with categorical values
full['Fare'] = full['Fare'].replace(1, 'Very low')
full['Fare'] = full['Fare'].replace(2, 'Low')
full['Fare'] = full['Fare'].replace(3, 'Medium')
full['Fare'] = full['Fare'].replace(4, 'High')

# Create dummy variables
full = pd.get_dummies(full, columns = ["Fare"], prefix="Fare")

# Drop the un-needed Fare Band column
drop = ['Fare Band']
full.drop(drop, axis = 1, inplace = True)

full.head(n=3)
# Extracting Title from the Name feature
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)

# Displaying number of males and females for each Title
pd.crosstab(full['Sex'], full['Title'])
# Merge titles with similar titles
full['Title'] = full['Title'].replace('Mlle', 'Miss')
full['Title'] = full['Title'].replace('Ms', 'Miss')
full['Title'] = full['Title'].replace('Mme', 'Mrs')

# Now lets see the crosstab again
pd.crosstab(full['Sex'], full['Title'])
# Now I will replace the rare titles with the value "Rare"
full['Title'] = full['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev',
                                       'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
# Let's see how these new Title categories vary with mean survival rate
full[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Create dummy variables for Title
full = pd.get_dummies(full, columns = ["Title"], prefix="Title")

# Drop unnecessary name column
drop = ['Name']
full.drop(drop, axis = 1, inplace = True)

full.head(3)
# First creating the Family Size feature
full['Family Size'] = full['SibSp'] + full['Parch'] + 1

# Explore Family Size vs Survived
g = sns.factorplot(x="Family Size", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
# Now to bin the Family Size feature into bins
full['Lone Traveller'] = full['Family Size'].map(lambda s: 1 if s == 1 else 0)
full['Party of 2'] = full['Family Size'].map(lambda s: 1 if s == 2  else 0)
full['Medium Family'] = full['Family Size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
full['Large Family'] = full['Family Size'].map(lambda s: 1 if s >= 5 else 0)

# Delete the no longer needed Family Size
drop = ['Family Size']
full.drop(drop, axis = 1, inplace = True)

full.head(n=3)
# Convert Sex into categorical value - 0 for male and 1 for female
full["Sex"] = full["Sex"].map({"male": 0, "female":1})
full.head()
# First, I want to create a new column that extracts the prefix from the ticket.
# In the case where there is no prefix, but a number... It will return the whole number.

# I will do this by first of all creating a list of all the values in the column
tickets = list(full['Ticket'])

# Using a for loop I will create a list including the prefixes of all the values in the list
prefix = []
for t in tickets:
    split = t.split(" ", 1) # This will split each value into a list of 2 values, surrounding the " ". For example, the ticker "A/5 21171" will be split into [A/5, 21171]
    prefix.append(split)
    
# Now I want to take the first value within these lists for each value. I will put these into another list using a for loop.
tickets = []
for t in prefix:
    ticket = t[0]
    tickets.append(ticket)
    
full['Ticket_Prefix'] = pd.Series(tickets)
full.head(3)
# Create the function
def TicketPrefix(row):
    for t in row['Ticket_Prefix']:
        if t[0] == '0':
            val = 0
        elif t[0] == '1':
            val = 0
        elif t[0] == '2':
            val = 0
        elif t[0] == '3':
            val = 0
        elif t[0] == '4':
            val = 0
        elif t[0] == '5':
            val = 0
        elif t[0] == '6':
            val = 0
        elif t[0] == '7':
            val = 0
        elif t[0] == '8':
            val = 0
        elif t[0] == '9':
            val = 0
        else:
            val = 1
        return val
    
# Create a new column that appolies the above function to create its values        
full['Ticket Has Prefix'] = full.apply(TicketPrefix, axis = 1)

# Clean up variables not needed anymore
drop = ['Ticket', 'Ticket_Prefix', 'PassengerId'] # We delete passenger ID here as it is no use for the modeling
full.drop(drop, axis = 1, inplace = True)

full.head(3)
corr = full.corr()
plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(150, 250, as_cmap=True)
sns.heatmap(corr, cmap="YlGnBu", vmin = -1.0, vmax = 1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True);
# First I will take a copy of the full dataset, in case we need to change or use the full dataset for something else.
copy = full.copy()

# Remember that the training dataset had only 891 rows, and the test had 418. 
# At the start we simply concatenated the two datasets together to form our "full" dataset
# Hence, we can simply split the two datasets apart again to extract the training and test data.
train_setup = copy[:891]
test_setup = copy[891:]

# I'll take another copy just to be safe
train = train_setup.copy()
test = test_setup.copy()

# And print the shape of these dataframes to confirm it has done what I intended
print('train: ', train.shape, '\ntest: ', test.shape)
# Convert the "Survived" column in the training data to an integere
train['Survived'] = train['Survived'].astype(int)

# Drop the "Survived" column from the test data
drop = ['Survived']
test.drop(drop, axis=1, inplace=True)

print('train: ', train.shape, '\ntest: ', test.shape)
# Take another copy for the training features, because why not
train_x = train.copy()

# Now delete the "Survived" column from the training features
del train_x['Survived']

# And create a new column called train_y representing what we are trying to predict for the training dat
train_y = train['Survived']

# Confirm this all worked
print('train_x - features for training: ', train_x.shape, '\ntrain_y - target variable for training: ', train_y.shape,)
# To do this, we will use the neat package train_test_split
# This allows you to split your dataset into smaller training and test samples, controlling how much goes into each.
# For this, we will assign 25% to the validation dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_x, train_y, test_size=0.25, random_state=42)

# X_train = predictor features for estimation dataset
# X_test = predictor variables for validation dataset
# Y_train = target variable for the estimation dataset
# Y_test = target variable for the estimation dataset

print('X_train: ', X_train.shape, '\nX_test: ', X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
Image(filename='../input/titanic-images/DT.png')
Image(filename='../input/titanic-images/randomforest.png')
Image(filename='../input/titanic-images/ETC.png')
Image(filename='../input/titanic-images/ada.png')
Image(filename='../input/titanic-images/xgb.png')
Image(filename='../input/titanic-images/GBC.png')
Image(filename='../input/titanic-images/SVM.png')
Image(filename='../input/titanic-images/KNN.png')
Image(filename='../input/titanic-images/MLP.png')
Image(filename='../input/titanic-images/logreg.png')
Image(filename='../input/titanic-images/GNB.png')
Image(filename='../input/titanic-images/voting.png')
# First I will use ShuffleSplit as a way of randomising the cross validation samples.
shuff = ShuffleSplit(n_splits=3, test_size=0.2, random_state=50)

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, Y_train)
dt_scores = cross_val_score(dt, X_train, Y_train, cv = shuff)
dt_scores = dt_scores.mean()
dt_apply_acc = metrics.accuracy_score(Y_test, dt.predict(X_test))

# Random Forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, Y_train)
rf_scores = cross_val_score(rf, X_train, Y_train, cv = shuff)
rf_scores = rf_scores.mean()
rf_apply_acc = metrics.accuracy_score(Y_test, rf.predict(X_test))

# Extra Trees
etc = ExtraTreesClassifier(random_state=0)
etc.fit(X_train, Y_train)
etc_scores = cross_val_score(etc, X_train, Y_train, cv = shuff)
etc_scores = etc_scores.mean()
etc_apply_acc = metrics.accuracy_score(Y_test, etc.predict(X_test))

# Adaboost classifier
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, Y_train)
ada_scores = cross_val_score(ada, X_train, Y_train, cv = shuff)
ada_scores = ada_scores.mean()
ada_apply_acc = metrics.accuracy_score(Y_test, ada.predict(X_test))

# xgboost
xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, Y_train)
xgb_scores = cross_val_score(xgb, X_train, Y_train, cv = shuff)
xgb_scores = xgb_scores.mean()
xgb_apply_acc = metrics.accuracy_score(Y_test, xgb.predict(X_test))

#gradient boosting classifier
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, Y_train)
gbc_scores = cross_val_score(gbc, X_train, Y_train, cv = shuff)
gbc_scores = gbc_scores.mean()
gbc_apply_acc = metrics.accuracy_score(Y_test, gbc.predict(X_test))

# Support Vector Machine Classifier
svc = SVC(random_state=0)
svc.fit(X_train, Y_train)
svc_scores = cross_val_score(svc, X_train, Y_train, cv = shuff)
svc_scores = svc_scores.mean()
svc_apply_acc = metrics.accuracy_score(Y_test, svc.predict(X_test))

# Linear Support Vector Machine Classifier
lsvc = LinearSVC(random_state=0)
lsvc.fit(X_train, Y_train)
lsvc_scores = cross_val_score(lsvc, X_train, Y_train, cv = shuff)
lsvc_scores = lsvc_scores.mean()
lsvc_apply_acc = metrics.accuracy_score(Y_test, lsvc.predict(X_test))

# K-Nearest Neighbours
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_scores = cross_val_score(knn, X_train, Y_train, cv = shuff)
knn_scores = knn_scores.mean()
knn_apply_acc = metrics.accuracy_score(Y_test, knn.predict(X_test))

# Multi Layer Perceptron Classifier
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train, Y_train)
mlp_scores = cross_val_score(mlp, X_train, Y_train, cv = shuff)
mlp_scores = mlp_scores.mean()
mlp_apply_acc = metrics.accuracy_score(Y_test, mlp.predict(X_test))

# Perceptron
pcn = Perceptron(random_state=0)
pcn.fit(X_train, Y_train)
pcn_scores = cross_val_score(pcn, X_train, Y_train, cv = shuff)
pcn_scores = pcn_scores.mean()
pcn_apply_acc = metrics.accuracy_score(Y_test, pcn.predict(X_test))

#Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)
lr_scores = cross_val_score(lr, X_train, Y_train, cv = shuff)
lr_scores = lr_scores.mean()
lr_apply_acc = metrics.accuracy_score(Y_test, lr.predict(X_test))

# Stochastic Gradient Descent
sgd = SGDClassifier(random_state=0)
sgd.fit(X_train, Y_train)
sgd_scores = cross_val_score(sgd, X_train, Y_train, cv = shuff)
sgd_scores = sgd_scores.mean()
sgd_apply_acc = metrics.accuracy_score(Y_test, sgd.predict(X_test))

# Gaussian Naive Bayes
gss = GaussianNB()
gss.fit(X_train, Y_train)
gss_scores = cross_val_score(gss, X_train, Y_train, cv = shuff)
gss_scores = gss_scores.mean()
gss_apply_acc = metrics.accuracy_score(Y_test, gss.predict(X_test))


models = pd.DataFrame({
    '1_Model': ['Gradient Boosting Classifier',
              'Logistic Regression',             
              'Support Vector Machine',
              'Linear SVMC',
              'Random Forest', 
              'KNN',
              'Gaussian Naive Bayes',
              'Perceptron',
              'Stochastic Gradient Descent',
              'Decision Tree',
              'XGBoost',
              'Adaboost',
              'Extra Trees', 
              'Multi Layer Perceptron'],
    '2_Mean Cross Validation Score': [gbc_scores,
                                      lr_scores, 
                                      svc_scores, 
                                      lsvc_scores,
                                      rf_scores, 
                                      knn_scores, 
                                      gss_scores, 
                                      pcn_scores, 
                                      sgd_scores, 
                                      dt_scores,
                                      xgb_scores, 
                                      ada_scores, 
                                      etc_scores, 
                                      mlp_scores], 
    '3_Accuracy when applied to Test': [gbc_apply_acc,
                                      lr_apply_acc, 
                                      svc_apply_acc, 
                                      lsvc_apply_acc,
                                      rf_apply_acc, 
                                      knn_apply_acc, 
                                      gss_apply_acc, 
                                      pcn_apply_acc, 
                                      sgd_apply_acc, 
                                      dt_apply_acc,
                                      xgb_apply_acc, 
                                      ada_apply_acc, 
                                      etc_apply_acc, 
                                      mlp_apply_acc]
                                                    })

# Finally I will plot the scores for cross validation and test, to see the top performers
g = sns.factorplot(x="2_Mean Cross Validation Score", y="1_Model", data = models,
                    kind="bar", palette=mycols, orient = "h", size = 5, aspect = 2.5,
                  order = ['Adaboost', 'Logistic Regression', 'Support Vector Machine',
                          'Linear SVMC', 'Gradient Boosting Classifier', 'XGBoost', 
                          'Multi Layer Perceptron', 'Decision Tree', 'Random Forest', 
                          'Extra Trees', 'KNN', 'Gaussian Naive Bayes', 'Perceptron', 
                          'Stochastic Gradient Descent'])
g.despine(left = True);

h = sns.factorplot(x="3_Accuracy when applied to Test", y="1_Model", data = models,
                    kind="bar", palette=mycols, orient = "h", size = 5, aspect = 2.5,  
                  order = ['KNN', 'Support Vector Machine', 'XGBoost', 'Logistic Regression', 
                          'Linear SVMC', 'Stochastic Gradient Descent', 'Extra Trees', 
                          'Gradient Boosting Classifier', 'Multi Layer Perceptron', 
                          'Random Forest', 'Decision Tree', 'Adaboost', 'Gaussian Naive Bayes', 
                          'Perceptron'])
h.despine(left = True);
models
# Because the chi2 test will be the same for each model, we only need to run this test once.
# When we have established which features are the most important via this test, we can re-use this 
# reduced dataset for all other estimators without "feature_importances_"

# chi2 test of independence

# fit a model using a score function of chi2
Kbest = SelectKBest(score_func=chi2, k=10)
fit = Kbest.fit(X_train, Y_train)

# Create a table with the results and score for each features
scores = pd.DataFrame({'Columns': X_test.columns.values, 'Score': fit.scores_})

# Visualise the scores of dependence with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot('Score','Columns',data = scores, palette=mycols,orient = "h")
g.set_xlabel("Importance")
g = g.set_title("Feature Importances using Chi-squared")
# First I will take some copies of the original estimation, validation and test datasets
# so that we don't mess with these
chi2_reduced_train = X_train.copy()
chi2_reduced_test = X_test.copy()
chi2_reduced_final_test = test.copy()

# Now I will drop all of the columns that I deemed to be irrelevant
# I played with a variety of options here, but this is what I found to work best
drop = ['SibSp', 'Age_0-16', 'Age_16-32', 'Age_32-48', 'Age_48-64', 'Cabin_A', 'Cabin_F',
        'Cabin_G', 'Embarked_Q', 'Embarked_S', 'Fare_Low', 'Fare_Medium', 'Title_Rare',
        'Title_Master','Title_Rare', 'Ticket Has Prefix',
        'Age_64+', 'Cabin_A', 'Cabin_F', 'Cabin_G', 'Cabin_T']

# Reduce features of estimation, validation and test for use in modelling
chi2_reduced_train.drop(drop, axis = 1, inplace = True)
chi2_reduced_test.drop(drop, axis = 1, inplace = True)
chi2_reduced_final_test.drop(drop, axis = 1, inplace = True)

# You'll see that we now have just 18 features
print('X_train: ', chi2_reduced_train.shape, '\nX_test: ', chi2_reduced_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# Now I will do the same for the extra trees method of feature reduction
# Which once again, can be reused for estimators that do not have the feature_importances_ attribute

# First, let's fit a model using the extra trees classifier
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)

# And create a table with the importances
scores = pd.DataFrame({'Columns': X_test.columns.values, 'Score': model.feature_importances_})
scores.sort_values(by='Score', ascending=False)

# Finally let's visualise this
plt.subplots(figsize=(15, 10))
g = sns.barplot('Score','Columns',data = scores, palette=mycols,orient = "h")
g.set_xlabel("Importance")
g = g.set_title("Feature Importances using Trees")
# In a similar fashion, I will reduce the estimation, validation and test datasets according to
# the extra trees feature importances.

# Take another copy
etc_reduced_train = X_train.copy()
etc_reduced_test = X_test.copy()
etc_reduced_final_test = test.copy()

# Once again, I tried a few options here of which features the drop. I decided these were the best choice.
drop = ['Age_0-16', 'Age_16-32', 'Age_48-64', 'Age_64+', 'Cabin_A', 'Cabin_B',
        'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G','Cabin_T',
        'Embarked_Q', 'Title_Rare']

# Reduce features of estimation, validation and test datasets
etc_reduced_train.drop(drop, axis = 1, inplace = True)
etc_reduced_test.drop(drop, axis = 1, inplace = True)
etc_reduced_final_test.drop(drop, axis = 1, inplace = True)

# Let's see the new shape of the data
print('X_train: ', etc_reduced_train.shape, '\nX_test: ', etc_reduced_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# KNN - chi-squared
knn = KNeighborsClassifier()

# Fit estimator to reduced dataset
knn.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
knn_scores = cross_val_score(knn, chi2_reduced_train, Y_train, cv = shuff)
knn_scores = knn_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(knn_scores*100))
knn_apply_acc = metrics.accuracy_score(Y_test, knn.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(knn_apply_acc*100))
print("-"*50)

##############################################################################

# KNN - extra trees
knn = KNeighborsClassifier()

# Fit estimator to reduced dataset
knn.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
knn_scores = cross_val_score(knn, etc_reduced_train, Y_train, cv = shuff)
knn_scores = knn_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(knn_scores*100))
knn_apply_acc = metrics.accuracy_score(Y_test, knn.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(knn_apply_acc*100))
# SVM - chi-squared
svc = SVC()

# Fit estimator to reduced dataset
svc.fit(chi2_reduced_train, Y_train)
        
# Compute cross validated scores and take the mean
svc_scores = cross_val_score(svc, chi2_reduced_train, Y_train, cv = shuff)
svc_scores = svc_scores.mean()
        
print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(svc_scores*100))
svc_apply_acc = metrics.accuracy_score(Y_test, svc.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(svc_apply_acc*100))
print("-"*50)

##############################################################################

# SVM - extra trees
svc = SVC()

# Fit estimator to reduced dataset
svc.fit(etc_reduced_train, Y_train)
        
# Compute cross validated scores and take the mean
svc_scores = cross_val_score(svc, etc_reduced_train, Y_train, cv = shuff)
svc_scores = svc_scores.mean()
        
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(svc_scores*100))
svc_apply_acc = metrics.accuracy_score(Y_test, svc.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(svc_apply_acc*100)) 
# Sort feature importances from GBC model trained earlier
indices = np.argsort(gbc.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = gbc.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("GBC feature importance");
# Take some copies
gbc_red_train = X_train.copy()
gbc_red_test = X_test.copy()
gbc_final_test = test.copy()

# Fit a model to the estimation data
gbc = gbc.fit(gbc_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
gbc_feat_red = SelectFromModel(gbc, prefit = True)

# Reduce estimation, validation and test datasets
gbc_X_train = gbc_feat_red.transform(gbc_red_train)
gbc_X_test = gbc_feat_red.transform(gbc_red_test)
gbc_final_test = gbc_feat_red.transform(gbc_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', gbc_X_train.shape, '\nX_test: ', gbc_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
gbc_rfecv_train = X_train.copy()
gbc_rfecv_test = X_test.copy()
gbc_rfecv_final_test = test.copy()

# Initialise RFECV
gbc_rfecv = RFECV(estimator = gbc, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
gbc_rfecv.fit(gbc_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
gbc_rfecv_X_train = gbc_rfecv.transform(gbc_rfecv_train)
gbc_rfecv_X_test = gbc_rfecv.transform(gbc_rfecv_test)
gbc_rfecv_final_test = gbc_rfecv.transform(gbc_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(gbc_rfecv.support_)
print(gbc_rfecv.ranking_)

print('X_train: ', gbc_rfecv_X_train.shape, '\nX_test: ', gbc_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# GBC - feature_importances_
gbc = GradientBoostingClassifier(random_state=0)

# Fit estimator to reduced dataset
gbc.fit(gbc_X_train, Y_train)

# Compute cross validated scores and take the mean
gbc_scores = cross_val_score(gbc, gbc_X_train, Y_train, cv = shuff)
gbc_scores = gbc_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(gbc_scores*100))
gbc_apply_acc = metrics.accuracy_score(Y_test, gbc.predict(gbc_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(gbc_apply_acc*100))
print("-"*50)

##############################################################################

# GBC - RFECV
gbc = GradientBoostingClassifier(random_state=0)

# Fit estimator to reduced dataset
gbc.fit(gbc_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
gbc_scores = cross_val_score(gbc, gbc_rfecv_X_train, Y_train, cv = shuff)
gbc_scores = gbc_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(gbc_scores*100))
gbc_apply_acc = metrics.accuracy_score(Y_test, gbc.predict(gbc_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(gbc_apply_acc*100))
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(xgb.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = xgb.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("XGB feature importance");
# Take some copies
xgb_red_train = X_train.copy()
xgb_red_test = X_test.copy()
xgb_final_test = test.copy()

# Fit a model to the estimation data
xgb = xgb.fit(xgb_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
xgb_feat_red = SelectFromModel(xgb, prefit = True)

# Reduce estimation, validation and test datasets
xgb_X_train = xgb_feat_red.transform(xgb_red_train)
xgb_X_test = xgb_feat_red.transform(xgb_red_test)
xgb_final_test = xgb_feat_red.transform(xgb_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', xgb_X_train.shape, '\nX_test: ', xgb_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
xgb_rfecv_train = X_train.copy()
xgb_rfecv_test = X_test.copy()
xgb_rfecv_final_test = test.copy()

# Initialise RFECV
xgb_rfecv = RFECV(estimator = xgb, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
xgb_rfecv.fit(xgb_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
xgb_rfecv_X_train = xgb_rfecv.transform(xgb_rfecv_train)
xgb_rfecv_X_test = xgb_rfecv.transform(xgb_rfecv_test)
xgb_rfecv_final_test = xgb_rfecv.transform(xgb_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(xgb_rfecv.support_)
print(xgb_rfecv.ranking_)

print('X_train: ', xgb_rfecv_X_train.shape, '\nX_test: ', xgb_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# XGB - feature_importances_
xgb = XGBClassifier()

# Fit estimator to reduced dataset
xgb.fit(xgb_X_train, Y_train)

# Compute cross validated scores and take the mean
xgb_scores = cross_val_score(xgb, xgb_X_train, Y_train, cv = shuff)
xgb_scores = xgb_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(xgb_scores*100))
xgb_apply_acc = metrics.accuracy_score(Y_test, xgb.predict(xgb_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(xgb_apply_acc*100))
print("-"*50)

##############################################################################

# XGB - RFECV
xgb = XGBClassifier()

# Fit estimator to reduced dataset
xgb.fit(xgb_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
xgb_scores = cross_val_score(xgb, xgb_rfecv_X_train, Y_train, cv = shuff)
xgb_scores = xgb_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(xgb_scores*100))
xgb_apply_acc = metrics.accuracy_score(Y_test, xgb.predict(xgb_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(xgb_apply_acc*100))
# MLP - chi-squared
mlp = MLPClassifier()

# Fit estimator to reduced dataset
mlp.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
mlp_scores = cross_val_score(mlp, chi2_reduced_train, Y_train, cv = shuff)
mlp_scores = mlp_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(mlp_scores*100))
mlp_apply_acc = metrics.accuracy_score(Y_test, mlp.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(mlp_apply_acc*100))
print("-"*50)

##############################################################################

# MLP - extra trees
mlp = MLPClassifier()

# Fit estimator to reduced dataset
mlp.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
mlp_scores = cross_val_score(mlp, etc_reduced_train, Y_train, cv = shuff)
mlp_scores = mlp_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(mlp_scores*100))
mlp_apply_acc = metrics.accuracy_score(Y_test, mlp.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(mlp_apply_acc*100))
# LSVC - chi-squared
lsvc = LinearSVC()

# Fit estimator to reduced dataset
lsvc.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lsvc_scores = cross_val_score(lsvc, chi2_reduced_train, Y_train, cv = shuff)
lsvc_scores = lsvc_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(lsvc_scores*100))
lsvc_apply_acc = metrics.accuracy_score(Y_test, lsvc.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(lsvc_apply_acc*100))
print("-"*50)

##############################################################################

# LSVC - extra trees
lsvc = LinearSVC()

# Fit estimator to reduced dataset
lsvc.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lsvc_scores = cross_val_score(lsvc, etc_reduced_train, Y_train, cv = shuff)
lsvc_scores = lsvc_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(lsvc_scores*100))
lsvc_apply_acc = metrics.accuracy_score(Y_test, lsvc.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(lsvc_apply_acc*100))
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(rf.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = rf.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("RF feature importance");
# Take some copies
rf_red_train = X_train.copy()
rf_red_test = X_test.copy()
rf_final_test = test.copy()

# Fit a model to the estimation data
rf = rf.fit(rf_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
rf_feat_red = SelectFromModel(rf, prefit = True)

# Reduce estimation, validation and test datasets
rf_X_train = rf_feat_red.transform(rf_red_train)
rf_X_test = rf_feat_red.transform(rf_red_test)
rf_final_test = rf_feat_red.transform(rf_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', rf_X_train.shape, '\nX_test: ', rf_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
rf_rfecv_train = X_train.copy()
rf_rfecv_test = X_test.copy()
rf_rfecv_final_test = test.copy()

# Initialise RFECV
rf_rfecv = RFECV(estimator = rf, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
rf_rfecv.fit(rf_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
rf_rfecv_X_train = rf_rfecv.transform(rf_rfecv_train)
rf_rfecv_X_test = rf_rfecv.transform(rf_rfecv_test)
rf_rfecv_final_test = rf_rfecv.transform(rf_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(rf_rfecv.support_)
print(rf_rfecv.ranking_)

print('X_train: ', rf_rfecv_X_train.shape, '\nX_test: ', rf_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# RF - feature_importances_
rf = RandomForestClassifier()

# Fit estimator to reduced dataset
rf.fit(rf_X_train, Y_train)

# Compute cross validated scores and take the mean
rf_scores = cross_val_score(rf, rf_X_train, Y_train, cv = shuff)
rf_scores = rf_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(rf_scores*100))
rf_apply_acc = metrics.accuracy_score(Y_test, rf.predict(rf_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(rf_apply_acc*100))
print("-"*50)

##############################################################################

# RF - RFECV
rf = RandomForestClassifier()

# Fit estimator to reduced dataset
rf.fit(rf_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
rf_scores = cross_val_score(rf, rf_rfecv_X_train, Y_train, cv = shuff)
rf_scores = rf_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(rf_scores*100))
rf_apply_acc = metrics.accuracy_score(Y_test, rf.predict(rf_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(rf_apply_acc*100))
# LR - chi-squared
lr = LogisticRegression()

# Fit estimator to reduced dataset
lr.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lr_scores = cross_val_score(lr, chi2_reduced_train, Y_train, cv = shuff)
lr_scores = lr_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(lr_scores*100))
lr_apply_acc = metrics.accuracy_score(Y_test, lr.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(lr_apply_acc*100))
print("-"*50)

##############################################################################

# LR - extra trees
lr = LogisticRegression()

# Fit estimator to reduced dataset
lr.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lr_scores = cross_val_score(lr, etc_reduced_train, Y_train, cv = shuff)
lr_scores = lr_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(lr_scores*100))
lr_apply_acc = metrics.accuracy_score(Y_test, lr.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(lr_apply_acc*100))
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(dt.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = dt.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("DT feature importance");
# Take some copies
dt_red_train = X_train.copy()
dt_red_test = X_test.copy()
dt_final_test = test.copy()

# Fit a model to the estimation data
dt = dt.fit(dt_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
dt_feat_red = SelectFromModel(dt, prefit = True)

# Reduce estimation, validation and test datasets
dt_X_train = dt_feat_red.transform(dt_red_train)
dt_X_test = dt_feat_red.transform(dt_red_test)
dt_final_test = dt_feat_red.transform(dt_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', dt_X_train.shape, '\nX_test: ', dt_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
dt_rfecv_train = X_train.copy()
dt_rfecv_test = X_test.copy()
dt_rfecv_final_test = test.copy()

# Initialise RFECV
dt_rfecv = RFECV(estimator = dt, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
dt_rfecv.fit(dt_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
dt_rfecv_X_train = dt_rfecv.transform(dt_rfecv_train)
dt_rfecv_X_test = dt_rfecv.transform(dt_rfecv_test)
dt_rfecv_final_test = dt_rfecv.transform(dt_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(dt_rfecv.support_)
print(dt_rfecv.ranking_)

print('X_train: ', dt_rfecv_X_train.shape, '\nX_test: ', dt_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# DT - feature_importances_
dt = DecisionTreeClassifier()

# Fit estimator to reduced dataset
dt.fit(dt_X_train, Y_train)

# Compute cross validated scores and take the mean
dt_scores = cross_val_score(dt, dt_X_train, Y_train, cv = shuff)
dt_scores = dt_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(dt_scores*100))
dt_apply_acc = metrics.accuracy_score(Y_test, dt.predict(dt_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(dt_apply_acc*100))
print("-"*50)

##############################################################################

# DT - RFECV
dt = DecisionTreeClassifier()

# Fit estimator to reduced dataset
dt.fit(dt_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
dt_scores = cross_val_score(dt, dt_rfecv_X_train, Y_train, cv = shuff)
dt_scores = dt_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(dt_scores*100))
dt_apply_acc = metrics.accuracy_score(Y_test, dt.predict(dt_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(dt_apply_acc*100))
ada = AdaBoostClassifier()
ada.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(ada.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = ada.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("ADA feature importance");
# Take some copies
ada_red_train = X_train.copy()
ada_red_test = X_test.copy()
ada_final_test = test.copy()

# Fit a model to the estimation data
ada = ada.fit(ada_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
ada_feat_red = SelectFromModel(ada, prefit = True)

# Reduce estimation, validation and test datasets
ada_X_train = ada_feat_red.transform(ada_red_train)
ada_X_test = ada_feat_red.transform(ada_red_test)
ada_final_test = ada_feat_red.transform(ada_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', ada_X_train.shape, '\nX_test: ', ada_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
ada_rfecv_train = X_train.copy()
ada_rfecv_test = X_test.copy()
ada_rfecv_final_test = test.copy()

# Initialise RFECV
ada_rfecv = RFECV(estimator = ada, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
ada_rfecv.fit(ada_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
ada_rfecv_X_train = ada_rfecv.transform(ada_rfecv_train)
ada_rfecv_X_test = ada_rfecv.transform(ada_rfecv_test)
ada_rfecv_final_test = ada_rfecv.transform(ada_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(ada_rfecv.support_)
print(ada_rfecv.ranking_)

print('X_train: ', ada_rfecv_X_train.shape, '\nX_test: ', ada_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# ADA - feature_importances_
ada = AdaBoostClassifier()

# Fit estimator to reduced dataset
ada.fit(ada_X_train, Y_train)

# Compute cross validated scores and take the mean
ada_scores = cross_val_score(ada, ada_X_train, Y_train, cv = shuff)
ada_scores = ada_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(ada_scores*100))
ada_apply_acc = metrics.accuracy_score(Y_test, ada.predict(ada_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(ada_apply_acc*100))
print("-"*50)

##############################################################################

# ADA - RFECV
ada = AdaBoostClassifier()

# Fit estimator to reduced dataset
ada.fit(ada_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
ada_scores = cross_val_score(ada, ada_rfecv_X_train, Y_train, cv = shuff)
ada_scores = ada_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(ada_scores*100))
ada_apply_acc = metrics.accuracy_score(Y_test, ada.predict(ada_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(ada_apply_acc*100))
etc = ExtraTreesClassifier()
etc.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(etc.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = etc.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("ETC feature importance");
# Take some copies
etc_red_train = X_train.copy()
etc_red_test = X_test.copy()
etc_final_test = test.copy()

# Fit a model to the estimation data
etc = etc.fit(etc_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
etc_feat_red = SelectFromModel(etc, prefit = True)

# Reduce estimation, validation and test datasets
etc_X_train = etc_feat_red.transform(etc_red_train)
etc_X_test = etc_feat_red.transform(etc_red_test)
etc_final_test = etc_feat_red.transform(etc_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', etc_X_train.shape, '\nX_test: ', etc_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
etc_rfecv_train = X_train.copy()
etc_rfecv_test = X_test.copy()
etc_rfecv_final_test = test.copy()

# Initialise RFECV
etc_rfecv = RFECV(estimator = etc, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
etc_rfecv.fit(etc_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
etc_rfecv_X_train = etc_rfecv.transform(etc_rfecv_train)
etc_rfecv_X_test = etc_rfecv.transform(etc_rfecv_test)
etc_rfecv_final_test = etc_rfecv.transform(etc_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(etc_rfecv.support_)
print(etc_rfecv.ranking_)

print('X_train: ', etc_rfecv_X_train.shape, '\nX_test: ', etc_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# ETC - feature_importances_
etc = ExtraTreesClassifier()

# Fit estimator to reduced dataset
etc.fit(etc_X_train, Y_train)

# Compute cross validated scores and take the mean
etc_scores = cross_val_score(etc, etc_X_train, Y_train, cv = shuff)
etc_scores = etc_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(etc_scores*100))
etc_apply_acc = metrics.accuracy_score(Y_test, etc.predict(etc_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(etc_apply_acc*100))
print("-"*50)

##############################################################################

# ETC - RFECV
etc = ExtraTreesClassifier()

# Fit estimator to reduced dataset
etc.fit(etc_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
etc_scores = cross_val_score(etc, etc_rfecv_X_train, Y_train, cv = shuff)
etc_scores = etc_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(etc_scores*100))
etc_apply_acc = metrics.accuracy_score(Y_test, etc.predict(etc_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(etc_apply_acc*100))
# K-Nearest Neighbours
knn = KNeighborsClassifier()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', knn.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(knn_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(knn_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
knn_param_grid = {'n_neighbors': [11], 
                  'weights': ['uniform'], 
                  'algorithm': ['auto'], 
                  'leaf_size': [5],
                  'p': [1]
                 }

# Run the GridSearchCV against the above grid
gsKNN = GridSearchCV(knn, param_grid = knn_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsKNN.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
KNN_best = gsKNN.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
KNN_pred_acc = metrics.accuracy_score(Y_test, gsKNN.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsKNN.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsKNN.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(KNN_pred_acc*100))
# Support Vector Machine
svc = SVC()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', svc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(svc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(svc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
svc_param_grid = {'C': [0.2],
                  'kernel': ['rbf'],
                  'degree': [1],
                  'gamma': [0.3],
                  'coef0': [0.1],
                  'max_iter': [-1],
                  'decision_function_shape': ['ovo'],
                  'probability': [True]
                 }

# Run the GridSearchCV against the above grid
gsSVC = GridSearchCV(svc, param_grid = svc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsSVC.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
SVC_best = gsSVC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
SVC_pred_acc = metrics.accuracy_score(Y_test, gsSVC.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsSVC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsSVC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(SVC_pred_acc*100))
# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=0)

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', gbc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(gbc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(gbc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
gbc_param_grid = {'loss' : ["deviance"],
                  'n_estimators' : [50],
                  'learning_rate': [0.1],
                  'max_depth': [2],
                  'min_samples_split': [2],
                  'min_samples_leaf': [3]
                 }

# Run the GridSearchCV against the above grid
gsGBC = GridSearchCV(gbc, param_grid = gbc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsGBC.fit(gbc_X_train, Y_train)

# Choose the best estimator from the GridSearch
GBC_best = gsGBC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
GBC_pred_acc = metrics.accuracy_score(Y_test, gsGBC.predict(gbc_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsGBC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsGBC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(GBC_pred_acc*100))
# XGBoost
xgb = XGBClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', xgb.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(xgb_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(xgb_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
xgb_param_grid = {'n_jobs': [-1],
                  'min_child_weight': [2], 
                  'max_depth': [3], 
                  'gamma': [0.1], 
                  'learning_rate': [0.05], 
                  'n_estimators': [200], 
                  'subsample': [0.75],
                  'colsample_bytree': [0.3],
                  'colsample_bylevel': [0.2], 
                  'booster': ['gbtree'], 
                  "reg_alpha": [0.1],
                  'reg_lambda': [0.6]
                 }

# Run the GridSearchCV against the above grid
gsXGB = GridSearchCV(xgb, param_grid = xgb_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsXGB.fit(xgb_X_train, Y_train)

# Choose the best estimator from the GridSearch
XGB_best = gsXGB.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
XGB_pred_acc = metrics.accuracy_score(Y_test, gsXGB.predict(xgb_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsXGB.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsXGB.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(XGB_pred_acc*100))
# Multi Layer Perceptron
mlp = MLPClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', mlp.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(mlp_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(mlp_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters

mlp_param_grid = {'hidden_layer_sizes': [(100, )], 
                  'activation': ['relu'],
                  'solver': ['adam'], 
                  'alpha': [0.0001], 
                  'batch_size': ['auto'], 
                  'learning_rate': ['constant'], 
                  'max_iter': [300], 
                  'tol': [0.01],
                  'learning_rate_init': [0.01], 
                  'power_t': [0.7], 
                  'momentum': [0.7], 
                  'early_stopping': [True],
                  'beta_1': [0.9], 
                  'beta_2': [ 0.999], 
                  'epsilon': [0.00000001] 
                 }

# Run the GridSearchCV against the above grid
gsMLP = GridSearchCV(mlp, param_grid = mlp_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsMLP.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
MLP_best = gsMLP.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
MLP_pred_acc = metrics.accuracy_score(Y_test, gsMLP.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsMLP.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsMLP.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(MLP_pred_acc*100))
# Linear Support Vector Machine
lsvc = LinearSVC()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', lsvc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(lsvc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(lsvc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
lsvc_param_grid = {'tol': [0.0001],
                   'C': [0.1],
                   'fit_intercept': [True], 
                   'intercept_scaling': [0.2], 
                   'max_iter': [500]
                  }

# Run the GridSearchCV against the above grid
gsLSVC = GridSearchCV(lsvc, param_grid = lsvc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsLSVC.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
LSVC_best = gsLSVC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
LSVC_pred_acc = metrics.accuracy_score(Y_test, gsLSVC.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsLSVC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsLSVC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(LSVC_pred_acc*100))
# Random Forest
rf = RandomForestClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', rf.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(rf_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(rf_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
rf_param_grid = {'n_estimators': [500], 
                 'criterion': ['gini'], 
                 'max_features': [None],
                 'max_depth': [5],
                 'min_samples_split': [3],
                 'min_samples_leaf': [5],
                 'max_leaf_nodes': [None], 
                 'random_state': [0,], 
                 'oob_score': [True],
                 'n_jobs': [-1] 
                 }

# Run the GridSearchCV against the above grid
gsRF = GridSearchCV(rf, param_grid = rf_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsRF.fit(rf_rfecv_X_train, Y_train)

# Choose the best estimator from the GridSearch
RF_best = gsRF.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
RF_pred_acc = metrics.accuracy_score(Y_test, gsRF.predict(rf_rfecv_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsRF.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsRF.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(RF_pred_acc*100))
# Logistic Regression
lr = LogisticRegression()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', lr.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(lr_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(lr_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
lr_param_grid = {'tol': [0.00001],
                 'C': [0.4],
                 'fit_intercept': [True], 
                 'intercept_scaling': [0.5], 
                 'max_iter': [500], 
                 'solver': ['liblinear']  
                  }

# Run the GridSearchCV against the above grid
gsLR = GridSearchCV(lr, param_grid = lr_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsLR.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
LR_best = gsLR.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
LR_pred_acc = metrics.accuracy_score(Y_test, gsLR.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsLR.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsLR.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(LR_pred_acc*100))
# Decision Tree
dt = DecisionTreeClassifier()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', dt.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(dt_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(dt_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
dt_param_grid = {'criterion': ['gini'], 
                 'max_features': ['auto'],
                 'max_depth': [4],
                 'min_samples_split': [4],
                 'min_samples_leaf': [3],
                 'max_leaf_nodes': [15] 
                }

# Run the GridSearchCV against the above grid
gsDT = GridSearchCV(dt, param_grid = dt_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsDT.fit(dt_X_train, Y_train)

# Choose the best estimator from the GridSearch
DT_best = gsDT.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
DT_pred_acc = metrics.accuracy_score(Y_test, gsDT.predict(dt_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsDT.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsDT.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(DT_pred_acc*100))
# Adaboost
ada_dt = DecisionTreeClassifier()
ada = AdaBoostClassifier(ada_dt)

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', ada.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(ada_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(ada_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
ada_param_grid = {'n_estimators': [200], 
                  'learning_rate': [0.01],
                  'base_estimator__criterion' : ["gini"],
                  'base_estimator__max_depth': [3],
                  'base_estimator__min_samples_split': [10],
                  'base_estimator__min_samples_leaf': [1]
                 }

# Run the GridSearchCV against the above grid
gsADA = GridSearchCV(ada, param_grid = ada_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsADA.fit(ada_rfecv_X_train, Y_train)

# Choose the best estimator from the GridSearch
ADA_best = gsADA.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
ADA_pred_acc = metrics.accuracy_score(Y_test, gsADA.predict(ada_rfecv_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsADA.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsADA.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(ADA_pred_acc*100))
# Extra Trees
etc = ExtraTreesClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', etc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(etc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(etc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
etc_param_grid = {"max_depth": [3],
                  "min_samples_split": [3],
                  "min_samples_leaf": [5],
                  "n_estimators" :[50],
                  "criterion": ["gini"]
                 }

# Run the GridSearchCV against the above grid
gsETC = GridSearchCV(etc, param_grid = etc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsETC.fit(etc_rfecv_X_train, Y_train)

# Choose the best estimator from the GridSearch
ETC_best = gsETC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
ETC_pred_acc = metrics.accuracy_score(Y_test, ETC_best.predict(etc_rfecv_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsETC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsETC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(ETC_pred_acc*100))
# First thing I want to do, is compare how similarly each of the optimised estimators predict the test dataset.
# I'll do this by creating "columns" using pd.Series to concatenate together
knn_ensemble = pd.Series(gsKNN.predict(etc_reduced_final_test), name = "KNN")
svc_ensemble = pd.Series(gsSVC.predict(etc_reduced_final_test), name = "SVC")
gbc_ensemble = pd.Series(gsGBC.predict(gbc_final_test), name = "GBC")
xgb_ensemble = pd.Series(gsXGB.predict(xgb_final_test), name = "XGB")
mlp_ensemble = pd.Series(gsMLP.predict(etc_reduced_final_test), name = "MLP")
lsvc_ensemble = pd.Series(gsLSVC.predict(etc_reduced_final_test), name = "LSVC")
rf_ensemble = pd.Series(gsRF.predict(rf_rfecv_final_test), name = "RF")
lr_ensemble = pd.Series(gsLR.predict(etc_reduced_final_test), name = "LR")
dt_ensemble = pd.Series(gsDT.predict(dt_final_test), name = "DT")
ada_ensemble = pd.Series(gsADA.predict(ada_rfecv_final_test), name = "ADA")
etc_ensemble = pd.Series(gsETC.predict(etc_rfecv_final_test), name = "ETC")

# Concatenate all classifier results
ensemble_results = pd.concat([knn_ensemble, svc_ensemble, gbc_ensemble, xgb_ensemble, 
                              mlp_ensemble, lsvc_ensemble, rf_ensemble, 
                              lr_ensemble, dt_ensemble, ada_ensemble, etc_ensemble], axis=1)


plt.subplots(figsize=(20, 15))
g= sns.heatmap(ensemble_results.corr(),annot=True, cmap = "YlGnBu", linewidths = 1.0)
# Set up the voting classifier with all of the optimised models I have built.
votingC = VotingClassifier(estimators=[('KNN', KNN_best), ('SVC', SVC_best),('GBC', GBC_best),
                                       ('XGB', XGB_best), ('MLP', MLP_best), ('RF', RF_best), 
                                       ('LR', LR_best), ('DT', DT_best), ('ADA', ADA_best), 
                                       ('ETC', ETC_best)], voting='soft', n_jobs=4)

# Fit the model to the training data
votingC = votingC.fit(X_train, Y_train)

# Take the cross validated training scores as an average
votingC_scores = cross_val_score(votingC, X_train, Y_train, cv = shuff)
votingC_scores = votingC_scores.mean()

# Print the results and include how accurately the voting ensemble was able to predict
# the validation dataset
print('Mean Cross Validated Score: {:.2f}'. format(votingC_scores*100))
votingC_apply_acc = metrics.accuracy_score(Y_test, votingC.predict(X_test))
print('Accuracy when applied to Test: {:.2f}'. format(votingC_apply_acc*100))
# Output predictions into a csv file for Kaggle upload
KNN_test_pred = KNN_best.predict(etc_reduced_final_test)
KNN_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": KNN_test_pred})
# KNN_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('K-Nearest Neighbours predictions uploaded to CSV!')


# Output predictions into a csv file for Kaggle upload
SVC_test_pred = SVC_best.predict(etc_reduced_final_test)
SVC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": SVC_test_pred})
# SVC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Support Vector Machine Classifier predictions uploaded to CSV!')


# Output predictions into a csv file for Kaggle upload
GBC_test_pred = GBC_best.predict(gbc_final_test)
GBC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": GBC_test_pred})
# GBC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Gradient Boosting Classifier predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
XGB_test_pred = XGB_best.predict(xgb_final_test)
XGB_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": XGB_test_pred})
# XGB_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('XGBoost predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
MLP_test_pred = MLP_best.predict(etc_reduced_final_test)
MLP_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": MLP_test_pred})
# MLP_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Multi Layer Perceptron predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
LSVC_test_pred = LSVC_best.predict(etc_reduced_final_test)
LSVC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": LSVC_test_pred})
# LSVC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Linear Support Vector Machine Classifier predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
RF_test_pred = RF_best.predict(rf_rfecv_final_test)
RF_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": RF_test_pred})
# RF_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Random Forest predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
LR_test_pred = LR_best.predict(etc_reduced_final_test)
LR_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": LR_test_pred})
# LR_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Logistic Regression predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
DT_test_pred = DT_best.predict(dt_final_test)
DT_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": DT_test_pred})
# DT_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Decision Tree predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
ADA_test_pred = ADA_best.predict(ada_rfecv_final_test)
ADA_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": ADA_test_pred})
# ADA_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Adaboost predictions uploaded to CSV!')

# Extra Trees
ETC_test_pred = ETC_best.predict(etc_rfecv_final_test)
ETC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": ETC_test_pred})
# ETC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Extra Trees predictions uploaded to CSV!')

# Voting
votingC_test_pred = votingC.predict(test)
votingC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": votingC_test_pred})
# votingC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Voting predictions uploaded to CSV!')

print('-'*50)
print("Here's a preview of what the CSV looks like...")
votingC_submission.head()
Image(filename='../input/titanic-images/end2.png')
