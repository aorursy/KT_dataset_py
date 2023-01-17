# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
telco_df = pd.read_excel("../input/Telco-Customer-Churn.xlsx")
# Importing the packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns
telco_df.head()
telco_df.info()
telco_df.isna().sum()
telco_df.describe()
corr = telco_df.corr()
corr
sns.heatmap(corr, annot=True)
telco_df.columns
sns.countplot(x='gender', data=telco_df)
sns.factorplot(x='gender', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='DeviceProtection', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='StreamingMovies', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='StreamingTV', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='InternetService', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='SeniorCitizen', col='Churn', kind='count', data=telco_df);
sns.distplot(telco_df['tenure'], color = 'green')

plt.title('Customer tenure with Telecom Provider')



sns.distplot(telco_df['MonthlyCharges'], color = 'green')

plt.title('Monthly Charges')



sns.factorplot(x='PhoneService', col='Churn', kind='count', data=telco_df);

sns.factorplot(x='MultipleLines', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='OnlineBackup', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='TechSupport', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='Contract', col='Churn', kind='count', data=telco_df);
sns.factorplot(x='PaperlessBilling', col='Churn', kind='count', data=telco_df);
plt.rcParams['figure.figsize'] = (18, 5)

sns.factorplot(x='PaymentMethod', col='Churn', kind='count', data=telco_df);

plt.xticks(rotation =70)



plt.show
sns.countplot(x="Churn", data=telco_df)

telco_df.Churn.value_counts()
cat_df = telco_df[['gender', 'Partner', 'Dependents',

       'PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod', 'Churn']]

cat_df.shape
cat_cols = pd.get_dummies(cat_df, drop_first=True)

cat_cols.head()
cat_cols.shape
num_df = telco_df.drop(['gender', 'Partner', 'Dependents',

       'PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod', 'Churn'], axis=1)

num_df.shape
num_df.info()
dataset = pd.concat([num_df,cat_cols], axis=1 )
dataset.shape
dataset.info()
dataset['TotalCharges']==' '
dataset['TotalCharges'][dataset['TotalCharges']==' ']
dataset = dataset.drop(labels = list(dataset.TotalCharges[dataset.TotalCharges == " "].index))
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])
y = dataset["Churn_Yes"].values



X = dataset.drop(['Churn_Yes','customerID'], axis=1)
# Stratified sampling



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101,stratify=y)
# Importing the packages for Decision Tree Classifier



from sklearn import tree

my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=101, min_samples_leaf=3, class_weight="balanced")  #, class_weight="balanced"

my_tree_one
# Fit the decision tree model on your features and label



my_tree_one = my_tree_one.fit(X_train, y_train)
# The feature_importances_ attribute make it simple to interpret the significance of the predictors you include



list(zip(X_train.columns,my_tree_one.feature_importances_))


# The accuracy of the model on Train data



print(my_tree_one.score(X_train, y_train))





# The accuracy of the model on Test data



print(my_tree_one.score(X_test, y_test))
# Visualize the decision tree graph



with open('tree.dot','w') as dotfile:

    tree.export_graphviz(my_tree_one, out_file=dotfile, feature_names=X_train.columns, filled=True)

    dotfile.close()

    

# You may have to install graphviz package using 

# conda install graphviz

# conda install python-graphviz



from graphviz import Source



with open('tree.dot','r') as f:

    text=f.read()

    plot=Source(text)

plot   
y_pred = my_tree_one.predict(X_test)
#Print Confusion matrix on Train Data

from sklearn.metrics import confusion_matrix, classification_report



pred = my_tree_one.predict(X_test)

df_confusion = confusion_matrix(y_test, pred)

df_confusion
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,

            fmt='d')
# Remove few features and train
# Setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two



my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 101, class_weight='balanced')

my_tree_two = my_tree_two.fit(X_train, y_train)



#Print the score of both the decision tree



print("New Decision Tree Accuracy: ",my_tree_two.score(X_train, y_train))

print("Original Decision Tree Accuracy",my_tree_one.score(X_train,y_train))
# Making predictions on our Test Data 



pred = my_tree_two.predict(X_test)
print("New Decision Tree Accuracy on test data: ",my_tree_two.score(X_test, y_test))
# The accuracy of the model on Train data



print(my_tree_two.score(X_train, y_train))





# The accuracy of the model on Test data



print(my_tree_two.score(X_test, y_test))
# Building confusion matrix of our improved model



df_confusion_new = confusion_matrix(y_test, pred)

df_confusion_new
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_new, cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,

            fmt='d')
# Building and fitting Random Forest



from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(criterion = 'gini',  n_estimators = 100, max_depth = 10,random_state = 101, class_weight="balanced")
# Fitting the model on Train Data



my_forest = forest.fit(X_train, y_train)
# Print the accuracy score of the fitted random forest



print(my_forest.score(X_train, y_train))



print(my_forest.score(X_test, y_test))

# Making predictions



pred = my_forest.predict(X_test)
list(zip(X_train.columns,my_forest.feature_importances_))
df_confusion_rf = confusion_matrix(y_test, pred)

df_confusion_rf
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_rf, cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,

            fmt='d')
# Different parameters we want to test



max_depth = [5,10,15] 

criterion = ['gini', 'entropy']

min_samples_split = [5,10,15]
# Importing GridSearch



from sklearn.model_selection import GridSearchCV
# Building the model



my_tree_three = tree.DecisionTreeClassifier(class_weight="balanced")



# Cross-validation tells how well a model performs on a dataset using multiple samples of train data

grid = GridSearchCV(estimator = my_tree_three, cv=3, 

                    param_grid = dict(max_depth = max_depth, criterion = criterion, min_samples_split=min_samples_split), verbose=2)
grid.fit(X_train,y_train)
# Best accuracy score



print('Avg accuracy score across 54 models:', grid.best_score_)
# Best parameters for the model



grid.best_params_
# Building the model based on new parameters



my_tree_three = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 10, random_state=42, class_weight="balanced")
my_tree_three.fit(X_train,y_train)
# Accuracy Score for new model



my_tree_three.score(X_train,y_train)
# Different parameters we want to test



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Importing RandomizedSearchCV



from sklearn.model_selection import RandomizedSearchCV
forest_two = RandomForestClassifier(class_weight="balanced")



# Fitting 3 folds for each of 100 candidates, totalling 300 fits

rf_random = RandomizedSearchCV(estimator = forest_two, param_distributions = random_grid, 

                               n_iter = 100, cv = 3, verbose=2, random_state=42)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_