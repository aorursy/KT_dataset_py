#################################################################

# MSc in Data Analytics - Dublin Institute of Technology

# Machine Learning

# Assignment - Task 1

# Students: 

#     Rodrigo Bastos

#     Murali Rajendran

##################################################################



import pandas as pd

import numpy as np

import time

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
#######################################

# DATA PREPARATION

#######################################



# Import data into a dataframe

df = pd.read_csv('../input/weatherAUS.csv')

df.shape

# Drop columns not applicable to the models to be created.

# Resulting dataframe to be saved into a new variable (df2) to preserve the original data.

df2 = df.drop(['Date', 'Location', 'RISK_MM'], axis=1)
# Drop rows with NA/Blanks/Nulls

df2 = df2.dropna()

df2.shape
# Re-code boolean variables into integers (0, 1 for No, Yes)

df2['RainToday'].replace('No', 0, inplace=True)

df2['RainToday'].replace('Yes', 1, inplace=True)

df2['RainTomorrow'].replace('No', 0, inplace=True)

df2['RainTomorrow'].replace('Yes', 1, inplace=True)
# Create dummy variables from the categorical attributes.

# Saving the resulting dataframe into a new variable (df3).

df3 = pd.get_dummies(df2)

df3.shape
# Split the features from the target

features = df3.loc[:, df3.columns != 'RainTomorrow']

scaler = MinMaxScaler(feature_range=[0, 1]) # Scale features between 0 and 1

x = scaler.fit_transform(features)
# Split the target variable

target = df3.loc[:,['RainTomorrow']].values

y = np.ravel(target) # Converts vector into array to use in models.
# Feature Selection using Feature Importance from Random Forest Classifier

model = RandomForestClassifier()

model.fit(x, y)
# Save the feature scores into a dataframe and add the columns labels to it.

rfc_fi = pd.DataFrame(model.feature_importances_).transpose()

rfc_fi.columns = list(features)
# Re-Transpose the headed data and sort it by score (descending)

rfc_scores = rfc_fi.transpose()

rfc_scores.sort_values(0, ascending=False, inplace = True)
# The list below shows that 17 out of 65 features have a score above 0.01

# As a rule of thumb, we will use 20 as the number of features to use in subsequent feature selection techniques.

rfc_scores.head(20)
# Feature Selection using Univariate Selection (Chi-squared)

model = SelectKBest(score_func=chi2, k=20) # Using k = 20 due to results from Feature Importance from Random Forest Classifier.

model.fit(x, y)

np.set_printoptions(precision=3)
# Process feature scores into a sorted dataframe with the top 20 attributes.

us_scores_df = pd.DataFrame(model.scores_).transpose()

us_scores_df.columns = list(features)

us_scores_df = us_scores_df.transpose()

us_scores_df.sort_values(0, ascending=False, inplace = True)

us_scores_df.head(20) # returns the top 20 attributes by score.
# Feature Selection using Recursive Feature Extraction

model = LogisticRegression()

rfe = RFE(model, 20)

fit = rfe.fit(x, y)
# Process resulting scores into a dataframe with the selected 20 features.

rfe_scores_df = pd.DataFrame(fit.support_).transpose()

rfe_scores_df.columns = list(features)

rfe_scores_df = rfe_scores_df.transpose()

rfe_scores_df.sort_values(0, ascending=False, inplace = True)

rfe_scores_df.head(20) # returns the top 20 attributes by score.
# Extract the top 20 headers from the results of the 3 techniques

rfe_features = list(rfe_scores_df.head(20).transpose())

rfc_features = list(rfc_scores.head(20).transpose())

us_features = list(us_scores_df.head(20).transpose())

# Combine the results from the 3 techniques, and get the unique values.

combo_features = np.unique(np.array(rfe_features + rfc_features + us_features))

# The above analysis resulted in a list of 34 headers to be extracted from

# the pool of 65 features of the original dataset.
#Convert scaled features (array) into dataframe

features_df = pd.DataFrame(x)

features_df.columns = list(features)
# Split the original dataset between Training and Test sets at a ratio of 70/30

master_x_train, master_x_test, master_y_train, master_y_test = train_test_split(features_df, target, test_size =0.3, random_state = 0)
# Get train/test subsets based on results of 3 feature selection techniques.

y_train = np.ravel(master_y_train)

y_test = np.ravel(master_y_test)

rfe_x_train = master_x_train[rfe_features]

rfe_x_test = master_x_test[rfe_features]

rfc_x_train= master_x_train[rfc_features]

rfc_x_test= master_x_test[rfc_features]

us_x_train= master_x_train[us_features]

us_x_test= master_x_test[us_features]

combo_x_train= master_x_train[combo_features]

combo_x_test= master_x_test[combo_features]
############################################

# MODEL CREATION, TRAINING AND EVALUATION

############################################



# Function to output model metrics for comparison.

def EvaluateModel(model, x_test, y_test):

    print("Accuracy:  {}".format(model.score(x_test, y_test)))

    print("Precision: {}".format(precision_score(y_test, model.predict(x_test))))

    print("Recall:    {}".format(recall_score(y_test, model.predict(x_test), average='macro')))

    print("ROC AUC:   {}".format(roc_auc_score(y_test, model.predict_proba(x_test)[:,1])))
# Function to Train and Evaluate a given model and a set of data.

def BuildAndEvaluateModel(model, x_train, y_train, x_test, y_test):

    print('')

    start_time = time.time()

    print('Training Model...')

    model.fit(x_train, y_train)

    print('Model Trained... Evaluating... Duration so far: ' + str(time.time() - start_time))

    EvaluateModel(model, x_test, y_test)

    end_time = time.time()

    print('Done! Total Duration: ' + str(end_time - start_time))
# Logistic Regression Model

lr = LogisticRegression(solver = 'liblinear')
BuildAndEvaluateModel(lr, master_x_train, y_train, master_x_test, y_test)

BuildAndEvaluateModel(lr, rfe_x_train, y_train, rfe_x_test, y_test)

BuildAndEvaluateModel(lr, rfc_x_train, y_train, rfc_x_test, y_test)

BuildAndEvaluateModel(lr, us_x_train, y_train, us_x_test, y_test)

BuildAndEvaluateModel(lr, combo_x_train, y_train, combo_x_test, y_test)
# Decision Tree Model

dt = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=6, min_samples_leaf=5)
BuildAndEvaluateModel(dt, master_x_train, y_train, master_x_test, y_test)

BuildAndEvaluateModel(dt, rfe_x_train, y_train, rfe_x_test, y_test)

BuildAndEvaluateModel(dt, rfc_x_train, y_train, rfc_x_test, y_test)

BuildAndEvaluateModel(dt, us_x_train, y_train, us_x_test, y_test)

BuildAndEvaluateModel(dt, combo_x_train, y_train, combo_x_test, y_test)
# Support Vector Machine Model

svm = SVC(gamma = 'auto', probability = True)
# Warning: Long training times

# Over 30 minutes on an i5 processor and 8gb ram

BuildAndEvaluateModel(svm, master_x_train, y_train, master_x_test, y_test)

BuildAndEvaluateModel(svm, rfe_x_train, y_train, rfe_x_test, y_test)

BuildAndEvaluateModel(svm, rfc_x_train, y_train, rfc_x_test, y_test)

BuildAndEvaluateModel(svm, us_x_train, y_train, us_x_test, y_test)

BuildAndEvaluateModel(svm, combo_x_train, y_train, combo_x_test, y_test)