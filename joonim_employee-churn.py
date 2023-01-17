# Data analysis libraries

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings("ignore")

from IPython.display import display

pd.options.display.max_columns = None



# Plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

sns.set_palette("Set1")



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import plotly.tools as tls



init_notebook_mode(connected=True) #do not miss this line



# ML libraries

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
# Read it

data_raw = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')



# Copy it

data = data_raw.copy()



# Peep it

data.head()
# Numeric columns

num_data = data.select_dtypes("int")

num_data.info()
# Non-numeric columns

obj_data = data.select_dtypes(exclude=["int"])

obj_data.info()
# Print the unique values of columns with "object" dtype

for col in obj_data:

    print(col, ":", obj_data[col].unique())
# Convert 'object' to 'category' 

data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
# Provide the correct order of categories

data['BusinessTravel'] = data['BusinessTravel'].cat.reorder_categories(['Non-Travel', 'Travel_Rarely','Travel_Frequently'])
# Plot counts for all numeric features

data.hist(figsize=(20,20))

plt.show()
# Attrition distribution

print("Percentage of Current Employees = {:.1f}%".format(

    data[data['Attrition'] == 'No'].shape[0] / data.shape[0]*100))



print("Percentage of Former Employees = {:.1f}%".format(

    data[data['Attrition'] == 'Yes'].shape[0] / data.shape[0]*100))



# Count & Percent by Attrition

grouped = data[['Attrition']].groupby(["Attrition"])

output = pd.DataFrame()

output['n'] = grouped["Attrition"].count()

output.reset_index(inplace=True)



# Sort in ascending order for plotting

output = output.sort_values('n', ascending=True)



# Plotly object: Attrition Counts

p = [go.Bar(x = output.Attrition, 

            y = output.n, 

            text=output.n,

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='Attrition Count')

fig = go.Figure(data=p, layout=layout)



py.offline.iplot(fig)
# Attrition by Department

grouped = data.groupby(["Department","Attrition"])

output = pd.DataFrame()

output['n'] = grouped["Attrition"].count()



# Identifying sub-categories within cohorts

output['pct'] = output['n'].groupby("Department").transform(lambda x: x/x.sum())



# Reset index 

output = output.reset_index()



# Filter for only employees who have left 

output_yes = output[output["Attrition"] == "Yes"]



# Sort in ascending order for plotting

output_yes = output_yes.sort_values('n', ascending=True)



# Plotly object: Attrition Count by Dept

p = [go.Bar(x = output_yes.Department, 

            y = output_yes.n, 

            text=output_yes.n,

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='Attrition Count by Dept')

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
# Sort in ascending order for plotting

output_yes = output_yes.sort_values('pct', ascending=True)



# Change decimal to integer for readability 

output_yes['pct'] = output_yes['pct'] * 100



# Plotly object: Attrition Count by Dept

p = [go.Bar(x = output_yes.Department, 

            y = output_yes.pct, 

            text=output_yes.pct.round(decimals=2),

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='Attrition Rate by Department')

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
# Attrition by Department

grouped = data.groupby(["Department","JobRole","Attrition"])

output = pd.DataFrame()

output['n'] = grouped["Attrition"].count()



# Identifying sub-categories within cohorts

output['pct'] = output['n'].groupby("JobRole").transform(lambda x: x/x.sum())



# Reset index 

output = output.reset_index()



# Filter for only employees who have left 

output_yes = output[output["Attrition"] == "Yes"]



# Replace long dept names with acronyms

output_yes['Department'] = output_yes['Department'].str.replace('Research & Development', "R&D", case=False)

output_yes['Department'] = output_yes['Department'].str.replace('Human Resources', "HR", case=False)



# Add column with combined dept & job role

output_yes['dept_jobrole'] = output_yes['Department'].str.cat(output_yes[['JobRole']], sep=' : ')



# Sort in ascending order for plotting

output_yes = output_yes.sort_values('pct', ascending=True)



# Change decimal to integer for readability 

output_yes['pct'] = output_yes['pct'] * 100



# Plotly object: Attrition Count by Dept

p = [go.Bar(x = output_yes.dept_jobrole, 

            y = output_yes.pct,

            text=output_yes.pct.round(decimals=2),

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='Attrition Rate by Job Role',

                   yaxis=dict(title="Percent"),

                   margin = dict(b = 120

  ))

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
# Attrition by Department

grouped = data.groupby(['BusinessTravel',"Attrition"])

output = pd.DataFrame()

output['n'] = grouped["Attrition"].count()



# Identifying sub-categories within cohorts

output['pct'] = output['n'].groupby("BusinessTravel").transform(lambda x: x/x.sum())



# Reset index

output = output.reset_index()



# Filter for only employees who have left

output_yes = output[output['Attrition'] == "Yes"]



# Change decimal to integer for readability 

output_yes['pct'] = output_yes['pct'] * 100



# Plotly object: Attrition Count by Amount of Travel

p = [go.Bar(x = output_yes.BusinessTravel, 

            y = output_yes.n,

            text=output_yes.n.round(decimals=2),

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='Business Travel vs Attrition Count',

                   yaxis=dict(title="Count"))

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
# Plotly object: Attrition Pct by Amount of Travel

p = [go.Bar(x = output_yes.BusinessTravel, 

            y = output_yes.pct,

            text=output_yes.pct.round(decimals=2),

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='Business Travel vs Attrition Rate',

                   yaxis=dict(title="Percent"))

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
# Attrition by Department

grouped = data.groupby(['OverTime',"Attrition"])

output = pd.DataFrame()

output['n'] = grouped["Attrition"].count()



# Identifying sub-categories within cohorts

output['pct'] = output['n'].groupby("OverTime").transform(lambda x: x/x.sum())



# Reset index

output = output.reset_index().sort_values('n', ascending=False)



# Filter for only employees who have left

output_yes = output[output['Attrition'] == "Yes"]



# Change decimal to integer for readability 

output_yes['pct'] = output_yes['pct'] * 100



# Plotly object: Attrition Count by Amount of Travel

p = [go.Bar(x = output_yes.OverTime, 

            y = output_yes.n,

            text=output_yes.n.round(decimals=2),

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='OverTime vs Attrition Count',

                   yaxis=dict(title="Count"))

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
# Plotly object: Attrition Count by Amount of Travel

p = [go.Bar(x = output_yes.OverTime, 

            y = output_yes.pct,

            text=output_yes.pct.round(decimals=2),

            textposition = 'outside',

            orientation = 'v',

            opacity=0.6, 

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5)))]



# Pieces of Flair

layout = go.Layout(title='OverTime vs Attrition Rate',

                   yaxis=dict(title="Percent"))

fig = go.Figure(data=p, layout=layout)



# Plot it

py.offline.iplot(fig)
obj_data.info()
# Make a temporary copy 

df_temp = data.copy()



# Convert target variable from category to integer 

df_temp['Attrition'] = df_temp['Attrition'].apply(lambda x: 0 if x == 'No' else 1).astype('int')



# Drop non-useful columns

df_temp = df_temp.drop(

    ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)



# Create a correlation matrix

corr_metrics = df_temp.corr().round(decimals=3)



print('Most Positive Correlations: \n\n', corr_metrics['Attrition'].sort_values().tail(5))

print('\nMost Negative Correlations: \n\n', corr_metrics['Attrition'].sort_values().head(5))



corr_metrics.style.background_gradient(cmap='Blues')
# Drop non-useful variables

data.drop(['EmployeeCount', 'EmployeeNumber',

            'StandardHours', 'Over18'], axis=1, inplace=True)



# Drop highly-correlated redundant variables

data = data.drop(['MonthlyIncome'], axis=1)
# Load encoder from Sci-kit Learn 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Create a label encoder object

le = LabelEncoder()



# Label Encoding will be used for columns with 2 or less unique values

le_count = 0

for col in data.columns[1:]:

    if data[col].dtype != 'int64':

        if len(list(data[col].unique())) <= 2:

            le.fit(data[col])

            data[col] = le.transform(data[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))
# Turn categorical variables into dummy variables

data_ml = pd.get_dummies(data, 

                      drop_first=True) # Avoid dummy trap



data_ml.head()
from sklearn.model_selection import train_test_split



# Choose the dependent variable column (churn) and set it as target

target = data_ml['Attrition']



# Drop column churn and set everything else as features

features = data_ml.drop("Attrition",axis=1)



# Use that function to create the splits both for target and for features

# Set the test sample to be 25% of your observations

train_labels, test_labels, \

train_features, test_features = train_test_split(target,

                                                 features,

                                                 test_size=0.25,

                                                 random_state=42)



# Check shape of the data

print("Shape of Train Features: ", train_features.shape) 

print("Shape of Train Labels: ", train_labels.shape) 

print("Shape of Test Features: ", test_features.shape) 

print("Shape of Test Labels: ", test_labels.shape) 

# Import decision tree 

from sklearn.tree import DecisionTreeClassifier



# Train our decision tree

tree = DecisionTreeClassifier(random_state=10)

tree.fit(train_features, train_labels)



# Predict the labels for the test data

tree_predictions = tree.predict(test_features)



# Create the classification report 

from sklearn.metrics import classification_report

class_rep_tree = classification_report(test_labels, tree_predictions)



# Print summary

print("Unbalanced")

print("=" * 80)

print("Decision Tree: \n", class_rep_tree)

print("Decision Tree Accuracy score: {}".format(accuracy_score(test_labels, tree_predictions)), "\n")



#####



# Import LogisticRegression

from sklearn.linear_model import LogisticRegression



# Train our logistic regression and predict labels for the test set

logreg = LogisticRegression(random_state=10)

logreg.fit(train_features, train_labels)

logreg_predictions = logreg.predict(test_features)



# Create the classification report

class_rep_log = classification_report(test_labels, logreg_predictions)



# Print summary

print("Logistic Regression: \n", class_rep_log)

print("Logistic Regression Accuracy score: {}".format(accuracy_score(test_labels, logreg_predictions)))
##### Data Preparation #####



# Subset only the churners

yes_only = data_ml.loc[data_ml['Attrition'] == 1]



# Sample the non-churners to be the same number as there are churners

no_only = data_ml.loc[data_ml['Attrition'] == 0].sample(len(yes_only), random_state=10)



# Concatenate the dataframes no_only and yes_only

attrition_bal = pd.concat([yes_only, no_only])



# Choose the dependent variable column (churn) and set it as target

target = attrition_bal['Attrition']



# Drop column churn and set everything else as features

features = attrition_bal.drop("Attrition",axis=1)



# Use that function to create the splits both for target and for features

# Set the test sample to be 25% of your observations

train_labels, test_labels, \

train_features, test_features = train_test_split(target,

                                                 features,

                                                 test_size=0.25,

                                                 random_state=42)



# Check shape of the data

print("Shape of Train Features: ", train_features.shape) 

print("Shape of Train Labels: ", train_labels.shape) 

print("Shape of Test Features: ", test_features.shape) 

print("Shape of Test Labels: ", test_labels.shape) 
##### Decision Tree Model #####



# Train our decision tree on the balanced data

tree = DecisionTreeClassifier(random_state=10)

tree.fit(train_features, train_labels)



# Predict the labels for the test data

tree_predictions = tree.predict(test_features)



print("Balanced")

print("=" * 80)

print("Decision Tree: \n", classification_report(test_labels, tree_predictions))



# Accuracy score of the prediction for the training set

print("Training Set Accuracy: ", tree.score(train_features,train_labels)*100)



# Accuracy score of the prediction for the test set

print("Testing Set Accuracy: ", tree.score(test_features,test_labels)*100, "\n")



##### Logistic Regression Model #####



# Train our logistic regression on the balanced data

logreg = LogisticRegression(random_state=10)

logreg.fit(train_features, train_labels)

pred_labels_logit = logreg.predict(test_features)



# Compare the balanced data models

print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))



# Accuracy score of the prediction for the training set

print("Training Set Accuracy: ", logreg.score(train_features,train_labels)*100)



# Accuracy score of the prediction for the test set

print("Testing Set Accuracy: ", logreg.score(test_features,test_labels)*100)
# Import modules

from sklearn.model_selection import KFold, cross_val_score



# Set up our K-fold cross-validation

kf = KFold(n_splits=20, random_state=10)



# Train our models using KFold cv

tree_score = cross_val_score(tree, features, target, cv=kf)

logit_score = cross_val_score(logreg, features, target, cv=kf)



# Print the mean of each array of scores

print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.metrics import accuracy_score



rfc_model = RFC(n_estimators=100, random_state=42, max_depth=11, max_features=11).fit(train_features, train_labels)

rfc_prediction = rfc_model.predict(test_features)

rfc_score = accuracy_score(test_labels, rfc_prediction)



# Check performance 

print("Balanced")

print("=" * 80)

print("Random Forest:\n", classification_report(test_labels, rfc_prediction))



# Set up our K-fold cross-validation

kf = KFold(n_splits=20, random_state=10)



# Train our models using KFold cv

rfc_score_cv = cross_val_score(rfc_model, features, target, cv=kf)



# Print the accuracy

print("Accuracy: ", rfc_score)



# Print the mean  

print("Cross Validation Mean Score:", np.mean(rfc_score_cv))
from sklearn.ensemble import AdaBoostClassifier as ABC



ada_model = ABC(n_estimators=100, random_state=42, learning_rate=.80).fit(train_features, train_labels)

ada_prediction = ada_model.predict(test_features)

ada_score = accuracy_score(test_labels, ada_prediction)



# Train our models using KFold cv

ada_score_cv = cross_val_score(ada_model, features, target, cv=kf)



# Check performance 

print("Balanced")

print("=" * 80)

print("ADA Boost:\n", classification_report(test_labels, ada_prediction))



# Print the accuracy

print("Accuracy: ", ada_score)



# Print the CV mean  

print("Cross Validation Mean Score:", np.mean(ada_score_cv))
# Import libraries

import h2o

from h2o.automl import H2OAutoML



# Initiate instance

h2o.init()



# Convert data to h2o data frame 

df = h2o.H2OFrame(data.copy())



# Split data 

train, valid, test = df.split_frame(ratios=[0.7,0.15], seed=1234)



# Set target variable

response = "Attrition"



# Set target to categorical

train[response] = train[response].asfactor()

valid[response] = valid[response].asfactor()

test[response] = test[response].asfactor()



# Non-target features

predictors = df.columns[:-1]



print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])



# Set AutoML parameters

aml = H2OAutoML(max_models = 10, 

                max_runtime_secs=300, 

                seed = 1234)



# Train multiple models

aml.train(x=predictors, 

          y=response, 

          training_frame=train, 

          validation_frame=valid)



# Check leaderboard

lb = aml.leaderboard

print(lb.head())



# Generate predictions on a test set 

#preds = aml.leader.predict(test)



#metalearner = h2o.get_model(aml.leader.metalearner()['name'])

#metalearner.std_coef_plot()