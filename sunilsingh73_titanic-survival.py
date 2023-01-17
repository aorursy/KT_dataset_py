import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



import warnings

warnings.filterwarnings('ignore')
# Read datasets

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_df.head()
test_df.head()
train_df.shape
train_df.describe()
train_df.info()
round(train_df.isnull().sum()/len(train_df.index) * 100, 2)
## 'Cabin' column as almost 77% data is missing value we will not consider it for model building

## Around 20% age data is missing, populate with median age as mean and median seems very close??

train_df.fillna({'Age': train_df['Age'].median()}, inplace=True)

# Keep test dataset in sync

test_df.fillna({'Age': test_df['Age'].median()}, inplace=True)



# Check missing values again

print(round(train_df.isnull().sum()/len(train_df.index) * 100, 2))

print(round(test_df.isnull().sum()/len(test_df.index) * 100, 2))

## only 2% values are missing for 'Embarked'. Lets check unique value_counts 

print(train_df['Embarked'].value_counts()) 

print(test_df['Embarked'].value_counts()) 
# Impute missing 'Embarked' values by mode which is 'S'

train_df.fillna({'Embarked': 'S'}, inplace=True)

# Keep test dataset in sync

test_df.fillna({'Embarked': 'S'}, inplace=True)



# Check missing values again

print(round(train_df.isnull().sum()/len(train_df.index) * 100, 2))

print(round(test_df.isnull().sum()/len(test_df.index) * 100, 2))
# Look at unique values in categorical columns

print(train_df['Pclass'].unique(),  train_df['Sex'].unique(), train_df['SibSp'].unique(), train_df['Parch'].unique(), train_df['Embarked'].unique())
# Encode gender as binary column

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female':0})

# Keep test set in sync

test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female':0})
train_df.info()
# One-hot encoding for 'Embarked'

embarked_dummy_1 = pd.get_dummies(train_df['Embarked'], drop_first=True)

train_df = pd.concat([train_df, embarked_dummy_1], axis=1)

train_df = train_df.drop('Embarked', axis=1)



# Keep test set in sync

embarked_dummy_2 = pd.get_dummies(test_df['Embarked'], drop_first=True)

test_df = pd.concat([test_df, embarked_dummy_2], axis=1)

test_df = test_df.drop('Embarked', axis=1)
train_df.head()
# Scale numeric columns in training set

# Pclass, Age, SibSp, Parch & Fare

scaler = MinMaxScaler()

train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = scaler.fit_transform(train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
# Scale test set 

test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = scaler.transform(test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
## Split X & y in training set

#  we'll not consider identity like columns passenger-id, name and ticket-number for the model

#  We'll also discard 'Cabin' column due to large number of missing values

X_train = train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Q', 'S']]

y_train = train_df['Survived']
## Lets view co-relation between columns

plt.figure(figsize=(10, 8))

sns.heatmap(X_train.corr(), annot=True)

plt.show()
## Function for building the model and reporting stats

def build_logistics_model(X, y):

    # Build the model

    logit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())

    logit_model = logit.fit()

    # Print stats

    print(logit_model.summary())

    # Calculate and print VIF

    vif_df = pd.DataFrame()

    vif_df['Columns'] = X.columns

    vif_df['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print()

    print(vif_df)

    # Return the model

    return logit_model
# Build and explore first model

logit_model = build_logistics_model(X_train, y_train)
# Lets drop 'Q' as p value is very high and rebuild the model

X_train = X_train.drop('Q', axis=1)

logit_model = build_logistics_model(X_train, y_train)
# Lets drop 'Parch' as p value is very high and rebuild the model

X_train = X_train.drop('Fare', axis=1)

logit_model = build_logistics_model(X_train, y_train)
# Make prediction using training set

y_train_prob = logit_model.predict(sm.add_constant(X_train))
# Function for plotting ROC curve and calculating 

def roc_analysis(y_actual, y_probability):

    # Plot ROC

    sns.set(style='darkgrid')

    fpr, tpr, theshold = roc_curve(y_actual, y_probability)

    ax, fig = plt.subplots(figsize=(10, 6))

    plt.xlim(-0.1, 1.0)

    plt.ylim(0.0, 1.0)

    plt.title("Receiver Operating Charecteristics Curve", fontsize=14)

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    sns.scatterplot(x=fpr, y=tpr)

    plt.show()



    # Calculate auc

    roc_auc = roc_auc_score(y_actual, y_probability)



    return roc_auc
auc = roc_analysis(y_train, y_train_prob)

print('\n', 'AUC =', auc)
## Function for calculating accuracy, sensitivity, specificity and precision for cut-offs 0.1 - 0.9

def generate_logit_metrics(y_actual, y_probability):

    cut_offs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 

               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    accuracy, sensitivity, specificity, precision = [], [], [], []        

    for cut_off in cut_offs:

        # Predict for given cut-offall

        y_pred = pd.Series(y_probability).apply(lambda x: 1 if x >= cut_off else 0)

        # Generate confusion matrix for this cut-off

        cm = confusion_matrix(y_train, y_pred)

        TN = cm[0, 0]

        TP = cm[1, 1]

        FP = cm[0, 1]

        FN = cm[1, 0]

        # Calculate accuracy

        accuracy.append((TN+TP)/(TN+TP+FP+FN))

        # Calculate sensitivity/TPR/Recall

        sensitivity.append(TP/(TP+FN))

        # Calculate specificity

        specificity.append(TN/(TN+FP))

        # calculate precision

        precision.append(TP/(TP+FP))

    return pd.DataFrame({'Cutoff':cut_offs, 'Accuracy':accuracy, 'Sensitivity':sensitivity, 

                         'Specificity':specificity, 'Precision':precision}).fillna(0)
# Inspect metrics

logit_metrics = generate_logit_metrics(y_train, y_train_prob)

logit_metrics
# Plot accuracy, sensitivity and specificivity metric together against cut-offs

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot('Cutoff', 'Accuracy', data=logit_metrics)

ax.plot('Cutoff', 'Sensitivity', data=logit_metrics)

ax.plot('Cutoff', 'Specificity', data=logit_metrics)

ax.set_xticks(logit_metrics['Cutoff'])

plt.ylabel('Metrics')

h, l = ax.get_legend_handles_labels()

ax.legend(handles=h, labels=['Accuracy', 'Sensitivity', 'Specificity'])

plt.show()
## Determine the probability-cutoff for the model

# 0.35 seems to be optimal cut-off for our model
# Prepare test dataset for prediction

X_test = test_df[X_train.columns]

# Make prediction (probability) using test set

y_test_prob = logit_model.predict(sm.add_constant(X_test))

# Create final prediction set

y_test_pred = pd.DataFrame()

# Append 'PassengerId' & probability to the prediction set

y_test_pred['PassengerId'] = test_df['PassengerId']

y_test_pred['Probability'] = y_test_prob

# Apply probability-cutoff to get final prediction

y_test_pred['Survived'] = y_test_prob.apply(lambda x: 1 if x >= 0.35 else 0)

y_test_pred.head()
# Finally save prediction to csv for submission

y_test_pred[['PassengerId', 'Survived']].set_index('PassengerId').to_csv('gender_submission_1.csv')