import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel(r"/kaggle/input/sales-pipeline-conversion-at-a-saas-startup/Sales Dataset.xlsx")

df.head()
df.info()
df.shape
df.describe()
df_dub = df.copy()



# Checking for duplicates and dropping the entire duplicate row if any

df_dub.drop_duplicates(subset=None, inplace=True)

df_dub.shape
df.shape
del df_dub
# List of variables to map



varlist =  ['Opportunity Status']



# Defining the map function

def binary_map(x):

    return x.map({'Won': 1, "Loss": 0})



# Applying the function to the housing list

df[varlist] = df[varlist].apply(binary_map)

df.head()
df= df.drop(['Opportunity ID'],1)

df.head()
df['Technology\nPrimary'].describe()
df['Technology\nPrimary'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Technology\nPrimary'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Technology\nPrimary", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['City'].describe()
df['City'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['City'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "City", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['B2B Sales Medium'].describe()
df['B2B Sales Medium'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['B2B Sales Medium'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "B2B Sales Medium", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['Sales Velocity'].describe()
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Sales Velocity'])

plt.show()
# As we can see there are a number of outliers in the data.

# We will cap the outliers to 95% value for analysis.
percentiles = df['Sales Velocity'].quantile([0.05,0.95]).values

df['Sales Velocity'][df['Sales Velocity'] <= percentiles[0]] = percentiles[0]

df['Sales Velocity'][df['Sales Velocity'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Sales Velocity'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Sales Velocity', x = 'Opportunity Status', data = df)

plt.show()
df['Opportunity Status'].describe()
df['Opportunity Status'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Opportunity Status'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.show()
df['Sales Stage Iterations'].describe()
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Sales Stage Iterations'])

plt.show()
percentiles = df['Sales Stage Iterations'].quantile([0.05,0.95]).values

df['Sales Stage Iterations'][df['Sales Stage Iterations'] <= percentiles[0]] = percentiles[0]

df['Sales Stage Iterations'][df['Sales Stage Iterations'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Sales Stage Iterations'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Sales Stage Iterations', x = 'Opportunity Status', data = df)

plt.show()
df['Opportunity Size (USD)'].describe()
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Opportunity Size (USD)'])

plt.show()
percentiles = df['Opportunity Size (USD)'].quantile([0.05,0.95]).values

df['Opportunity Size (USD)'][df['Opportunity Size (USD)'] <= percentiles[0]] = percentiles[0]

df['Opportunity Size (USD)'][df['Opportunity Size (USD)'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Opportunity Size (USD)'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Opportunity Size (USD)', x = 'Opportunity Status', data = df)

plt.show()
df['Client Revenue Sizing'].describe()
df['Client Revenue Sizing'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Client Revenue Sizing'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Client Revenue Sizing", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['Client Employee Sizing'].describe()
df['Client Employee Sizing'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Client Employee Sizing'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Client Employee Sizing", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['Business from Client Last Year'].describe()
df['Business from Client Last Year'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Business from Client Last Year'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Business from Client Last Year", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['Compete Intel'].describe()
df['Compete Intel'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Compete Intel'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Compete Intel", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
df['Opportunity Sizing'].describe()
df['Opportunity Sizing'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(df['Opportunity Sizing'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set_yscale('log')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Opportunity Sizing", hue = "Opportunity Status", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df[['Technology\nPrimary', 'City', 'B2B Sales Medium', 'Client Revenue Sizing',

                            'Client Employee Sizing', 'Business from Client Last Year',

                            'Compete Intel', 'Opportunity Sizing']], drop_first=True)

dummy1.head()

# Adding the results to the master dataframe

df = pd.concat([df, dummy1], axis=1)

df.head()
df = df.drop(['Technology\nPrimary', 'City', 'B2B Sales Medium', 'Client Revenue Sizing',

              'Client Employee Sizing', 'Business from Client Last Year',

              'Compete Intel', 'Opportunity Sizing'], axis = 1)

df.head()
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = df.drop(['Opportunity Status'], axis=1)
X.head()
X.shape
# Putting response variable to y

y = df['Opportunity Status']
y.head()
y.shape
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=125)
X_train.head()
X_train.shape
X_test.head()
X_test.shape
y_train.head()
y_train.shape
y_test.head()
y_test.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['Sales Velocity','Sales Stage Iterations',

         'Opportunity Size (USD)']] = scaler.fit_transform(X_train[['Sales Velocity','Sales Stage Iterations',

                                                                    'Opportunity Size (USD)']])



X_train.head()
X_test[['Sales Velocity','Sales Stage Iterations',

         'Opportunity Size (USD)']] = scaler.transform(X_test[['Sales Velocity','Sales Stage Iterations',

                                                               'Opportunity Size (USD)']])



X_test.head()
# Checking the Opportunity Status Rate

Opportunity = round((sum(df['Opportunity Status'])/len(df['Opportunity Status'].index))*100,2)

print("We have almost {} %  Opportunity rate after successful data manipulation".format(Opportunity))
from  sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

model = LogisticRegression(solver='lbfgs',max_iter=1000)
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
trainaccuracy = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', trainaccuracy)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features_to_remove = vif.loc[vif['VIF'] >= 4.99,'Features'].values

features_to_remove = list(features_to_remove)

print(features_to_remove)
X_train = X_train.drop(columns=features_to_remove, axis = 1)

X_train.head()
X_test = X_test.drop(columns=features_to_remove, axis = 1)

X_test.head()
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
accuracytrain = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', accuracytrain)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train, predict_train )

print(confusion)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our model

trainsensitivity= TP / float(TP+FN)

trainsensitivity
# Let us calculate specificity

trainspecificity= TN / float(TN+FP)

trainspecificity
# Calculate false postive rate - predicting Opportunity when company does not have Opportunity

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print(TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
draw_roc(y_train,predict_train)
#Using sklearn utilities for the same
from sklearn.metrics import precision_score, recall_score

precision_score(y_train,predict_train)
recall_score(y_train,predict_train)
# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data\n\n',predict_test)
confusion2 = metrics.confusion_matrix(y_test, predict_test )

print(confusion2)
# Let's check the overall accuracy.

testaccuracy= accuracy_score(y_test,predict_test)

testaccuracy
# Let's see the sensitivity of our model

testsensitivity=TP / float(TP+FN)

testsensitivity
# Let us calculate specificity

testspecificity= TN / float(TN+FP)

testspecificity
# Let us compare the values obtained for Train & Test:

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

print("Train Data Sensitivity :{} %".format(round((trainsensitivity*100),2)))

print("Train Data Specificity :{} %".format(round((trainspecificity*100),2)))

print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))

print("Test Data Sensitivity  :{} %".format(round((testsensitivity*100),2)))

print("Test Data Specificity  :{} %".format(round((testspecificity*100),2)))