# Load libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib.pyplot import show

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.datasets import make_classification

from matplotlib import pyplot as plt

import imblearn

from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score



from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt



from sklearn.model_selection  import train_test_split

from sklearn import metrics

from sklearn.metrics import *

import statsmodels.formula.api as smf

import random

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics, datasets, tree

%matplotlib inline



from sklearn.metrics import mean_squared_error

import math



from imblearn.under_sampling import RandomUnderSampler

import imblearn



from sklearn.preprocessing import scale

from sklearn.neural_network import MLPClassifier

from sklearn import metrics, datasets

from sklearn.model_selection import GridSearchCV

# Load data

sample_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/sample_submission.csv", low_memory = False)

train_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv", low_memory = False)

test_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv", low_memory = False)
sample_df.shape
train_df.shape
test_df.shape
train_df.columns

# predict variable = Response
test_df.columns
train_df.head(3)
train_df.dtypes
train_df.isnull().sum()
test_df.isnull().sum()
# Left join sample_df and test_df

df = sample_df.merge(test_df, on='id', how='left')

df.shape
# Stack train_df and test_df + sample_df

df = pd.concat([train_df,df])
df.shape
df_response = df.loc[df['Response'] == 1]

df_no_response = df.loc[df['Response'] == 0]
# Create a function for countplot

def get_graph(data, xname1, title):



    sns.set(style="darkgrid")

    total = float(len(data)) 

    ax = sns.countplot(x = xname1, data = data)

    ax.set_title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/total),

                ha="center") 

    show()
get_graph(df, "Gender", "Gender")
from matplotlib import pyplot as plt

ax = sns.countplot(x = "Age", data = df)

ax.set_title("Age")

plt.xticks(rotation=90)

plt.xticks(size=7)
ax = sns.distplot(df["Age"])

ax.set_title("Age")
get_graph(df, "Driving_License", "Driving_License")
ax = sns.countplot(x = "Region_Code", data = df)

ax.set_title("Region_Code")

plt.xticks(rotation=60)

plt.xticks(size=7)
ax = sns.distplot(df["Region_Code"])

ax.set_title("Region_Code")
get_graph(df, "Previously_Insured", "Previously_Insured")
get_graph(df, "Vehicle_Age", "Vehicle_Age")
get_graph(df, "Vehicle_Damage", "Vehicle_Damage")
ax = sns.distplot(df["Annual_Premium"])

ax.set_title("Annual_Premium")

# plt.xticks(rotation=90)

# plt.xticks(size=7)


ax = sns.distplot(df["Policy_Sales_Channel"])

ax.set_title("Policy_Sales_Channel")
ax = sns.boxplot(df["Policy_Sales_Channel"])

ax.set_title("Policy_Sales_Channel")


ax = sns.distplot(df["Vintage"])

ax.set_title("Vintage")
ax = sns.boxplot(df["Vintage"])

ax.set_title("Vintage")
get_graph(df, "Response", "Response")
# Create a function for multi-variate countplot

def get_compare_graph_count(data, xname1, hname, title):



    sns.set(style="darkgrid")

    total = float(len(data)) 

    ax = sns.countplot(x = xname1, hue = hname, data = data)

    ax.set_title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+(p.get_width()/2.),

                height + 3,

                '{:1.2f}'.format(height/total),

                ha="center") 

    show()
get_compare_graph_count(df, "Gender", "Response", "Gender vs Response")
ax = sns.boxplot(x=df["Response"], y = df["Age"])

ax.set_title("Age vs Response")
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(df_response["Age"],  ax=ax1)

sns.distplot(df_response["Age"],  ax=ax2)

ax1.set_title("Response Age")

ax2.set_title("No Response Age")

get_compare_graph_count(df, "Driving_License", "Response", "Driving_License vs Response")
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(df_response["Region_Code"],  ax=ax1)

sns.distplot(df_response["Region_Code"],  ax=ax2)

ax1.set_title("Response Region_Code")

ax2.set_title("No Response Region_Code")
ax = sns.boxplot(x=df["Response"], y = df["Annual_Premium"])

ax.set_title("Annual_Premium vs Response")
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(df_response["Annual_Premium"],  ax=ax1)

sns.distplot(df_response["Annual_Premium"],  ax=ax2)

ax1.set_title("Response Annual_Premium")

ax2.set_title("No Response Annual_Premium")
ax = sns.boxplot(x=df["Response"], y = df["Policy_Sales_Channel"])

ax.set_title("Policy_Sales_Channel vs Response")
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(df_response["Policy_Sales_Channel"],  ax=ax1)

sns.distplot(df_response["Policy_Sales_Channel"],  ax=ax2)

ax1.set_title("Response Policy_Sales_Channel")

ax2.set_title("No Response Policy_Sales_Channel")
get_compare_graph_count(df, "Previously_Insured", "Response", "Previously_Insured vs Response")
get_compare_graph_count(df, "Vehicle_Age", "Response", "Vehicle_Age vs Response")
get_compare_graph_count(df, "Vehicle_Damage", "Response", "Vehicle_Damage vs Response")
ax = sns.boxplot(x=df["Response"], y = df["Vintage"])

ax.set_title("Vintage vs Response")
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(df_response["Vintage"],  ax=ax1)

sns.distplot(df_response["Vintage"],  ax=ax2)

ax1.set_title("Response Vintage")

ax2.set_title("No Response Vintage")
d = df.corr()
corr1 = df[["Age", "Driving_License", "Region_Code", "Previously_Insured", "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"]]

f = plt.figure(figsize=(19, 15))

plt.matshow(corr1.corr(), fignum=f.number)

plt.xticks(range(corr1.shape[1]), corr1.columns, fontsize=14, rotation=45)

plt.yticks(range(corr1.shape[1]), corr1.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16)
df.dtypes
from scipy.stats import chi2_contingency



# Cramer's V function

def cramer_v(x, y):

    n = len(x)

    ct = pd.crosstab(x, y) # crosstab

    chi2 = chi2_contingency(ct)[0]

    v = np.sqrt(chi2 / (n * (np.min(ct.shape) - 1)))

    return v
cramer_v(df['Gender'], df['Response'])
cramer_v(df['Vehicle_Age'], df['Response'])
cramer_v(df['Vehicle_Damage'], df['Response'])
table = pd.crosstab(df["Response"], df["Gender"])



chi2, p, dof, expected = chi2_contingency(table.values)

print("chi-sqr stat: ",chi2,"p-value:" ,p)

sns.countplot(x="Gender", hue="Response", data=df)
table = pd.crosstab(df["Response"], df["Vehicle_Age"])



chi2, p, dof, expected = chi2_contingency(table.values)

print("chi-sqr stat: ",chi2,"p-value:" ,p)

sns.countplot(x="Vehicle_Age", hue="Response", data=df)
table = pd.crosstab(df["Response"], df["Vehicle_Damage"])



chi2, p, dof, expected = chi2_contingency(table.values)

print("chi-sqr stat: ",chi2,"p-value:" ,p)

sns.countplot(x="Vehicle_Damage", hue="Response", data=df)
df1 = pd.get_dummies(df, columns=['Vehicle_Age'])

df1 = pd.get_dummies(df1, columns=['Vehicle_Damage'])

df1 = pd.get_dummies(df1, columns=['Gender'])

df1 = df1.drop(['id'], axis = 1)

df1 = df1.drop(['Driving_License'], axis = 1)
df1.head(3)
X = df1.drop(['Response'], axis = 1)

y = df1[["Response"]]



# Split data; 25% test size from the combined dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.25,random_state=1234, stratify=y)

# Create a model (object) for classification

dtm = DecisionTreeClassifier()

# Build a decision tree

dtm.fit(X_train, y_train)

y_pred = dtm.predict(X_test)



# Build a confusion matrix and show the Classification Report

cm = metrics.confusion_matrix(y_test,y_pred)

print('\nConfusion Matrix','\n',cm)

print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))
# Create a function for evaluation metrics

def print_result(cm, y_test, y_pred):

    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

    specificity = cm[1,1]/(cm[1,0]+cm[1,1])

    mse = mean_squared_error(y_test, y_pred)

    rmse = math.sqrt(mse)



    print ('Accuracy:', accuracy_score(y_test, y_pred))

    print ('F1 score:', f1_score(y_test, y_pred))

    print ('Recall(Specificity):', recall_score(y_test, y_pred))

    print('Sensitivity : ', sensitivity )

    print ('Precision:', precision_score(y_test, y_pred))

    print('RMSE:', math.sqrt(mse))

    print ('AUC:', roc_auc_score(y_test, y_pred))
print_result(cm, y_test, y_pred)
# Create a function for roc plotting

def plot_roc(y_test, y_pred, title):

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred)



    plt.subplots(1, figsize=(4,4))

    plt.title(title, fontsize = 15)

    plt.plot(false_positive_rate1, true_positive_rate1)

    plt.plot([0, 1], ls="--")

    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate', fontsize = 10)

    plt.xlabel('False Positive Rate', fontsize = 10)

    plt.show()

    print ('AUC:', roc_auc_score(y_test, y_pred))
plot_roc(y_test, y_pred, 'ROC Curve for DT Before Resampling')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.25,random_state=1234, stratify=y)

# Create NB model

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

# Build a confusion matrix

cm = metrics.confusion_matrix(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))
print_result(cm, y_test, y_pred)
plot_roc(y_test, y_pred, 'ROC Curve for NB Before Resampling')
# Build a random forest classification model

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.25,random_state=1234, stratify=y)

rfcm = RandomForestClassifier()

rfcm.fit(X_train, y_train)

y_pred = rfcm.predict(X_test)



# Build a confusion matrix and show the Classification Report

cm = metrics.confusion_matrix(y_test,y_pred)

print('\nConfusion Matrix','\n',cm)

print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))

# Build a confusion matrix and show the Classification Report

cm = metrics.confusion_matrix(y_test,y_pred)

print('\nConfusion Matrix','\n',cm)

print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))

print_result(cm, y_test, y_pred)
plot_roc(y_test, y_pred, 'ROC Curve for RF Before Resampling')
# Normalize the data

Xn = scale(X)

# Split data

Xn_train, Xn_test = train_test_split(Xn, test_size =.25,random_state=1234, stratify=y)

y_train_1, y_test_1 = train_test_split(y, test_size=.25, random_state=1234, stratify=y)



# Create a model

nnm = MLPClassifier()



nnm = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000,activation='logistic')



# Make predictions

nnm.fit(Xn_train, y_train_1)

y_pred_1 = nnm.predict(Xn_test)



# Build a confusion matrix and show the Classification Report

cm = metrics.confusion_matrix(y_test_1,y_pred_1)

print('\nConfusion Matrix','\n',cm)

print('\nClassification Report','\n',metrics.classification_report(y_test_1,y_pred_1))
print_result(cm, y_test_1, y_pred_1)
plot_roc(y_test_1, y_pred_1, 'ROC Curve for NN Before Resampling')
X = df1.drop(['Response'], axis = 1)

y = df1["Response"]

# define undersample strategy

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size =.25,random_state=1234, stratify=y)



rus = RandomUnderSampler(sampling_strategy='majority')

x_train_rus, y_train_rus = rus.fit_sample(x_train, y_train.ravel())
print("Before RUS, counts of label '1': {}".format(sum(y_train==1)))

print("Before RUS, counts of label '0': {} \n".format(sum(y_train==0)))

print("After RUS, counts of label '1': {}".format(sum(y_train_rus==1)))

print("After RUS, counts of label '0': {}".format(sum(y_train_rus==0)))
# Create instance of Decision Tree Classifier

dtm_rus = DecisionTreeClassifier()





# Fit the instance with training data

dtm_rus.fit(x_train_rus, y_train_rus)



# Predict using the fitted model

y_pred_dt_rus = dtm_rus.predict(x_test)
# Build a confusion matrix and show the Classification Report

cm_dt_rus = metrics.confusion_matrix(y_test,y_pred_dt_rus)

print('\nConfusion Matrix DT RUS','\n',cm_dt_rus)

print('\nClassification Report DECISION TREE RUS','\n',metrics.classification_report(y_test,y_pred_dt_rus))

print('--------------------------------------------------------------------------------')
print_result(cm_dt_rus, y_test, y_pred_dt_rus)
plot_roc(y_test, y_pred_dt_rus, 'ROC Curve for DT After Resampling')
# Create a model (object) for classification

rfcm_rus = RandomForestClassifier()



# Build a random forest classification model

rfcm_rus.fit(x_train_rus, y_train_rus)



y_pred_rf_rus = rfcm_rus.predict(x_test)
# Build a confusion matrix and show the Classification Report

cm_rf_rus = metrics.confusion_matrix(y_test,y_pred_rf_rus)

print('\nConfusion Matrix RF RUS','\n',cm_rf_rus)

print('\nClassification Report RF RUS','\n',metrics.classification_report(y_test,y_pred_rf_rus))

print("---------------------------------------------------------------------------------------")

print_result(cm_rf_rus, y_test, y_pred_rf_rus)
plot_roc(y_test, y_pred_rf_rus, 'ROC Curve for RF After Resampling')
# Create a model (object) for classification

nb_rus = GaussianNB()



# Build a random forest classification model

nb_rus.fit(x_train_rus, y_train_rus)

y_pred_nb_rus = nb_rus.predict(x_test)

# Build a confusion matrix and show the Classification Report

cm_nb_rus = metrics.confusion_matrix(y_test,y_pred_nb_rus)

print('\nConfusion Matrix NB RUS','\n',cm_nb_rus)

print('\nClassification Report NB RUS','\n',metrics.classification_report(y_test,y_pred_nb_rus))

print("---------------------------------------------------------------------------------------")

print_result(cm_nb_rus, y_test, y_pred_nb_rus)
plot_roc(y_test, y_pred_nb_rus, 'ROC Curve for NB After Resampling')
# Normalize the data

Xn = scale(X)
# Set the 'stratify' option 'y' to sample 

Xn_train, Xn_test = train_test_split(Xn, test_size =.25,random_state=1234, stratify=y)

y_train, y_test = train_test_split(y, test_size=.25, random_state=1234, stratify=y)
# define undersample strategy

rus = RandomUnderSampler(sampling_strategy='majority')

xn_train_rus, y_train_rus = rus.fit_sample(Xn_train, y_train.ravel())



nnm_rus = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000,activation='logistic')

# nnm_smote = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000,activation='logistic')



# Make predictions

nnm_rus.fit(xn_train_rus, y_train_rus)

# nnm_smote.fit(xn_train_smote, y_train_smote)



y_pred_nn_rus = nnm_rus.predict(Xn_test)

# y_pred_nn_smote = nnm_smote.predict(Xn_test)
print('\n ** Performance Scores **')

# Build a confusion matrix and show the Classification Report

cm_nn_rus = metrics.confusion_matrix(y_test,y_pred_nn_rus)

print('\nConfusion Matrix','\n',cm_nn_rus)

print('\nClassification Report Neural Network - RUS','\n',metrics.classification_report(y_test,y_pred_nn_rus))



print("---------------------------------------------------------------------------------")
print_result(cm_nn_rus, y_test, y_pred_nn_rus)
plot_roc(y_test, y_pred_nn_rus, 'ROC Curve for NN After Resampling')