# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

import time

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, fbeta_score

from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve

df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df
df.isnull()
df.isnull().sum()
# remove 11 rows with spaces in TotalCharges (0.15% missing data)

df['TotalCharges'] = df['TotalCharges'].replace(' ',np.nan)   

df = df.dropna(how = 'any') 

df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'].isnull().sum()

# data overview

print ('Rows     : ', df.shape[0])

print ('Columns  : ', df.shape[1])

print ('\nFeatures : \n', df.columns.tolist())

print ('\nMissing values :  ', df.isnull().sum().values.sum())

print ('\nUnique values :  \n', df.nunique())

df.info()

df.isnull().sum()
print(df.Churn.value_counts())

df['Churn'].value_counts().plot('bar').set_title('Churn')
df['SeniorCitizen'].value_counts()
# replace values for SeniorCitizen as a categorical feature

df['SeniorCitizen'] = df['SeniorCitizen'].replace({1:'Yes',0:'No'})
df = df.dropna(how='all') # remove samples with null fields

df = df[~df.duplicated()] # remove duplicates

df[df.TotalCharges == ' '] # display all 11 rows with spaces in TotalCharges column (0.15% missing data)
# see all numerical columns

df.describe()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

df[num_cols].describe()
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], 

             hue='Churn', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1);
# Correlation Matrix for variables

sns.set(rc={'figure.figsize':(8,6)})

sns.heatmap(df.corr(), cmap="seismic", annot=False, vmin=-1, vmax=1)
# to view numerical features in charts

fig, ax = plt.subplots(1, 3, figsize=(15, 3))

df[num_cols].hist(bins=20, figsize=(10, 7), ax=ax)
# To analyse categorical feature distribution

# Note: senior citizens and customers without phone service are minority (less represented) in the data

# Note: "No Internet Service" is a repeated feature in 6 other charts



categorical_features = [

 'gender',

 'SeniorCitizen',

 'Partner',

 'Dependents',

 'PhoneService',

 'MultipleLines',

 'InternetService',

 'OnlineSecurity',

 'OnlineBackup',

 'DeviceProtection',

 'TechSupport',

 'StreamingTV',

 'StreamingMovies',

 'PaymentMethod',

 'PaperlessBilling',

 'Contract' ]



ROWS, COLS = 4, 4

fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 20) )

row, col = 0, 0

for i, categorical_feature in enumerate(categorical_features):

    if col == COLS - 1:

        row += 1

    col = i % COLS

#     df[categorical_feature].value_counts().plot('bar', ax=ax[row, col]).set_title(categorical_feature)

    df[df.Churn=='No'][categorical_feature].value_counts().plot('bar', 

                width=.5, ax=ax[row, col], color='blue', alpha=0.5).set_title(categorical_feature)

    df[df.Churn=='Yes'][categorical_feature].value_counts().plot('bar', 

                width=.3, ax=ax[row, col], color='orange', alpha=0.7).set_title(categorical_feature)

    plt.legend(['No Churn', 'Churn'])

    fig.subplots_adjust(hspace=0.7)


# to look at Contract & Payment Method in relation to the target variable

# note: users who have a month-to-month contract and Electronic check PaymentMethod are more likely to churn

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

df[df.Churn == 'No']['Contract'].value_counts().plot('bar', ax=ax[0], color='blue', alpha=0.5).set_title('Contract')

df[df.Churn == 'Yes']['Contract'].value_counts().plot('bar', width=.3, ax=ax[0], color='orange', alpha=0.7)

df[df.Churn == 'No']['PaymentMethod'].value_counts().plot('bar', ax=ax[1], color='blue', alpha=0.5).set_title('PaymentMethod')

df[df.Churn == 'Yes']['PaymentMethod'].value_counts().plot('bar', width=.3, ax=ax[1], color='orange', alpha=0.7)

plt.legend(['No Churn', 'Churn'])
# look at distributions of numerical features in relation to the target variable

# the greater TotalCharges and tenure are the less is the probability of churn



fig, ax = plt.subplots(1, 3, figsize=(15, 3))

df[df.Churn == "No"][num_cols].hist(bins=35, color="blue", alpha=0.5, ax=ax)

df[df.Churn == "Yes"][num_cols].hist(bins=35, color="orange", alpha=0.7, ax=ax)

plt.legend(['No Churn', 'Churn'], shadow=True, loc=9)
# change MonthlyCharges to categorical column

def monthlycharges_split(df) :   

    if df['MonthlyCharges'] <= 30 :

        return '0-30'

    elif (df['MonthlyCharges'] > 30) & (df['MonthlyCharges'] <= 70 ):

        return '30-70'

    elif (df['MonthlyCharges'] > 70) & (df['MonthlyCharges'] <= 99 ):

        return '70-99'

    elif df['MonthlyCharges'] > 99 :

        return '99plus'

df['monthlycharges_group'] = df.apply(lambda df:monthlycharges_split(df), axis = 1)



# change TotalCharges to categorical column

def totalcharges_split(df) :   

    if df['TotalCharges'] <= 2000 :

        return '0-2k'

    elif (df['TotalCharges'] > 2000) & (df['TotalCharges'] <= 4000 ):

        return '2k-4k'

    elif (df['TotalCharges'] > 4000) & (df['TotalCharges'] <= 6000) :

        return '4k-6k'

    elif df['TotalCharges'] > 6000 :

        return '6kplus'

df['totalcharges_group'] = df.apply(lambda df:totalcharges_split(df), axis = 1)



# change Tenure to categorical column

def tenure_split(df) :   

    if df['tenure'] <= 20 :

        return '0-20'

    elif (df['tenure'] > 20) & (df['tenure'] <= 40 ):

        return '20-40'

    elif (df['tenure'] > 40) & (df['tenure'] <= 60) :

        return '40-60'

    elif df['tenure'] > 60 :

        return '60plus'

df['tenure_group'] = df.apply(lambda df:tenure_split(df), axis = 1)



# # Separating categorical and numerical columns

# Id_col     = ['customerID']

# target_col = ['Churn']

# cat_cols   = df.nunique()[df.nunique() < 6].keys().tolist()

# cat_cols   = [x for x in cat_cols if x not in target_col]

# num_cols   = [x for x in df.columns if x not in cat_cols + target_col + Id_col]



# target_col
# new features monthlycharges_group

plt.figure(figsize = [10,5])

df[df.Churn == "No"]['monthlycharges_group'].value_counts().plot('bar', color="blue", alpha=0.5).set_title('monthlycharges_group')

df[df.Churn == "Yes"]['monthlycharges_group'].value_counts().plot('bar', color="orange", alpha=0.7, width=0.3)

plt.legend(['No Churn', 'Churn'], shadow=True, loc=1)
# new features totalcharges_group

plt.figure(figsize = [10,5])

df[df.Churn == "No"]['totalcharges_group'].value_counts().plot('bar', color="blue", alpha=0.5).set_title('totalcharges_group')

df[df.Churn == "Yes"]['totalcharges_group'].value_counts().plot('bar', color="orange", alpha=0.7, width=0.3)

plt.legend(['No Churn', 'Churn'], shadow=True, loc=1)
# new features tenure_group

plt.figure(figsize = [10,5])

df[df.Churn == "No"]['tenure_group'].value_counts().plot('bar', color="blue", alpha=0.5).set_title('tenure_group')

df[df.Churn == "Yes"]['tenure_group'].value_counts().plot('bar', color="orange", alpha=0.7, width=0.3)

plt.legend(['No Churn', 'Churn'], shadow=True, loc=1)
# store df to csv file

df.to_csv('/kaggle/working/df.csv', index=False)

df = pd.read_csv('/kaggle/working/df.csv')

# Data preprocessing



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



# customer id col

Id_col     = ['customerID']

# Target columns

target_col = ['Churn']

#categorical columns

cat_cols   = df.nunique()[df.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

#numerical columns

num_cols   = [x for x in df.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values

bin_cols   = df.nunique()[df.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]



#Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    df[i] = le.fit_transform(df[i])

    

#Duplicating columns for multi value columns

df = pd.get_dummies(data = df, columns = multi_cols)



#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(df[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)



#dropping original values merging scaled values for numerical columns

df1 = df.drop(columns = num_cols, axis = 1)

df1 = df1.merge(scaled, left_index=True, right_index=True, how = "left")



# note: df has 21 columns including unscaled num_cols; df1 has 54 columns including scaled num_cols

# I defined 2 separate df & df1 for comparison, to check if the columns are correctly labelled after encoding/get_dummies
# check if there is any null fields (ie, ensure all fields are filled)

df1[df1.TotalCharges.isnull()]

df1.describe()

df1.columns

df1.dtypes

# Correlation Matrix for variables

sns.set(rc={'figure.figsize':(15,13)})

sns.heatmap(df1.corr(), cmap="seismic", annot=False, vmin=-1, vmax=1)


# drop 'customerID' column, feature not needed in model selection

df1 = df1.drop('customerID', axis=1)



# there are a lot of repeated features (no internet service), so drop them

df1 = df1.drop(columns=['OnlineSecurity_No internet service', 'OnlineBackup_No internet service', 

                        'DeviceProtection_No internet service', 'TechSupport_No internet service', 

                        'StreamingTV_No internet service', 'StreamingMovies_No internet service'], axis=1)



# original 54 columns, reduced to 47 columns
df1.columns
# Correlation Matrix for variables

sns.set(rc={'figure.figsize':(12,10)})

sns.heatmap(df1.corr(), cmap="seismic", annot=False, vmin=-1, vmax=1)
# store df1 to csv file

df1.to_csv('/kaggle/working/df1.csv', index=False)
df1 = pd.read_csv('df1.csv')

X, y = df1.drop('Churn',axis=1), df1[['Churn']]



import statsmodels.api as sm

X = sm.add_constant(X)  # need to add this to define the Intercept

# model / fit / summarize results

model = sm.OLS(y, X)

result = model.fit()

result.summary()
## to find significant features using LassoCV (all X_scaled)

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from sklearn.preprocessing import StandardScaler, PolynomialFeatures



print('Use LassoCV to find the optimal ALPHA value for L1 regularization')

# Scale the Predictors on both the train and validation set

std = StandardScaler()

std.fit(X.values)

X_scaled = std.transform(X.values)

print('X_scaled', X_scaled.shape)

# Run the cross validation, find the best alpha, refit the model on all the data with that alpha

alphavec = 10**np.linspace(-3,3,200)   # alpha varies from 0.001 to 1000

lasso_model = LassoCV(alphas = alphavec, cv=5)

lasso_model.fit(X_scaled, y)

# This is the best alpha value found

print('LASSO best alpha: ', lasso_model.alpha_ )

# display all coefficients in the model with optimal alpha

list(zip(X.columns, lasso_model.coef_))
# see if you can extract the above results using Regular Expression



plot_feature = ['TotalCharges', 'InternetService_Fiber optic', 'tenure_group_60plus', 'tenure', 'Contract_Month-to-month', 

                'totalcharges_group_6kplus', 'monthlycharges_group_99plus', 'PaymentMethod_Electronic check', 

                'totalcharges_group_0-2k', 'OnlineSecurity_No', 'TechSupport_No', 'tenure_group_40-60', 

                'totalcharges_group_4k-6k', 'PaperlessBilling', 'StreamingTV_Yes', 'MultipleLines_No', 'StreamingMovies_Yes', 

                'SeniorCitizen', 'monthlycharges_group_70-9', 'tenure_group_20-40', 'OnlineBackup_No', 'MonthlyCharges', 

                'monthlycharges_group_0-30', 'Dependents', 'InternetService_No', 'MultipleLines_Yes', 'DeviceProtection_No', 

                'Contract_One year', 'PaymentMethod_Mailed check', 'gender', 'PaymentMethod_Credit card (automatic)']



lasso_coeff = [0.209954752, 0.075144498, 0.061184581, 0.061182631, 0.046630292, 0.036007041, 0.034846244, 0.031775227, 

               0.029645254, 0.024949481, 0.024875392, 0.024679595, 0.021639644, 0.020966614, 0.020143496, 0.019954793, 

               0.019936301, 0.016463024, 0.015436581, 0.012221305, 0.011015587, 0.008054301, 0.007701626, 0.006895811, 

               0.00642757, 0.005009993, 0.002481356, 0.002102214, 0.001449537, 0.001066809, 0.000525379]



sns.barplot(y = plot_feature, x = lasso_coeff, color='b')


## To look for top features using Random Forest

# Create decision tree classifer object

rfc = RandomForestClassifier(random_state=0, n_estimators=100)



# Train model, note that NO scaling is required

model = rfc.fit(X, y)



# Plot the top features based on its importance

(pd.Series(model.feature_importances_, index=X.columns)

   .nlargest(47)   # can adjust based on how many top features you want

   .plot(kind='barh', figsize=[20,15])

    .invert_yaxis()) # Ensures that the feature with the most importance is on top, in descending order



plt.yticks(size=15)

plt.title('Top Features derived by Random Forest', size=20)
df1 = pd.read_csv('df1.csv')

X, y = df1.drop('Churn',axis=1), df1[['Churn']]
# split data to 80:20 ratio for train/test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=71)

print('X_train', X_train.shape)

print('y_train', y_train.shape)

print('X_test', X_test.shape)

print('y_test', y_test.shape)



def model_report(model_name, model):

    print('\nSearch for OPTIMAL THRESHOLD, vary from 0.0001 to 0.9999, fit/predict on train/test data')

    model.fit(X_train, y_train)

    optimal_th = 0.5   # start with default threshold value

    

    for i in range(0,3):

        score_list = []

        print('\nLooping decimal place', i+1) 

        th_list = [np.linspace(optimal_th-0.4999, optimal_th+0.4999, 11), 

                  # eg [ 0.0001 , 0.1008, 0.2006, 0.3004, 0.4002, 0.5, 0.5998, 0.6996, 0.7994, 0.8992, 0.9999 ]

                 np.linspace(optimal_th-0.1, optimal_th+0.1, 21), 

                  # eg 0.3xx [ 0.2 , 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 ]

                 np.linspace(optimal_th-0.01, optimal_th+0.01, 21)]

                  # eg 0.30x [ 0.29 , 0.291, 0.292, 0.293, 0.294, 0.295, 0.296, 0.297, 0.298, 0.299, 0.3  , 0.301, 0.302, 0.303, 0.304, 0.305, 0.306, 0.307, 0.308, 0.309, 0.31 ]

        for th in th_list[i]:

            y_pred = (model.predict_proba(X_test)[:,1] >= th)

            f1scor = f1_score(y_test, y_pred)

            score_list.append(f1scor)

            print('{:.3f}->{:.4f}'.format(th, f1scor), end=',  ')   # display score in 4 decimal pl

        optimal_th = float(th_list[i][score_list.index(max(score_list))])



    print('optimal F1 score = {:.4f}'.format(max(score_list)))

    print('optimal threshold = {:.3f}'.format(optimal_th))



    print(model_name, 'accuracy score is')

    print('Training: {:.2f}%'.format(100*model.score(X_train, y_train)))  # score uses accuracy

    print('Test set: {:.2f}%'.format(100*model.score(X_test, y_test)))   # should use cross validation



    y_pred = (model.predict_proba(X_test)[:,1] >= 0.25)

    print('\nAdjust threshold to 0.25:')

    print('Precision: {:.4f},   Recall: {:.4f},   F1 Score: {:.4f}'.format(

        precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

    print(model_name, 'confusion matrix: \n', confusion_matrix(y_test, y_pred))



    y_pred = model.predict(X_test)

    print('\nDefault threshold of 0.50:')

    print('Precision: {:.4f},   Recall: {:.4f},   F1 Score: {:.4f}'.format(

        precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

    print(model_name, 'confusion matrix: \n', confusion_matrix(y_test, y_pred))



    y_pred = (model.predict_proba(X_test)[:,1] >= 0.75)

    print('\nAdjust threshold to 0.75:')

    print('Precision: {:.4f},   Recall: {:.4f},   F1 Score: {:.4f}'.format(

        precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

    print(model_name, 'confusion matrix: \n', confusion_matrix(y_test, y_pred))



    y_pred = (model.predict_proba(X_test)[:,1] >= optimal_th)

    print('\nOptimal threshold {:.3f}'.format(optimal_th))

    print('Precision: {:.4f},   Recall: {:.4f},   F1 Score: {:.4f}'.format(

        precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

    print(model_name, 'confusion matrix: \n', confusion_matrix(y_test, y_pred))

    

    global model_f1, model_auc, model_ll, model_roc_auc

    model_f1 = f1_score(y_test, y_pred)



    y_pred = model.predict_proba(X_test)

    model_ll = log_loss(y_test, y_pred)

    print(model_name, 'Log-loss: {:.4f}'.format(model_ll))

    y_pred = model.predict(X_test)

    model_roc_auc = roc_auc_score(y_test, y_pred)

    print(model_name, 'roc_auc_score: {:.4f}'.format(model_roc_auc)) 

    y_pred = model.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    model_auc = auc(fpr, tpr)

    print(model_name, 'AUC: {:.4f}'.format(model_auc))



    # plot the ROC curve

    plt.figure(figsize = [6,6])

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % model_auc)

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic')

    plt.legend(loc="lower right")

    # plt.savefig('roc_auc_score')

    plt.show()

  

    return



# initialise lists to collect the results to plot later

model_list = []

f1_list = []

auc_list = []

ll_list = []

roc_auc_list = []

time_list = []
print('\n"""""" GaussianNB """"""')

time1 = time.time()

gnb = GaussianNB()

model_report('GaussianNB', gnb)



model_list.append('GaussianNB')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)



print('\n"""""" BernoulliNB """"""')

time1 = time.time()

bnb = BernoulliNB()

model_report('BernoulliNB', bnb)



model_list.append('BernoulliNB')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)



### this model does not work

# print('\n"""""" MultinomialNB """"""')

# mnb = MultinomialNB()

# model_report('MultinomialNB', mnb)
print('\n"""""" LogisticRegression """"""')

print('\nSearch for optimal hyperparameter C in LogisticRegresssion, vary C from 0.001 to 1000, using KFold(5) Cross Validation on train data')

kf = KFold(n_splits=5, random_state=21, shuffle=True)  #produce the k folds

score_list = []

c_list = 10**np.linspace(-3,3,200)

for c in c_list:

    logit = LogisticRegression(C = c)

    cvs = (cross_val_score(logit, X_train, y_train, cv=kf, scoring='f1')).mean()

    score_list.append(cvs)

    print('{:.4f}'.format(cvs), end=", ")   # 4 decimal pl

print('optimal cv F1 score = {:.4f}'.format(max(score_list)))

optimal_c = float(c_list[score_list.index(max(score_list))])

print('optimal value of C = {:.3f}'.format(optimal_c))



time1 = time.time()

logit = LogisticRegression(C = optimal_c)

model_report('LogisticRegression', logit)



model_list.append('LogisticRegression')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)

print('\n"""""" KNN """""" (quite slow)')

print('\nSearch for optimal hyperparameter K in KNN, vary K from 1 to 20, using KFold(5) Cross Validation on train data')

kf = KFold(n_splits=5, random_state=21, shuffle=True)  #produce the k folds

k_scores = []

for k in range(1, 21):

    knn = KNeighborsClassifier(n_neighbors = k)

    cvs = cross_val_score(knn, X_train, y_train, cv=kf, scoring='f1').mean()

    k_scores.append(cvs)

    print('{:.4f}'.format(cvs), end=", ")

print('optimal cv F1 score = {:.4f}'.format(max(k_scores)))   # 4 decimal pl

optimal_k = k_scores.index(max(k_scores))+1   # index 0 is for k=1

print('optimal value of K =', optimal_k)



time1 = time.time()

knn = KNeighborsClassifier(n_neighbors = optimal_k)

model_report('KNN', knn)



print('\nCompare with KNN classification_report (same as default threshold 0.50)')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.classification_report(y_test, y_pred))



model_list.append('KNN')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)
print('\n"""""" DecisionTreeClassifier """"""')



print('\nSearch for optimal max_depth in DecisionTree, vary 2 to 10, using KFold(5) Cross Validation on train data')

kf = KFold(n_splits=5, random_state=21, shuffle=True)  #produce the k folds

d_scores = []

for d in range(2, 11):

    decisiontree = DecisionTreeClassifier(max_depth=d)

    cvs = cross_val_score(decisiontree, X_train, y_train, cv=kf, scoring='f1').mean()

    d_scores.append(cvs)

    print('{:.4f}'.format(cvs), end=", ")

print('optimal F1 score = {:.4f}'.format(max(d_scores)))   # 4 decimal pl

optimal_d = d_scores.index(max(d_scores))+2   # index 0 is for d=2

print('optimal max_depth =', optimal_d)



time1 = time.time()

decisiontree = DecisionTreeClassifier(max_depth=optimal_d)

model_report('DecisionTreeClassifier', decisiontree)



model_list.append('DecisionTreeClassifier')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)
print('\n"""""" RandomForestClassifier """""" (quite slow)')



print('\nSearch for optimal n_estimators in RandomForest, vary 100 to 500, using KFold(5) Cross Validation on train data')

kf = KFold(n_splits=5, random_state=21, shuffle=True)  #produce the k folds

score_list = []

n_list = []

for n in [100, 150, 200, 250, 300, 350, 400, 450, 500]:

    randomforest = RandomForestClassifier(n_estimators=n)

    cvs = (cross_val_score(randomforest, X_train, y_train, cv=kf, scoring='f1')).mean()

    score_list.append(cvs)

    n_list.append(n)

    print('{:.0f}->{:.4f}'.format(n, cvs), end=", ")   # display score in 4 decimal place

print('optimal F1 score = {:.4f}'.format(max(score_list)))

optimal_n = int(n_list[score_list.index(max(score_list))])

print('optimal n_estimators = {:.0f}'.format(optimal_n))



time1 = time.time()

randomforest = RandomForestClassifier(n_estimators=optimal_n)

model_report('RandomForestClassifier', randomforest)



model_list.append('RandomForestClassifier')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)
print('\n"""""" LinearSVC """"""')

time1 = time.time()

linearsvc = LinearSVC()

# model_report('LinearSVC', linearsvc)   # model has no attribute 'predict_proba'

linearsvc.fit(X_train, y_train)

print('LinearSVC accuracy score is')

print('Training: {:.2f}%'.format(100*linearsvc.score(X_train, y_train)))  # score uses accuracy

print('Test set: {:.2f}%'.format(100*linearsvc.score(X_test, y_test)))   # should use cross validation



y_pred = linearsvc.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

print('LinearSVC confusion matrix: \n', confusion_matrix(y_test, y_pred))



model_f1 = f1_score(y_test, y_pred)



model_ll = log_loss(y_test, y_pred)

print('LinearSVC Log-loss: {:.4f}'.format(model_ll))

model_roc_auc = roc_auc_score(y_test, y_pred)

print('LinearSVC roc_auc_score: {:.4f}'.format(model_roc_auc)) 

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

model_auc = auc(fpr, tpr)

print('LinearSVC AUC: {:.4f}'.format(model_auc))



# plot the ROC curve

plt.figure(figsize = [6,6])

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % model_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

# plt.savefig('roc_auc_score')

plt.show()



model_list.append('LinearSVC')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

time_list.append(time.time() - time1)
print('\n"""""" SVC """""" (extremely slow)')

time1 = time.time()

svc = SVC(gamma='scale', probability=True)

model_report('SVC', svc)



model_list.append('SVC')

f1_list.append(model_f1)

auc_list.append(model_auc)

ll_list.append(model_ll)

roc_auc_list.append(model_roc_auc)

# time_list.append(time.time() - time1)   # use this line for actual time spent, or

time_list.append(0)                       # use this line to be able to see time spent for other models
## plot the classification report scores

fig, ax = plt.subplots(5, 1, figsize=(18, 28))

# fig.set_figwidth(10)

# fig.set_figheight(6)

# fig.suptitle('Main Title',fontsize = 16)

ax[0].bar(model_list, f1_list)

ax[0].set_title('F1-score')

ax[1].bar(model_list, auc_list)

ax[1].set_title('AUC-score');

ax[2].bar(model_list, ll_list)

ax[2].set_title('Log-Loss-Score')

ax[3].bar(model_list, roc_auc_list)

ax[3].set_title('ROC AUC Score')

ax[4].bar(model_list, time_list)

ax[4].set_title('Time taken')

# Fine-tune figure; make subplots farther from each other, or nearer to each other.

fig.subplots_adjust(hspace=0.2, wspace=0.2)
# plot the ROC curves

plt.figure(figsize=(10,10))



y_pred = gnb.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='blue',

        lw=3, label='GaussianNB (area = %0.2f)' % auc_list[0])



y_pred = bnb.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='green',

        lw=3, label='BernoulliNB (area = %0.2f)' % auc_list[1])



y_pred = logit.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='red',

        lw=2, label='LogisticRegression (area = %0.2f)' % auc_list[2])



y_pred = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='yellow',

        lw=3, label='KNN (area = %0.2f)' % auc_list[3])



y_pred = decisiontree.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='purple',

        lw=2, label='DecisionTree (area = %0.2f)' % auc_list[4])



y_pred = randomforest.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='brown',

        lw=2, label='RandomForest (area = %0.2f)' % auc_list[5])



y_pred = linearsvc.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='cyan',

        lw=2, label='LinearSVC (area = %0.2f)' % auc_list[6])



y_pred = svc.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='magenta',

        lw=2, label='SVC (area = %0.2f)' % auc_list[7])



plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate', fontsize=13)

plt.ylabel('True Positive Rate', fontsize=14)

plt.title('Receiver Operating Characteristic', fontsize=17)

plt.legend(loc='lower right', fontsize=13)

plt.show()
def make_confusion_matrix(model, threshold=0.5):

    # Predict class 1 if probability of being in class 1 is greater than threshold

    # (model.predict(X_test) does this automatically with a threshold of 0.5)

    y_pred = (logit.predict_proba(X_test)[:, 1] >= threshold)

    conf = confusion_matrix(y_test, y_pred)

    plt.figure(figsize = [5,5])

    sns.heatmap(conf, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',

           xticklabels=['no churn', 'churn'],

           yticklabels=['no churn', 'churn']);

    plt.xlabel('prediction')

    plt.ylabel('actual')

# Let's see how our confusion matrix changes with changes to the cutoff! 

from ipywidgets import interactive, FloatSlider

logit = LogisticRegression(C = optimal_c)

logit.fit(X_train, y_train)

interactive(lambda threshold: make_confusion_matrix(logit, threshold), threshold=(0.0,1.0,0.01))