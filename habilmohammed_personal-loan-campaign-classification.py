# import the necessary packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.naive_bayes import BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

from sklearn import svm

from sklearn.utils import resample

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Read the dataset from csv file and print the first 10 rows

data = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')

data.head(10)
print('Number of features in the data : ', data.shape[1])

print('Number of samples in the data  : ', data.shape[0])
# List the information about each column such as non null count, datatype etc.

data.info()
# gives an idea about distribution of each attribute

data.describe().T
data[data['Experience'] < 0].shape
sns.pairplot(data.iloc[:,1:])
# Lets get the categorical features. We consider the features which are having less than 25 values as categorical.

cat_features = [feature for feature in data.columns if data[feature].nunique()<25]

print('Categorical Features : ', cat_features)

data[cat_features].head()
# Continues features

cont_features = [feature for feature in data.columns if feature not in cat_features + ['ID']]

cont_features
for feature in cat_features:

    sns.countplot(data[feature])

    plt.show()
for feature in cont_features:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    sns.distplot(data[feature], kde=False, ax=ax1)

    sns.boxplot(data[feature], ax=ax2)

    plt.show()

sns.scatterplot(x='Income',y='CCAvg',hue='Personal Loan', data=data)
plt.figure(figsize=(15,8))

sns.pointplot(x='Income',y='CCAvg', data=data, estimator=np.median, ci=None)
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=data)
plt.figure(figsize=(15,8))

sns.boxplot(x='Experience',y='Income',hue='Personal Loan',data=data)
print(data['ZIP Code'].min())

print(data['ZIP Code'].max())

data['ZIP Code'].unique()
data['ZIP Code'] = data['ZIP Code'].apply(lambda zip: zip*10 if zip/10000 < 1 else zip)
sns.boxplot(data['ZIP Code'])
zipBins = np.arange(90000, 99999, 500)

zipCat = pd.cut(data['ZIP Code'], bins = zipBins)

data['ZIPCat'] = zipCat

data.head()
woe_df = data.groupby('ZIPCat')['Personal Loan'].mean()

woe_df = pd.DataFrame(woe_df)

woe_df = woe_df.rename(columns = {'Personal Loan': 'Good'})

woe_df['Bad'] = 1 - woe_df['Good']

woe_df['Good'] = np.where(woe_df['Good'] == 0, 0.000001, woe_df['Good'])

woe_df['Bad'] = np.where(woe_df['Bad'] == 0, 0.000001, woe_df['Bad'])

woe_df['WoE'] = np.log(woe_df['Good']/woe_df['Bad'])

data.loc[:, 'ZIP_WoE'] = data['ZIPCat'].map(woe_df['WoE'])

data
print('Number of customers with No House mortgage: ',data[data['Mortgage'] == 0].shape[0], ' out of ', data.shape[0])

print('Number of customers with No spending on credit card: ',data[data['CCAvg'] == 0].shape[0], ' out of ', data.shape[0])
# If CCAvg > 0, set the CreditCard as 1, else 0

data['CreditCard'] = np.where(data['CCAvg'] > 0 , 1, 0)
print('Number of rows with Negative Experience : ', sum(data['Experience'] < 0))

print('Ages of rows rows with Negative Experince : ', data[data['Experience'] < 0].Age.unique())
sns.lmplot('Age', 'Experience', data=data[data['Experience'] > 0])
lr_model = LinearRegression()

lr_model.fit(data[['Age']], data['Experience'])

data.loc[data['Experience'] < 0, 'Experience'] = lr_model.predict( data[data['Experience'] < 0].loc[:,'Age':'Age'] )
print('Number of rows with Negative Experience : ', sum(data['Experience'] < 0))

print('Ages of rows rows with Negative Experince : ', data[data['Experience'] < 0].Age.unique())
data.loc[data['Experience'] < 0, 'Experience'] = 0

data[data['Experience'] < 0]
sns.scatterplot(data['Age'], data['Experience'])
plt.figure(figsize=(16,8))

sns.heatmap(data.corr(), annot=True)
# Lets remove the unwanted variables split the data to X and y

X = data.drop(['ID', 'ZIP Code', 'ZIPCat','Personal Loan'], axis = 1)

y = data['Personal Loan']

X
vif=pd.DataFrame()

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by='VIF', ascending=False)

vif
X.drop('Age', axis = 1, inplace=True)

vif=pd.DataFrame()

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by='VIF', ascending=False)

vif
X.drop('CreditCard', axis = 1, inplace=True)

vif=pd.DataFrame()

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by='VIF', ascending=False)

vif
X.drop('ZIP_WoE', axis = 1, inplace=True)

vif=pd.DataFrame()

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by='VIF', ascending=False)

vif
# Split the data to test and train set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)



print('Number of 0s in Train dataset : ', y_train.value_counts()[0])

print('Number of 1s in Train dataset : ', y_train.value_counts()[1])

print('----------------')

print('Number of 0s in Test dataset : ', y_test.value_counts()[0])

print('Number of 1s in Test dataset : ', y_test.value_counts()[1])



X_train = X_train.apply(zscore)

X_test = X_test.apply(zscore)
# Function to print the different metrics such as confusion matrix, roc, accuracy, precision, recall etc



def printModelMetrics(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    print('Train set accuracy = ', metrics.accuracy_score(y_train, y_train_pred))

    print('Test set accuracy = ', metrics.accuracy_score(y_test, y_test_pred))

    print(metrics.classification_report(y_test, y_test_pred))



    cm = metrics.confusion_matrix(y_test, y_test_pred)

    cm = pd.DataFrame(cm, columns=['Predicted No Loan', 'Predicted Loan'], index=['Truth No Loan', 'Truth Loan'])

    sns.heatmap(cm, annot=True, fmt='g', cbar=False)

    plt.show()



    y_test_proba = model.predict_proba(X_test)

    y_test_proba = y_test_proba[:,1]

    # generate a no skill prediction (majority class)

    ns_probs = [0 for _ in range(len(y_test))]

    # calculate scores

    ns_auc = metrics.roc_auc_score(y_test, ns_probs)

    lr_auc = metrics.roc_auc_score(y_test, y_test_proba)

    # summarize scores

    print('ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves

    ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)

    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, y_test_proba)

    # plot the roc curve for the model

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model Skill')

    

    # axis labels

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    # show the legend

    plt.legend()

    # show the plot

    plt.show()
lr_model = LogisticRegression(solver='liblinear')

lr_model.fit(X_train, y_train)

printModelMetrics(lr_model, X_train, X_test, y_train, y_test)
# Naive Bayes 

nb_model = BernoulliNB()



nb_model.fit(X_train, y_train)

printModelMetrics(lr_model, X_train, X_test, y_train, y_test)
# KNN Classifier

X_trainScaled = X_train.apply(zscore)

X_testScaled = X_test.apply(zscore)



model = KNeighborsClassifier(n_neighbors=20, weights='distance')

model.fit(X_trainScaled, y_train)

printModelMetrics(model, X_trainScaled, X_testScaled, y_train, y_test)


model = svm.SVC(gamma = 0.025, C=3, probability=True)

model.fit(X_train, y_train)

printModelMetrics(model, X_train, X_test, y_train, y_test)
#Lets oversample the positive classes in the data



# Separate majority and minority classes

df_majority = data[data['Personal Loan']==0]

df_minority = data[data['Personal Loan']==1]



# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=df_majority.shape[0],    # to match majority class

                                 random_state=123) # reproducible results



#Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

df_upsampled['Personal Loan'].value_counts()



X = df_upsampled.drop(['ID', 'ZIP Code', 'ZIPCat', 'Personal Loan', 'Experience'], axis = 1)

y = df_upsampled['Personal Loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)



print('Number of 0s in Train dataset : ', y_train.value_counts()[0])

print('Number of 1s in Train dataset : ', y_train.value_counts()[1])

print('----------------')

print('Number of 0s in Test dataset : ', y_test.value_counts()[0])

print('Number of 1s in Test dataset : ', y_test.value_counts()[1])
lr_model = LogisticRegression(solver='liblinear')

lr_model.fit(X_train, y_train)

printModelMetrics(lr_model, X_train, X_test, y_train, y_test)
nb_model = BernoulliNB()



nb_model.fit(X_train, y_train)

printModelMetrics(lr_model, X_train, X_test, y_train, y_test)
X_trainScaled = X_train.apply(zscore)

X_testScaled = X_test.apply(zscore)



model = KNeighborsClassifier(n_neighbors=20, weights='distance')

model.fit(X_trainScaled, y_train)

printModelMetrics(model, X_trainScaled, X_testScaled, y_train, y_test)
model = svm.SVC(gamma = 0.2, C=5, probability=True)

model.fit(X_train, y_train)

printModelMetrics(model, X_train, X_test, y_train, y_test)