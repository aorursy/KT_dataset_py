# Library for DataFrame and Data manipulation

import pandas as pd

# Matrix operations and statistical functions

import numpy as np

# Plotting and dependency for seaborn

import matplotlib.pyplot as plt

# Graphs and chart used in this notebook

import seaborn as sns

# to convert categorical values into numerical value

from sklearn.preprocessing import LabelEncoder

# these imports are self explanatory

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
# Reading Data from csv file

data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()
# Describe some stats of numerical features apperantly there should be  numerical features

data.describe()
# Describe data type of the features , or the type of value they contain

data.info()
# There are some missing values in 'TotalCharges' but insted of representing it by 'NAN' it represents it by blank space

data[data['TotalCharges']==' ']
# extracting index of rows where TotalCharges has a white space 

ws = data[data['TotalCharges']==' '].index

# removing all those rows whose index have been extracted just now

data.drop(ws,axis=0,inplace=True)

# converting 'TotalCharges' data type from object to float, because it contains real values

data['TotalCharges'] = data.TotalCharges.astype('float')
# standardization , all values will me subtracted by column's mean and divided by cloumn's standard deviation

def standardized(data,col):

    mean = data[col].mean()

    std = data[col].std()

    data[col] = (data[col]-mean)/std

    

# this function is to plot countplot, since majority of features are categorical, i would stick to countplot most of the time

def plot_comparison(col,val,some=None):

    sub_data = data[data[col]==val]

    value = sub_data[sub_data['Churn']=='Yes']

    print('In '+str(col)+' = '+str(val)+' , it is {:.2f} % likely that customer will leave'.format((len(value)/len(sub_data)*100)))

    sns.countplot(sub_data['Churn'],palette="bright",ax=some)
data
sns.countplot(data['Churn'])

print(len(data[data['Churn']=='Yes'])/len(data)*100)
data = data.drop('customerID',axis=1)


fig,ax1 = plt.subplots(figsize=(12,6))

sns.countplot(x='gender',data=data,hue='Churn')

sns.countplot(data['SeniorCitizen'])
plot_comparison('SeniorCitizen',1)
plot_comparison('SeniorCitizen',0)
# no. of non-senior citizen churning

data['SeniorCitizen'].value_counts()[0]*23/100
# no. of senior citizen churning

data['SeniorCitizen'].value_counts()[1]*45/100
fig,ax1 = plt.subplots(1,3,figsize=(12,4))

# A pie plot to show the share of Yes and No 

data['Partner'].value_counts().plot(kind='pie')

# first plot from right shows no. of Potential Churners among Partner = 'Yes' category 

plot_comparison('Partner','Yes',ax1[0])

# plot in middle shows no. of Potential Churners among Partner = 'No' category 

plot_comparison('Partner','No',ax1[1])
fig,ax1 = plt.subplots(1,3,figsize=(12,4))

data['Dependents'].value_counts().plot(kind='pie')

plot_comparison('Dependents','Yes',ax1[0])

plot_comparison('Dependents','No',ax1[1])
# No. of Non-dependent churning

data['Dependents'].value_counts()[0]*31.28/100
sns.catplot(hue='Partner',x='Dependents',col='Churn',kind="count",data=data)
#distribution plot

sns.distplot(data['tenure'],rug=False,hist=False)
# to check mean,standard deviation,Qurtile and minimum maximum value in this feature 

data['tenure'].describe()
sns.boxplot(y=data['tenure'],x=data['Churn'])
# all rows where Churn=='Yes'

churn = data[data['Churn']=='Yes'] 

# Finding first and third quartile

Q1,Q3 = churn['tenure'].quantile([.25,.75])

# Inter Quartile range

IQR = Q3-Q1

# all the values greater than Q3+1.5*IQR and less than Q1-1.5*IQR are considered outliers 

outliers = Q3+1.5*IQR

outliers
data['tenure'].max()
# A scatter plot between every Numerical variable in the dataset

sns.pairplot(data)
data
fig,ax1 = plt.subplots(1,3,figsize=(12,4))

plot_comparison('PhoneService','Yes',ax1[1])

plot_comparison('PhoneService','No',ax1[2])

sns.countplot(data['PhoneService'],ax=ax1[0])
data['MultipleLines'] = data['MultipleLines'].map(lambda x: 'Yes' if x=='Yes' else 'No')



fig,ax1 = plt.subplots(1,3,figsize=(12,4))

plot_comparison('MultipleLines','Yes',ax1[1])

plot_comparison('MultipleLines','No',ax1[2])



sns.countplot(data['MultipleLines'],ax=ax1[0])
# Preparing Contingency table for Chi_square test 

yes = data[data['MultipleLines']=='Yes']

no = data[data['MultipleLines']=='No']

yyes = len(yes[yes['PhoneService']=='Yes'])

yno = len(yes[yes['PhoneService']=='No'])

nyes = len(no[no['PhoneService']=='Yes'])

nno = len(no[no['PhoneService']=='No'])

table = [[yyes,yno],[nyes,nno]]
# H0 : two features are dependent

# H1 : two features are Independent

from scipy.stats import chi2_contingency

from scipy.stats import chi2

# Observed Frequency table

print('Table :',table)



stat, p, dof, expected = chi2_contingency(table)

# (nrows-1)*(ncols-1) in our case (2-1)*(2-1) = 1

print('degree of Freedom=%d' % dof)

# Expeceted Frequency table

print("Expected :",expected)



prob = 0.95

# tabulated value of chi-square at 5% of significance 

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
data['InternetService'].value_counts()
fig,ax1 = plt.subplots(2,2,figsize=(12,10))

sns.countplot(data['InternetService'],ax=ax1[0,0])

plot_comparison('InternetService','DSL',ax1[0,1])

plot_comparison('InternetService','Fiber optic',ax1[1,0])

plot_comparison('InternetService','No',ax1[1,1])
sns.catplot(hue='InternetService',x='PhoneService',col='Churn',kind="count",data=data)
fig,ax1 = plt.subplots(2,2,figsize=(12,10))

sns.countplot(data['OnlineSecurity'],ax=ax1[0,0])

plot_comparison('OnlineSecurity','Yes',ax1[0,1])

plot_comparison('OnlineSecurity','No',ax1[1,0])

plot_comparison('OnlineSecurity','No internet service',ax1[1,1])
fig,ax1 = plt.subplots(2,2,figsize=(12,10))

sns.countplot(data['OnlineBackup'],ax=ax1[0,0])

plot_comparison('OnlineBackup','Yes',ax1[0,1])

plot_comparison('OnlineBackup','No',ax1[1,0])

plot_comparison('OnlineBackup','No internet service',ax1[1,1])

fig,ax1 = plt.subplots(2,2,figsize=(12,10))

sns.countplot(data['DeviceProtection'],ax=ax1[0,0])

plot_comparison('DeviceProtection','Yes',ax1[0,1])

plot_comparison('DeviceProtection','No',ax1[1,0])

plot_comparison('DeviceProtection','No internet service',ax1[1,1])


fig,ax1 = plt.subplots(2,2,figsize=(12,10))

sns.countplot(data['Contract'],ax=ax1[0,0])

plot_comparison('Contract','Month-to-month',ax1[0,1])

plot_comparison('Contract','Two year',ax1[1,0])

plot_comparison('Contract','One year',ax1[1,1])


fig,ax1 = plt.subplots(1,3,figsize=(12,4))

plot_comparison('PaperlessBilling','Yes',ax1[1])

plot_comparison('PaperlessBilling','No',ax1[2])



sns.countplot(data['PaperlessBilling'],ax=ax1[0])
fig,ax1 = plt.subplots(1,2,figsize=(12,5))

sns.boxplot(x='PaperlessBilling',y='MonthlyCharges',data=data,ax=ax1[0])

sns.boxplot(x='PaperlessBilling',y='TotalCharges',data=data,ax=ax1[1])
f,ax = plt.subplots(figsize=(12,6))

sns.countplot(data['PaymentMethod'])


fig,ax1 = plt.subplots(2,2,figsize=(12,10))

plot_comparison('PaymentMethod','Electronic check',ax1[0,0])

plot_comparison('PaymentMethod','Mailed check',ax1[0,1])

plot_comparison('PaymentMethod','Bank transfer (automatic)',ax1[1,0])

plot_comparison('PaymentMethod','Credit card (automatic)',ax1[1,1])
sns.distplot(data['MonthlyCharges'])
sns.boxplot(data['MonthlyCharges'],data['Churn'])
sns.distplot(data['TotalCharges'])
sns.boxplot(y=data['TotalCharges'],x=data['Churn'])
# don't want to alter original dataset

copy = data.copy(deep=True)
# label encoder to change categorical values into Numerical values

le = LabelEncoder()

for col in copy.columns:

    #if data type of column is object then and only apply label encoder

    if copy[col].dtypes =='object':

        copy[col] = le.fit_transform(copy[col])

        

# seperating Independent and dependent features

X = copy.drop('Churn',axis=1)

y = copy['Churn']
# to select k no. of best features from the available list of features 

from sklearn.feature_selection import SelectKBest

#using chi-square test

from sklearn.feature_selection import chi2



bestfeatures = SelectKBest(score_func=chi2, k=12)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(15,'Score'))  #print 10 best features
# will build model based on these 12 features becuase there chi-square is very high

features = ['TotalCharges','tenure','MonthlyCharges','Contract','OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection',

           'SeniorCitizen','Dependents','PaperlessBilling','Partner']
# Standardization 

X = copy[features]

y = copy['Churn']

standardized(X,'tenure')

standardized(X,'TotalCharges')

standardized(X,'MonthlyCharges')

X_train, X_test, Y_train, Y_test= train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)


# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

#models.append(())

results = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))



#Logistic Regression on test data

lr = LogisticRegression()

lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

acc = accuracy_score(Y_test,pred)

acc
# LinearDiscriminantAnalysis on test data

lda = LinearDiscriminantAnalysis()

lda.fit(X_train,Y_train)

pred = lda.predict(X_test)

acc = accuracy_score(Y_test,pred)

acc
# Support Vector machine on test data

svc = SVC(gamma='auto')

svc.fit(X_train,Y_train)

pred = svc.predict(X_test)

acc = accuracy_score(Y_test,pred)

acc