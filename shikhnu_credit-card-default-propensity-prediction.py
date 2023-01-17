# Importing important libraries



import pandas            as pd

import numpy             as np

import matplotlib.pyplot as plt

import seaborn           as sns

import statsmodels.api   as sm



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection   import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model      import LogisticRegression

from sklearn.metrics           import classification_report

from sklearn.tree              import DecisionTreeClassifier

from sklearn.ensemble          import RandomForestClassifier

from scipy.stats               import randint as sp_randint

from imblearn.over_sampling    import SMOTE
# Read the dataset and display first five rows



df = pd.read_csv('../input/credit_card.csv')

df.head()
print("There are {} rows and {} columns in the dataset.".format(df.shape[0],df.shape[1]))
# To see the datatypes of the column



df.info()
# Five point summary of the dataset



df.describe().T
print("There are {} missing records in the dataset.".format(df.isnull().sum().sum()))
# Storing feature names in variable 'cols'



cols = df.columns.tolist()
for i in [ 'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:

    plt.figure(figsize=(10,5))

    sns.countplot(df[i])

    plt.show()
# Boxplot for Bill_Amt vs Limit_bal



plt.figure(figsize=(10,7))

sns.boxplot(data=df[['LIMIT_BAL','BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])

plt.show()
# Boxplot for Pay_Amt vs Limit_bal



plt.figure(figsize=(10,7))

sns.boxplot(data=df[['LIMIT_BAL','PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])

plt.show()
# Boxplot for column 'AGE'

plt.figure(figsize=(5,5))

sns.boxplot(data=df['AGE'])

plt.show()
# Outliers on numberical columns



num_var = df.select_dtypes(exclude='object')

for i in num_var:

    

    q1 = df[i].quantile(0.25)

    q3 = df[i].quantile(0.75)



    IQR = q3 - q1

    UL = q3 + 1.5*IQR

    LL = q1 - 1.5*IQR



    print('IQR of',i,'= ',IQR)

    print('UL of',i,'= ',UL)

    print('LL of',i,'= ',LL)

    print('Number of Outliers in',i,' = ',(df.shape[0] - df[(df[i]<UL) & (df[i]>LL)].shape[0]))

    print(' ')
mi0 = df[df['DEFAULT']==0]

mi1 = df[df['DEFAULT']==1]
con_col=['AGE','LIMIT_BAL','BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']



for i in con_col:

    plt.figure(figsize=(20,5))

    sns.distplot(mi0[i],color='g')

    sns.distplot(mi1[i],color='r')

    plt.show()
plt.figure(figsize=(25,20))

sns.heatmap(df.corr(),annot=True)

plt.show()
sns.pairplot(df)

plt.show()
def age(x):

    if x in range(21,41):

        return 1

    elif x in range(41,61):

        return 2

    elif x in range(61,80):

        return 3



df['AGE']=df['AGE'].apply(age)
def bins(x):

    if x == -2:

        return 'Paid Duly'

    if x == 0:

        return 'Paid Duly'

    if x == -1:

        return 'Paid Duly'

    if x in range(1,4):

        return '1 to 3'

    if x in range(4,7):

        return '4 to 6'

    if x in range(7,9):

        return '7 to 9'



for i in df[['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]:

    df[i]=df[i].apply(bins)
def rep(x):

    if x in [0,4,5,6]:

        return 4

    else:

        return x

df['EDUCATION']=df.EDUCATION.apply(rep)
# Dataset after feature engineering



df.head()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



for col in df.select_dtypes(include=object).columns:

    df[col] = le.fit_transform(df[col])
X =df.drop('DEFAULT',axis=1)

y = df['DEFAULT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print((df['DEFAULT'].value_counts()/df['DEFAULT'].shape)*100)

sns.countplot(df['DEFAULT'])

plt.show()
print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape)) 

print('Before OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
smote = SMOTE(sampling_strategy='minority')

X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
print('After OverSampling, the shape of train_X: {}'.format(X_train_sm.shape)) 

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_sm.shape))
logreg = LogisticRegression(solver='liblinear', fit_intercept=True)



logreg.fit(X_train_sm, y_train_sm)



y_prob_train = logreg.predict_proba(X_train_sm)[:,1]

y_pred_train = logreg.predict (X_train_sm)



print('Classification report - Train: ', '\n', classification_report(y_train_sm, y_pred_train))



y_prob = logreg.predict_proba(X_test)[:,1]

y_pred = logreg.predict (X_test)



print('Classification report - Test: ','\n', classification_report(y_test, y_pred))
Xc=sm.add_constant(X_train_sm)



model = sm.Logit ( y_train_sm , Xc ).fit ( )

model.summary ( )
cols = list(X_train_sm.columns)

pmax = 1

while (len(cols)>0):

    p= []

    X_1 = X_train_sm[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y_train_sm,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print("Important features: {}".format(selected_features_BE))

print("\nNumber of important features: {}".format(len(selected_features_BE)))
# Adding target column



selected_features_BE.append('DEFAULT')

df2=df[selected_features_BE]
X = df2.drop('DEFAULT',axis=1)

y = df2['DEFAULT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



smote = SMOTE(sampling_strategy='minority')

X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)



logreg = LogisticRegression(solver='liblinear', fit_intercept=True)



logreg.fit(X_train, y_train)



y_prob_train = logreg.predict_proba(X_train)[:,1]

y_pred_train = logreg.predict (X_train)



print('Classification report - Train: ', '\n', classification_report(y_train, y_pred_train))



y_prob = logreg.predict_proba(X_test)[:,1]

y_pred = logreg.predict (X_test)



print('Classification report - Test: ','\n', classification_report(y_test, y_pred))
# Defining an object for DTC and fitting for whole dataset

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1 )

dt.fit(X_train_sm, y_train_sm)



y_pred_train = dt.predict(X_train_sm)

y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)
#Classification for test before hyperparameter tuning

print(classification_report(y_test,y_pred))
dt = DecisionTreeClassifier(random_state=1)



params = {'criterion': ['gini','entropy'],

          'splitter' : ["best", "random"],

          'max_depth' : [2,4,6,8,10,12],

          'min_samples_split': [2,3,4,5],

          'min_samples_leaf': [1,2,3,4,5]}



rand_search_dt = RandomizedSearchCV(dt, param_distributions=params, cv=3)



rand_search_dt.fit(X_train_sm,y_train_sm)



rand_search_dt.best_params_
# Passing best parameter for the Hyperparameter Tuning

dt = DecisionTreeClassifier(**rand_search_dt.best_params_, random_state=1)



dt.fit(X_train_sm, y_train_sm)



y_pred = dt.predict(X_test)
#Classification for test after hyperparameter tuning

print(classification_report(y_test,y_pred))
#Create a Gaussian Classifier

rfc=RandomForestClassifier(n_estimators=100, random_state=1)



#Train the model using the training sets y_pred=clf.predict(X_test)

rfc.fit(X_train_sm,y_train_sm)



y_pred = rfc.predict(X_test)
#Classification for test after hyperparameter tuning

print(classification_report(y_test,y_pred))
rfc = RandomForestClassifier(random_state=1)



params = {'n_estimators': sp_randint(5,30),

          'criterion' : ['gini','entropy'],

          'max_depth' : sp_randint(2,10),

          'min_samples_split' : sp_randint(2,20),

          'min_samples_leaf' : sp_randint(1,20),

          'max_features' : sp_randint(2,18)}



rand_search_rfc = RandomizedSearchCV(rfc, param_distributions=params, random_state=1, cv=3)



rand_search_rfc.fit(X_train_sm,y_train_sm)



rand_search_rfc.best_params_
# Passing best parameter for the Hyperparameter Tuning

rfc = RandomForestClassifier(**rand_search_rfc.best_params_, random_state=1)



rfc.fit(X_train_sm, y_train_sm)



y_pred = rfc.predict(X_test)
#Classification for test after hyperparameter tuning

print(classification_report(y_test,y_pred))