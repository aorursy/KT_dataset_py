#Libraries used in the project

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px

import warnings

warnings.filterwarnings("ignore")
#Reading the input dataset 

data = pd.read_csv('../input/loandata/Loan payments data.csv')



# pd.options.display.max_columns = None

# pd.options.display.max_rows = None



data.head()
df = data.copy()
df.info()
df.describe()
#Changed the name of a value in 'education' column from 'Bechalor' to 'Bachelor'

df['education']= df['education'].replace('Bechalor','Bachelor')
#Find the number of missing values in each columns

df.isna().sum()
#Temporarily Filling the empty values in 'past_due_days' as '0'

df['past_due_days'] = df['past_due_days'].fillna(0)
#Filling the empty values in 'paid_off_time' as '-1'

df['paid_off_time'] = df['paid_off_time'].fillna(-1)
#Find the number of missing values in each columns

df.isna().sum()
#Number of unique values in each column

for cat in data.columns:

    print("Number of levels in category '{0}': \b  {1:2.0f} ".format(cat, df[cat].unique().size))
#Coverting the following columns to 'datetime'

df['effective_date'] = pd.to_datetime(df['effective_date'])

df['due_date'] = pd.to_datetime(df['due_date'])

df['paid_off_time'] = pd.to_datetime(df['paid_off_time']).dt.date
#To Convert the 'paid_off_time' column to datetime64 type

df['paid_off_time'] = pd.to_datetime(df['paid_off_time'])

df.head()
df.info()
df_fe = df.copy()
df_fe.head()
for i in range(len(df_fe[df_fe['loan_status']=="PAIDOFF"])):

    df_fe['past_due_days'][i] = (df_fe['paid_off_time'][i] - df_fe['effective_date'][i] + pd.Timedelta(days=1)).days - df_fe['terms'][i]

df_fe.head(10)
#Records where the difference in the paid_off_time and effective_date is greater than the terms

df_fe[(df_fe['past_due_days']>0)&(df_fe['loan_status']=='PAIDOFF')]
a = df_fe['loan_status'].value_counts()

pd.DataFrame(a)
plt.pie(df_fe['loan_status'].value_counts(),labels=df_fe['loan_status'].unique(),explode=[0,0.1,0],startangle=144,autopct='%1.f%%')

plt.title('Loan Status Distribution',fontsize = 20)

plt.show()
b= df_fe['Gender'].value_counts()

pd.DataFrame(b)
c = df_fe.groupby(['Gender'])['loan_status'].value_counts()

pd.DataFrame(c)
plt.figure(figsize = [10,5])

sns.countplot(df_fe['Gender'],hue=df_fe['loan_status'])

plt.legend(loc='upper right')

plt.title('Gender vs Loan Status',fontsize=20)

plt.xlabel('Gender', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
d = df_fe['education'].value_counts()

pd.DataFrame(d)
plt.figure(figsize = [10,5])

sns.countplot(df_fe['education'],hue=df_fe['loan_status'])

plt.legend(loc='upper right')

plt.title('Education vs Loan Status',fontsize=20)

plt.xlabel('Education', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
for i in df_fe['loan_status'].unique():

    agemean=df_fe[df_fe['loan_status']==i]['age'].mean()

    agemode=df_fe[df_fe['loan_status']==i]['age'].mode()

    print("average age of people whose loan status is'{0}': \b {1:2.2f} and mode is {2}".format(i,agemean, agemode[0]))
plt.figure(figsize = [14,5])

sns.countplot(df_fe['age'],hue=df_fe['loan_status'])

plt.legend(loc='upper left')

plt.title('Age vs Loan Status',fontsize=20)

plt.xlabel('Age', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
e = df_fe['Principal'].value_counts()

pd.DataFrame(e)
plt.figure(figsize = [10,5])

sns.countplot(df_fe['Principal'],hue=df_fe['loan_status'])

plt.legend(loc='upper left')

plt.title('Principal vs Loan Status',fontsize=20)

plt.xlabel('Principal', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
plt.figure(figsize = [10,5])

sns.countplot(df_fe['terms'],hue=df_fe['loan_status'])

plt.legend(loc='upper left')

plt.title('Terms vs Loan Status',fontsize=20)

plt.xlabel('Terms', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
g = df_fe.groupby(['effective_date'])['loan_status'].value_counts()

pd.DataFrame(g)
plt.figure(figsize = [10,5])

dates = df_fe['effective_date'].dt.date

sns.countplot(x=dates, hue=df_fe['loan_status'])

plt.legend(loc='upper right')

plt.title('Effective Date vs Loan Status',fontsize=20)

plt.xlabel('Effective Date', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
# specifies the parameters of our graphs

fig = plt.figure(figsize=(18,8), dpi=1600)

alpha_bar_chart = 0.55



# lets us plot many diffrent shaped graphs together 

ax1 = plt.subplot2grid((2,3),(0,0))

sns.countplot(df_fe['loan_status'],hue=df_fe['education'])

plt.legend(loc='upper right')

plt.title('Loan Status vs Education',fontsize=15)

plt.xlabel(None)

plt.ylabel('Count',fontsize=14)



plt.subplot2grid((2,3),(0,1),rowspan=2)

plt.pie(df_fe['loan_status'].value_counts(),labels=df_fe['loan_status'].unique(),explode=[0,0.1,0],startangle=165,autopct='%1.f%%',)

plt.grid(b=True, which='major', axis='y')

plt.title("Loan Status Distribution",fontsize=20)



ax3 = plt.subplot2grid((2,3),(0,2))

sns.countplot(df_fe['loan_status'],hue=df_fe['terms'])

plt.legend(loc='upper right')

plt.title('Loan Status vs Terms',fontsize=15)

plt.xlabel(None)

plt.ylabel('Count',fontsize=14)



ax4 = plt.subplot2grid((2,3),(1,0))

sns.countplot(df_fe['loan_status'],hue=df_fe['Principal'])

plt.legend(loc='upper right')

plt.title('Loan Status vs Principal',fontsize=15)

plt.xlabel('Loan Status',fontsize=14)

plt.ylabel('Count',fontsize=14)



ax5 = plt.subplot2grid((2,3),(1,2))

sns.countplot(df_fe['loan_status'],hue=df_fe['Gender'])

plt.legend(loc='upper right')

plt.title('Loan Status vs Gender',fontsize=15)

plt.xlabel('Loan Status',fontsize=14)

plt.ylabel('Count',fontsize=14)



plt.show()
px.scatter(df_fe, x="age", y="past_due_days", size ="terms" ,color="loan_status",

           hover_data=['Gender','Principal'], log_x=True, size_max=8)
# Relation between loan_status and past_due_days

%matplotlib inline

plt.figure(figsize = [9,5])

sns.boxplot(x='loan_status', y='past_due_days', data=df_fe)

plt.xlabel('Loan Status', fontsize=16)

plt.ylabel('Past Due Days', fontsize=16)

plt.show()
df_fe = df_fe.drop(['Loan_ID','effective_date','due_date','paid_off_time'],axis = 1)

df_fe.head()
df_fe.info()
p = df_fe.groupby(['loan_status'])['Principal'].value_counts()

pd.DataFrame(p)
df_fe_Pri = df_fe.copy()
df_fe_Pri.head()
df_fe_Pri[df_fe_Pri['terms']==7]
df_fe_Pri[(df_fe_Pri['Principal']!=800) &(df_fe_Pri['Principal']!=1000)]
#Dropping rows where 'Principal' is not equal to 800 and 1000 [12 rows]

df_fe_Pri = df_fe_Pri[(df_fe_Pri['Principal']==800) | (df_fe_Pri['Principal']==1000)]
#Dropping rows where 'terms' = 7 days [21 rows]

df_fe_Pri = df_fe_Pri[df_fe_Pri['terms']!=7]
df_fe_Pri.head()
df_fe_Pri.shape
df_clean = df_fe_Pri.copy()
def age_classification(age):

    if age.item()<21:

        return 'Young'

    elif age.item()>=21 and age.item()<31:

        return 'MidAge'

    elif age.item()>=31 and age.item()<41:

        return 'Senior'

    else:

        return 'Older'
#Categorizing age column

df_clean['age'] = df_clean[['age']].apply(age_classification,axis=1)
df_clean.info()
df_clean['terms'] = df_clean['terms'].astype('object')

df_clean['Principal'] = df_clean['Principal'].astype('object')
#Select the variables to be one-hot encoded

one_hot_features = ['education','Gender','Principal','age','terms']

# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).

one_hot_encoded = pd.get_dummies(df_clean[one_hot_features],drop_first=True)

one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

# Convert Categorical to Numerical for default column
one_hot_encoded.head()
df_encoded = pd.concat([df_clean,one_hot_encoded],axis=1)

df_encoded.head()
df_encoded.drop(['terms','education','Gender','age','Principal'],axis=1,inplace = True)

df_encoded.head()
df_clean['loan_status'].unique()
loan_status_dict = {'PAIDOFF':1,'COLLECTION':2,'COLLECTION_PAIDOFF':3}

df_encoded['loan_status'] = df_encoded.loan_status.map(loan_status_dict)

df_encoded.head()
df_model = df_encoded.copy()
df_model.info()
df_model.head()
correlation = df_model[df_model.columns].corr()

plt.figure(figsize=(12, 10))

plot = sns.heatmap(correlation, vmin = -1, vmax = 1,annot=True, annot_kws={"size": 10})

plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split   #splitting data



#Standardize rows into uniform scale



X = df_model.drop(['loan_status','past_due_days'],axis=1)

y = df_model['loan_status']



# scaler = MinMaxScaler()#StandardScaler,MinMaxScaler

# scaler.fit(X_Act)#df_model[cols_to_norm]



# # Scale and center the data

# fdf_normalized = scaler.fit_transform(X_Act)



# # # Create a pandas DataFrame

# fdf_normalized_df = pd.DataFrame(data=fdf_normalized, index=X_Act.index, columns=X_Act.columns)



# X = fdf_normalized_df



##Note: In this case, Scaling is not required



X.head()
#Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=400,test_size=0.30,stratify = y)
from collections import Counter

print("y : ",Counter(y))

print("y_train : ",Counter(y_train))

print("y_test : ",Counter(y_test))
# Actual Values(of Majority Class) of y_test

y_test.value_counts()

y_test.value_counts().head(1) / len(y_test)
# metrics

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, recall_score
def model_train(model, name):

    model.fit(X_train, y_train)                                          # Fitting the model

    y_pred = model.predict(X_test)                                       # Making prediction from the trained model

    cm = confusion_matrix(y_test, y_pred)                               

    print("Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix

    print(cm)

    print('-----------------------')

    print('-----------------------')

    cr = classification_report(y_test, y_pred)

    print(name +" Classification Report " +" Validation Data")           # Displaying the Classification Report

    print(cr)

    print('------------------------')

    print(name + " Bias")                                                 # Calculating bias

    bias = y_pred - y_test.mean()

    print("Bias "+ str(bias.mean()))

    

    print(name + " Variance")                                             # Calculate Variance

    var = np.var([y_test, y_pred], axis=0)

    print("Variance " + str(var.mean()) )

#     return auc, rec, model

    return model

# Building the Logistic Regression Model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1000,max_iter=500,class_weight='balanced')    # Set Large C value for low regularization to prevent overfitting

# logreg.fit(X_train, y_train)



dt_model = model_train(logreg, "Logistic Regression")

print('_________________________')

print("Coefficients: ",logreg.coef_)                                            # Coefficients for Logistic Regression

print("Intercepts: ",logreg.intercept_)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'gini',max_depth = 4, min_samples_leaf =2,random_state=101,class_weight='balanced')



dt_model = model_train(dt, "Decision Tree")
# pip install imbalanced-learn
#Let us try some sampling technique to remove class imbalance

from imblearn.over_sampling import SMOTE,KMeansSMOTE,SVMSMOTE

#Over-sampling: SMOTE

#SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, 

#based on those that already exist. It works randomly picking a point from the minority class and computing 

#the k-nearest neighbors for this point.The synthetic points are added between the chosen point and its neighbors.

smote = KMeansSMOTE(sampling_strategy='auto')



X_sm, y_sm = smote.fit_sample(X, y)

print(X_sm.shape, y_sm.shape)
#Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X_sm,y_sm,random_state=400,test_size=0.30,stratify = y_sm)#,stratify = y
from collections import Counter

print("y : ",Counter(y))

print("y_train : ",Counter(y_train))

print("y_test : ",Counter(y_test))
# Actual Values(of Majority Class) of y_test

y_test.value_counts()

y_test.value_counts().head(1) / len(y_test)
# Building the Logistic Regression Model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1000,max_iter=500,class_weight='balanced')#solver ='lbfgs',class_weight='balanced'    # Set Large C value for low regularization to prevent overfitting

# logreg.fit(X_train, y_train)



dt_model = model_train(logreg, "Logistic Regression")

print('_________________________')

print("Coefficients: ",logreg.coef_)                                            # Coefficients for Logistic Regression

print("Intercepts: ",logreg.intercept_)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'gini',max_depth = 4, min_samples_leaf =2,random_state=101)



dt_model = model_train(dt, "Decision Tree")
from sklearn.model_selection import GridSearchCV



random_grid = {'n_estimators': range(5,20),

              'max_features' : ['auto', 'sqrt'],

              'max_depth' : [10,20,30,40],

              'min_samples_split':[2,5,10],

              'min_samples_leaf':[1,2,4]}



rf = RandomForestClassifier()



rf_gs = GridSearchCV(rf, random_grid, cv = 3, n_jobs=1, verbose=2)



rf_gs.fit(X_train, y_train)

y_pred = rf_gs.predict(X_test)

print(rf_gs.best_estimator_)

print('-----------------------')

print("Grid Search Validation Data")

cm = confusion_matrix(y_test, y_pred)                               

print("Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix

print(cm)

print('-----------------------')

cr = classification_report(y_test, y_pred)

print("Grid Search Classification Report " +" Validation Data")           # Displaying the Classification Report

print(cr)

print('------------------------')

print("Grid Search Bias")                                                 # Calculating bias

bias = y_pred - y_test.mean()

print("Bias "+ str(bias.mean()))

    

print("Grid Search Variance")                                             # Calculate Variance

var = np.var([y_test, y_pred], axis=0)

print("Variance " + str(var.mean()) )
# Import Eli5 package

import eli5

from eli5.sklearn import PermutationImportance



# Find the importance of columns for prediction

perm = PermutationImportance(dt, random_state=10).fit(X_test,dt.predict(X_test))

eli5.show_weights(perm, feature_names = X.columns.tolist())