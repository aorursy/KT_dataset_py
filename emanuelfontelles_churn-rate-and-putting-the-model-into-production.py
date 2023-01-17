import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
sns.set_style()
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=',')
data.head()
print('Rows     :', data.shape[0])
print('Columns  :', data.shape[1])
print('\nFeatures :', data.columns.tolist())
print('\nUnique values :\n', data.nunique())
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Valores em falta', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
# Valores que estão faltando nos dados
missing_values = missing_values_table(data)
data.info()
#Data Manipulation

#Replacing spaces with null values in total charges column
data['TotalCharges'] = data["TotalCharges"].replace(" ", np.nan)

#Dropping null values from total charges column which contain .15% missing data 
data = data[data["TotalCharges"].notnull()]
data = data.reset_index()[data.columns]

#convert to float type
data["TotalCharges"] = data["TotalCharges"].astype(float)
data.info()
f, axes = plt.subplots(2, 3, sharey=False, sharex=False, figsize=(10,4))

data['OnlineSecurity'].value_counts(ascending=True).plot.barh(title='OnlineSecurity', ax=axes[0,0])
data['OnlineBackup'].value_counts(ascending=True).plot.barh(title='OnlineBackup', ax=axes[0,1])
data['DeviceProtection'].value_counts(ascending=True).plot.barh(title='DeviceProtection', ax=axes[0,2])
data['TechSupport'].value_counts(ascending=True).plot.barh(title='TechSupport', ax=axes[1,0])
data['StreamingTV'].value_counts(ascending=True).plot.barh(title='StreamingTV', ax=axes[1,1])
data['StreamingMovies'].value_counts(ascending=True).plot.barh(title='StreamingMovies', ax=axes[1,2])
plt.tight_layout()
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'MultipleLines']

for i in replace_cols :
    data[i]  = data[i].replace({'No internet service' : 'No'})
    data[i]  = data[i].replace({'No phone service' : 'No'})
f, axes = plt.subplots(2, 3, sharey=False, sharex=False, figsize=(10,4))

data['OnlineSecurity'].value_counts(ascending=True).plot.barh(title='OnlineSecurity', ax=axes[0,0])
data['OnlineBackup'].value_counts(ascending=True).plot.barh(title='OnlineBackup', ax=axes[0,1])
data['DeviceProtection'].value_counts(ascending=True).plot.barh(title='DeviceProtection', ax=axes[0,2])
data['TechSupport'].value_counts(ascending=True).plot.barh(title='TechSupport', ax=axes[1,0])
data['StreamingTV'].value_counts(ascending=True).plot.barh(title='StreamingTV', ax=axes[1,1])
data['StreamingMovies'].value_counts(ascending=True).plot.barh(title='StreamingMovies', ax=axes[1,2])
plt.tight_layout()
#replace values
data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})
f, axes = plt.subplots(3, 2, sharey=False, sharex=False, figsize=(12,12))

data['Contract'].value_counts(ascending=True, normalize=True).plot.barh(title='Contracts', ax=axes[0,0])
data['SeniorCitizen'].value_counts(ascending=True, normalize=True).plot.barh(title='SeniorCitizen', ax=axes[0,1])
sns.countplot(x='gender', hue='Contract', data=data, orient='v', ax=axes[1,0])
sns.countplot(x='Contract', hue='SeniorCitizen', data=data, orient='v', ax=axes[1,1])

data.groupby('gender')['SeniorCitizen'].value_counts(ascending=True, normalize=True).plot.barh(title='SeniorCitizen by gender', ax=axes[2,0])
data.groupby('Contract')['gender'].value_counts(ascending=True, normalize=False).unstack().plot.bar(title='Contracts by gender', ax=axes[2,1])
plt.tight_layout()
data['Contract'].value_counts(ascending=True, normalize=True)
#Tenure to categorical column
def tenure_lab(data):    
    if data['tenure'] <= 12 :
        return "Tenure_0-12"
    elif (data['tenure'] > 12) & (data['tenure'] <= 24 ):
        return "Tenure_12-24"
    elif (data['tenure'] > 24) & (data['tenure'] <= 48) :
        return "Tenure_24-48"
    elif (data['tenure'] > 48) & (data['tenure'] <= 60) :
        return "Tenure_48-60"
    elif data['tenure'] > 60 :
        return "Tenure_60-gt"
data["tenure_group"] = data.apply(lambda data:tenure_lab(data), axis = 1)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(10,4))

data['tenure'].hist(density=False, ax=ax1)
data['tenure_group'].value_counts(ascending=False, sort=True, normalize=True).sort_index().plot.bar(ax=ax2)

ax1.set_xlabel(r'Tenure')
ax1.set_ylabel(r'# months with the company')
plt.tight_layout()
order_tenure = list(data['tenure_group'].value_counts(ascending=False, sort=True, normalize=True).sort_index().index)
data[data['Churn']=='Yes']['TotalCharges'].hist();
data[data['Churn']=='No']['TotalCharges'].hist();
f, axes = plt.subplots(3, 2, sharey=False, sharex=False, figsize=(16,16))

sns.countplot(x='tenure_group', hue='Churn', data=data, order=order_tenure, orient='v', ax=axes[0,0])
sns.countplot(x='Churn', hue='gender', data=data, orient='v', ax=axes[0,1])
sns.countplot(x='Churn', hue='SeniorCitizen', data=data, orient='v', ax=axes[1,0])
sns.distplot(data[data['Churn']=='Yes']['MonthlyCharges'], kde=False, hist=True, norm_hist=False, ax=axes[1,1], label='Churn')
sns.distplot(data[data['Churn']=='No']['MonthlyCharges'], kde=False, hist=True, norm_hist=False, ax=axes[1,1], label='Non-Churn')

sns.distplot(data[data['Churn']=='Yes']['TotalCharges'], kde=False, hist=True, norm_hist=False, ax=axes[2,0], label='Churn')
sns.distplot(data[data['Churn']=='No']['TotalCharges'], kde=False, hist=True, norm_hist=False, ax=axes[2,0], label='Non-Churn')

pivot = pd.pivot_table(data, values=['MonthlyCharges', 'TotalCharges'], index=['tenure_group', 'Churn'])

pivot.unstack()['TotalCharges'].plot.bar(ax=axes[2,1], title='Average Total Charges by Tenure groups')


axes[0,0].set_title('Churn customers by Tenure groups')
axes[0,1].set_title('Churn customers by gender')
axes[1,0].set_title('Churn customers by SeniorCitizen')
axes[1,1].set_title('Distribution of MonthlyCharges by Churn')
axes[2,0].set_title('Distribution of TotalCharges by Churn')

axes[1,1].legend()
axes[2,0].legend()
axes[2,1].legend()
plt.tight_layout()
from sklearn.preprocessing import LabelEncoder

dataobject=data.select_dtypes(['object'])

def uni(columnname):
    print(columnname,"--" ,data[columnname].unique())

for i in range(1,len(dataobject.columns)):
    uni(dataobject.columns[i])
    
def labelencode(columnname):
    data[columnname] = LabelEncoder().fit_transform(data[columnname])
    
for i in range(1,len(dataobject.columns)):
    labelencode(dataobject.columns[i])
        
for i in range(1,len(dataobject.columns)):
     uni(dataobject.columns[i])
df = data.copy()
drop_list = ['customerID', 'gender', 'Dependents', 'PhoneService', 'DeviceProtection', 'TechSupport',
             'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']

df['Contract_0'] = ((df['Contract']==0).values).astype(int)
df['Contract_1'] = ((df['Contract']==1).values).astype(int)
df['Contract_2'] = ((df['Contract']==2).values).astype(int)

df['tenure_0'] = ((df['tenure_group']==0).values).astype(int)
df['tenure_1'] = ((df['tenure_group']==1).values).astype(int)
df['tenure_2'] = ((df['tenure_group']==2).values).astype(int)
df['tenure_3'] = ((df['tenure_group']==3).values).astype(int)
df['tenure_4'] = ((df['tenure_group']==4).values).astype(int)

df = df.drop(drop_list, axis=1)
df = df.drop(['Contract', 'tenure', 'tenure_group'], axis=1)
f, (ax1,ax2) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(16,6))

corr = df.corr(method='pearson')

sns.heatmap(data.corr(method='pearson'), cmap=plt.cm.inferno_r, ax=ax1)
sns.heatmap(corr, cmap=plt.cm.inferno_r, ax=ax2)
plt.tight_layout()
from sklearn.model_selection import train_test_split
data_to_train = data.drop(['customerID'], axis=1).copy()

train_set, validation_set = train_test_split(data_to_train.copy(), test_size=0.30)

#to perform cross-validation
target_to_train = data_to_train['Churn']
data_to_train = data_to_train.drop(['Churn'], axis=1)

#to perform holdout-set
train_target = train_set['Churn']
train_set = train_set.drop(['Churn'], axis=1)

validation_target = validation_set['Churn']
validation_set = validation_set.drop(['Churn'], axis=1)
x, y = train_set, train_target
from sklearn.linear_model import Lasso, LassoCV

lassocv = LassoCV(alphas=None, cv=40, max_iter=100000, normalize=True)
lassocv.fit(data_to_train, target_to_train);
from sklearn.ensemble import RandomForestClassifier

# Fit RandomForest Classifier
clf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
clf.fit(x, y);
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x, y);
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=0.01) 
svm.fit(x, y);
f, axes = plt.subplots(2, 2, sharey=False, sharex=False, figsize=(16,10))

imp1 = pd.Series(data=clf.feature_importances_, index=x.columns).sort_values(ascending=False)
imp2 = pd.Series(data=lassocv.coef_, index=x.columns).sort_values(ascending=False)
imp3 = pd.Series(data=logreg.coef_[0], index=x.columns).sort_values(ascending=False)
imp4 = pd.Series(data=svm.coef_[0], index=x.columns).sort_values(ascending=False)

sns.barplot(y=imp1.index, x=imp1.values, orient='h', ax=axes[0,0])
sns.barplot(y=imp2.index, x=imp2.values, orient='h', ax=axes[0,1])
sns.barplot(y=imp3.index, x=imp3.values, orient='h', ax=axes[1,0])
sns.barplot(y=imp4.index, x=imp4.values, orient='h', ax=axes[1,1])



axes[0,0].set_title("Feature importance using Random Forest")
axes[0,1].set_title("Feature importance using LASSO")
axes[1,0].set_title("Feature importance using Logistic Regression")
axes[1,1].set_title("Feature importance using Support Vector Machine")

plt.tight_layout();
from sklearn.model_selection import cross_val_score
cross_validation_pred_random_forest = cross_val_score(clf, data_to_train, target_to_train, cv=5)
cross_validation_pred_logreg = cross_val_score(logreg, data_to_train, target_to_train, cv=5)
cross_validation_pred_svm = cross_val_score(svm, data_to_train, target_to_train, cv=5)
validation_pred_random_forest = clf.predict(validation_set)
validation_pred_logreg = logreg.predict(validation_set)
validation_pred_svm = svm.predict(validation_set)
from sklearn.metrics import accuracy_score
acc_random_forest = accuracy_score(validation_target, validation_pred_random_forest)
acc_logreg = accuracy_score(validation_target, validation_pred_logreg)
acc_svm = accuracy_score(validation_target, validation_pred_svm)

print('Random Forest Accuracy:          ', acc_random_forest*100)
print('Logistic Regression Accuracy:    ', acc_logreg*100)
print('SVM Accuracy:                    ', acc_svm*100)

print('\nRandom Forest CV Accuracy:       ', cross_validation_pred_random_forest.mean()*100)
print('Logistic Regression CV Accuracy: ', cross_validation_pred_logreg.mean()*100)
print('SVM CV Accuracy:                 ', cross_validation_pred_svm.mean()*100)
from sklearn.metrics import confusion_matrix

f, axes = plt.subplots(1, 3, sharey=False, sharex=False, figsize=(16,5))

axes[0].set_title("Confusion Matrix to Random Forest")
axes[1].set_title("Confusion Matrix to Logistic Regression")
axes[2].set_title("Confusion Matrix to SVM")

confusion_matrix1 = confusion_matrix(validation_target, validation_pred_random_forest)
confusion_matrix2 = confusion_matrix(validation_target, validation_pred_logreg)
confusion_matrix3 = confusion_matrix(validation_target, validation_pred_svm)

sns.heatmap(confusion_matrix1, ax=axes[0], annot=True)
sns.heatmap(confusion_matrix2, ax=axes[1], annot=True)
sns.heatmap(confusion_matrix3, ax=axes[2], annot=True)

plt.tight_layout();
df.head()
train_set, validation_set = train_test_split(df.copy(), test_size=0.30)

#to perform holdout-set
train_target = train_set['Churn']
train_set = train_set.drop(['Churn'], axis=1)

validation_target = validation_set['Churn']
validation_set = validation_set.drop(['Churn'], axis=1)

x, y = train_set, train_target
clf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
clf.fit(x, y);

logreg = LogisticRegression()
logreg.fit(x, y);

svm = SVC(kernel='linear') 
svm.fit(x, y);

validation_pred_random_forest = clf.predict(validation_set)
validation_pred_logreg = logreg.predict(validation_set)
validation_pred_svm = svm.predict(validation_set)

acc_random_forest = accuracy_score(validation_target, validation_pred_random_forest)
acc_logreg = accuracy_score(validation_target, validation_pred_logreg)
acc_svm = accuracy_score(validation_target, validation_pred_svm)
print('Random Forest Accuracy:          ', acc_random_forest*100)
print('Logistic Regression Accuracy:    ', acc_logreg*100)
print('SVM Accuracy:                    ', acc_svm*100)