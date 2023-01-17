import os

print(os.listdir('../input'))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
# index_col function help to remove the coloumn



df= pd.read_csv('../input/german credit1.csv',index_col=0)
# shape of the data

df.shape
df.info()
# removing the Unnamed column

#df.drop([' '],axis =1 )

sns.pairplot(df)
def outliers_iqr(ys):

    quartile_1, quartile_3 = np.percentile(ys, [25, 75])

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)

    upper_bound = quartile_3 + (iqr * 1.5)

    return np.where((ys > upper_bound) | (ys < lower_bound))
df['Credit amount'].hist()
df['Credit amount_log'] = np.log(df['Credit amount'])
df['Credit amount_log'].hist()
# summary statistics help to understand the distribution of data

# if the SD of any variable is 0 then we need to get rid of that 

# we will not get for categorical variable , only for numerical and continious numerical variable



df.describe()
cols=df.columns.tolist()
cols
cols = cols[-1:] + cols[:-1]
cols
df=df[cols]
df.head()
# Null data

df.isnull().sum()
# this will help us to know the fields under each header



print("Purpose : ",df.Purpose.unique())

print("Job : ",df.Job.unique())

print("Sex : ",df.Sex.unique())

print("Housing : ",df.Housing.unique())

print("Saving account : ",df['Saving account'].unique())

print("Checking account : ",df['Checking account'].unique())

print("Risk : ",df['Risk'].unique())
print("Saving accounts : ",df['Saving account'].value_counts())

print("Checking account : ",df['Checking account'].value_counts())
sns.countplot('Risk', data=df)
sns.countplot('Sex', data=df)
dimension = (15,5)

fig, ax = plt.subplots(figsize=dimension)

sns.countplot('Purpose', data=df)
sns.countplot('Saving account', data=df)
sns.countplot('Checking account', data=df)
dimension = (11, 6)

fig, ax = plt.subplots(figsize=dimension)

sns.countplot('Purpose', data=df)
sns.catplot(x='Purpose', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean Credit Amount by purpose and Risk')

plt.show()
sns.catplot(x='Duration', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean Duration by Credit Amount and Risk')

plt.show()
sns.catplot(x='Duration', y='Credit amount', hue='Sex', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean Duration by Credit Amount and Sex')

plt.show()
sns.catplot(x='Job', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean job by Credit Amount and Risk')

plt.show()
sns.catplot(x='Job', y='Credit amount', hue='Sex', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean job by Credit Amount and sex')

plt.show()
sns.catplot(x='Checking account', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean Checking account by Credit Amount and Risk')

plt.show()
sns.catplot(x='Saving account', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)

plt.title('Mean Saving accounts by Credit Amount and Risk')

plt.show()
dimension = (15, 6)

fig, ax = plt.subplots(figsize=dimension)

sns.countplot(x="Duration", data=df, 

              palette="hls",  hue = "Risk")
category = ["Checking account", 'Sex']

cm = sns.light_palette("pink", as_cmap=True)

pd.crosstab(df[category[0]],df[category[1]]).style.background_gradient(cmap = cm)
category = ["Saving account", 'Sex']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df[category[0]],df[category[1]]).style.background_gradient(cmap = cm)
category = ["Purpose", 'Sex']

cm = sns.light_palette("blue", as_cmap=True)

pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)
category = ["Sex", 'Risk']

cm = sns.light_palette("red", as_cmap=True)

pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)
category = ["Housing",'Sex']

cm = sns.light_palette("black", as_cmap=True)

pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)
category = ["Job",'Sex']

cm = sns.light_palette("violet", as_cmap=True)

pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)
sns.catplot(x='Sex', y='Age', hue='Risk', kind='bar', palette='Set1', data=df, height=3, aspect=3)

plt.title('Mean Sex by Age and Risk')

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.boxplot(x='Checking account', y='Credit amount', hue=None, data=df, palette='Set1')
fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.boxplot(x='Saving account', y='Credit amount', hue=None, data=df, palette='Set1')
def scatters(credit, h=None, pal=None):

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

    sns.scatterplot(x="Credit amount",y="Duration", hue=h, palette='Set1', data=df, ax=ax1)

    sns.scatterplot(x="Age",y="Credit amount", hue=h, palette='Set1', data=df, ax=ax2)

    sns.scatterplot(x="Age",y="Duration", hue=h, palette='Set1', data=df, ax=ax3)

    plt.tight_layout()
scatters(df, h="Saving account")
scatters(df, h="Checking account")
scatters(df, h="Risk")
scatters(df, h="Sex")
# this will help to replace all the NAN values with little values in both saving and checking account



#df["Saving accounts"]=df["Saving accounts"].fillna(method="bfill")

#df["Checking account"]=df["Checking account"].fillna(method="bfill")



df.fillna('little',inplace=True)
print("Saving account : ",df['Saving account'].value_counts())

print("Checking account : ",df['Checking account'].value_counts())
df.info()
features = df.iloc[:,:10]

label = df.iloc[:,[-1]]
features.head()
label.head()
from sklearn.preprocessing import LabelEncoder
SexinNumeric=LabelEncoder()

HousinginNumeric=LabelEncoder()

SavingaccountinNumeric=LabelEncoder()

CheckingaccountinNumeric=LabelEncoder()

PurposeinNumeric=LabelEncoder()

RiskinNumeric=LabelEncoder()
features['SexinNumeric']=SexinNumeric.fit_transform(features['Sex'])

features['HousinginNumeric']=HousinginNumeric.fit_transform(features['Housing'])

features['SavingaccountinNumeric']=SavingaccountinNumeric.fit_transform(features['Saving account'])

features['CheckingaccountinNumeric']=CheckingaccountinNumeric.fit_transform(features['Checking account'])

features['PurposeinNumeric']=PurposeinNumeric.fit_transform(features['Purpose'])

label['RiskinNumeric']=RiskinNumeric.fit_transform(label['Risk'])
features.tail()
label.tail()
NewFeatures = features.drop(['Sex','Housing','Saving account','Checking account', 'Purpose'], axis='columns')

NewLabel = label.drop(['Risk'], axis='columns')
NewFeatures.head()
NewLabel.head()
NewFeatures = features.drop(['Sex','Housing','Saving account', 'Checking account', 'Purpose'], axis='columns').values

NewLabel = label.drop(['Risk'], axis='columns').values
#Create Train Test Split



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(NewFeatures,

                                                NewLabel,

                                                test_size=0.20,

                                                random_state=44)
# Verifying



print(f'X_train dimension: {X_train.shape}')

print(f'X_test dimension: {X_test.shape}')

print(f'\ny_train dimension: {y_train.shape}')

print(f'y_test dimension: {y_test.shape}')
#print("Saving account : ",X_train['SavingaccountinNumeric'].value_counts())

#print("Checking account : ",X_train['CheckingaccountinNumeric'].value_counts())
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train,y_train.ravel())
model.score(X_train,y_train)
model.score(X_test,y_test)
ypred=model.predict(X_test)
print(ypred)
from sklearn.metrics import accuracy_score
print(accuracy_score(ypred, y_test))
from sklearn.metrics import confusion_matrix

CM = confusion_matrix(ypred,y_test)

CM
sns.heatmap(CM, annot=True,fmt='d')

plt.xlabel('Predicted')

plt.ylabel('actual')
from sklearn.metrics import classification_report

print(classification_report(ypred,y_test))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
y_pred_prob = model.predict_proba(X_test)[:,1]
Log_roc = roc_auc_score(y_test,y_pred_prob)

fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob)
plt.figure()

plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc)

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive rate(100-specificity)')

plt.ylabel('True Positive rate(sensitivity)')

plt.legend(loc='lower right')

plt.show()
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=25) #k = 5

model2.fit(X_train,y_train.ravel())
model2.score(X_train,y_train)
model2.score(X_test,y_test)
ypred2=model2.predict(X_test)
print(ypred2)
CM2 = confusion_matrix(ypred,y_test)

CM2
sns.heatmap(CM2, annot=True,fmt='d')

plt.xlabel('Predicted')

plt.ylabel('actual')
print(classification_report(ypred2,y_test))
y_pred_prob2 = model2.predict_proba(X_test)[:,1]
Log_roc2 = roc_auc_score(y_test,y_pred_prob2)

fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob2)
plt.figure()

plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc2)

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive rate(100-specificity)')

plt.ylabel('True Positive rate(sensitivity)')

plt.legend(loc='lower right')

plt.show()
from sklearn import tree
model3 = tree.DecisionTreeClassifier()
model3=model3.fit(X_train,y_train)
model3.score(X_test,y_test)
ypred3=model3.predict(X_test)
print(ypred3)
from sklearn.metrics import confusion_matrix

CM3 = confusion_matrix(ypred3,y_test)

CM3
sns.heatmap(CM3, annot=True,fmt='d')

plt.xlabel('Predicted')

plt.ylabel('actual')
print(classification_report(ypred3,y_test))
y_pred_prob3 = model3.predict_proba(X_test)[:,1]
Log_roc3 = roc_auc_score(y_test,y_pred_prob3)

fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob3)
plt.figure()

plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc3)

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive rate(100-specificity)')

plt.ylabel('True Positive rate(sensitivity)')

plt.legend(loc='lower right')

plt.show()
from sklearn.ensemble import RandomForestClassifier

model4 = RandomForestClassifier(n_estimators=29)

model4.fit(X_train, y_train.ravel())
model4.score(X_train,y_train)
model4.score(X_test,y_test)
ypred4 = model4.predict(X_test)
print(ypred4)
from sklearn.metrics import confusion_matrix

CM4 = confusion_matrix(ypred4,y_test)

CM4
sns.heatmap(CM4, annot=True,fmt='d')

plt.xlabel('Predicted')

plt.ylabel('actual')
print(classification_report(ypred4,y_test))
y_pred_prob4 = model4.predict_proba(X_test)[:,1]
Log_roc4 = roc_auc_score(y_test,y_pred_prob4)

fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob4)
plt.figure()

plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc4)

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive rate(100-specificity)')

plt.ylabel('True Positive rate(sensitivity)')

plt.legend(loc='lower right')

plt.show()
from sklearn.svm import SVC
model5 = SVC(kernel='linear', probability=False)
model5.fit(X_train,y_train.ravel())
model5.score(X_train,y_train)
model5.score(X_test,y_test)
ypred5 = model5.predict(X_test)
print(ypred5)
print(classification_report(ypred5,y_test))
CM5 = confusion_matrix(ypred5,y_test)

CM5
sns.heatmap(CM5, annot=True,fmt='d')

plt.xlabel('Predicted')

plt.ylabel('actual')
# y_pred_prob5 = model5.predict_proba(X_test)[:,1]
Log_roc5 = roc_auc_score(y_test,y_pred_prob5)

fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob5)
plt.figure()

plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc5)

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive rate(100-specificity)')

plt.ylabel('True Positive rate(sensitivity)')

plt.legend(loc='lower right')

plt.show()
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model
print(cross_val_score(model,X_test,y_test,cv=10,scoring='accuracy').mean())
print(cross_val_score(model2,X_test,y_test,cv=10,scoring='accuracy').mean())
print(cross_val_score(model3,X_test,y_test,cv=10,scoring='accuracy').mean())
print(cross_val_score(model4,X_test,y_test,cv=10,scoring='accuracy').mean())
print(cross_val_score(model5,X_test,y_test,cv=10,scoring='accuracy').mean())