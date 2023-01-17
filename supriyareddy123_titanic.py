# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





# Importing libraries

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/train.csv')

df_test =  pd.read_csv('../input/test.csv')
print(df_train.columns)

print(df_test.columns)
df_train.sample(15)
df_train.dtypes
df_test.dtypes
df_train.shape
df_test.shape
df_train.isnull().any()
df_test.isnull().any()
df_train.describe()
# Boxplot for analyzing the outliers for fare

plt.boxplot(df_train['Fare'])
df_test.describe()
# Parch and Sex Vs Survived



g = sns.FacetGrid(df_train, col="Parch",  row="Sex", hue = 'Survived')

g = g.map(plt.hist, "Survived")
# Passenger class and Sex Vs Survived



g = sns.FacetGrid(df_train, col="Pclass",  row="Sex", hue = 'Survived')

g = g.map(plt.hist, "Survived")
# Correlation matrix - linear relation among independent attributes and with the Target attribute



sns.set(style="white")



# Compute the correlation matrix

correln = df_train.corr()



# Generate a mask for the upper triangle

#mask = np.zeros_like(correln, dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(correln,  cmap=cmap, vmax=.3, #mask=mask,

            linewidths=.5, cbar_kws={"shrink": .7})
# Correlation matrix - linear relation among independent attributes 



sns.set(style="white")



# Compute the correlation matrix

correln = df_test.corr()



# Generate a mask for the upper triangle

#mask = np.zeros_like(correln, dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(correln,  cmap=cmap, vmax=.3, #mask=mask,

            linewidths=.5, cbar_kws={"shrink": .7})
df_train['Embarked'].value_counts()
df_test['Embarked'].value_counts()
df_train['Embarked']=df_train['Embarked'].replace(np.nan,"S")

df_test['Embarked']=df_test['Embarked'].replace(np.nan,"S")
df_test[df_test['Fare'].isna()]['Pclass']
df_test['Fare'] = np.where(df_test['Fare'].isna(), np.nanmedian(df_test[df_test['Pclass']==3]['Fare']), df_test['Fare'])
df_train['Cabin']=df_train['Cabin'].replace(np.nan,"Unknown")

df_test['Cabin']=df_test['Cabin'].replace(np.nan,"Unknown")
df_train['Sex'] = df_train['Sex'].map({'male':0, 'female':1}).astype("category")  

df_train['Pclass'] = df_train['Pclass'].astype("category")

df_train['Embarked'] = df_train['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype("category")



df_test['Sex'] = df_test['Sex'].map({'male':0, 'female':1}).astype("category")  

df_test['Pclass'] = df_test['Pclass'].astype("category")

df_test['Embarked'] = df_test['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype("category")
df_train['Cab_alpha'] = df_train['Cabin'].str[0]

df_test['Cab_alpha'] = df_test['Cabin'].str[0]
# Cabin encoding and type conversion

df_train['Cab_alpha_en'] = df_train['Cab_alpha'].map({'U':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}).astype('category')

df_test['Cab_alpha_en'] = df_test['Cab_alpha'].map({'U':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}).astype('category')
df_train['title'] = df_train['Name'].str.split(',').apply(lambda x: x[1])

df_test['title'] = df_test['Name'].str.split(',').apply(lambda x: x[1])
df_train['title'] = df_train['title'].str.split('.').apply(lambda x: x[0])

df_test['title'] = df_test['title'].str.split('.').apply(lambda x: x[0])
df_train['title'].value_counts()
df_train['title'] = df_train['title'].apply(lambda x: x.strip())

df_test['title'] = df_test['title'].apply(lambda x: x.strip())
df_train['title'] = np.where(((df_train['title']== 'Rev') | (df_train['title']== 'Col') |

                                (df_train['title']== 'Major') | (df_train['title']== 'Capt') |

                                (df_train['title']== 'Don') | (df_train['title']== 'Sir') |

                                (df_train['title']== 'Jonkheer') | (df_train['title']== 'Dr')), 'Mr', df_train['title'])



df_train['title'] = np.where(((df_train['title']== 'Mlle') | (df_train['title']== 'Lady') | 

                             (df_train['title']== 'the Countess') | (df_train['title']== 'Mme') |

                             (df_train['title']== 'Dona')), 'Mrs', df_train['title'])



df_train['title'] = np.where(df_train['title']=='Ms', 'Miss',df_train['title'])
df_test['title'] = np.where(((df_test['title']== 'Rev') | (df_test['title']== 'Col') |

                                (df_test['title']== 'Major') | (df_test['title']== 'Capt') |

                                (df_test['title']== 'Don') | (df_test['title']== 'Sir') |

                                (df_test['title']== 'Jonkheer') | (df_test['title']== 'Dr')), 'Mr', df_test['title'])



df_test['title'] = np.where(((df_test['title']== 'Mlle') | (df_test['title']== 'Lady') | 

                             (df_test['title']== 'the Countess') | (df_test['title']== 'Mme') |

                             (df_test['title']== 'Dona')), 'Mrs', df_test['title'])



df_test['title'] = np.where(df_test['title']=='Ms', 'Miss',df_test['title'])
df_train['title'].value_counts()
df_train['title_enc'] = df_train['title'].map({'Mr':0, 'Miss':1,'Mrs':2,'Master':3}).astype('category')

df_test['title_enc'] = df_test['title'].map({'Mr':0, 'Miss':1,'Mrs':2,'Master':3}).astype('category')
# For train data

df_train['Age'] = np.where(((df_train['title']=='Mrs') & (df_train['Age'].isna())), np.nanmedian(df_train[df_train['title']=='Mrs']['Age']), df_train['Age'])

df_train['Age'] = np.where(((df_train['title']=='Miss') & (df_train['Age'].isna())), np.nanmedian(df_train[df_train['title']=='Miss']['Age']), df_train['Age'])

df_train['Age'] = np.where(((df_train['title']=='Mr') & (df_train['Age'].isna())), np.nanmedian(df_train[df_train['title']=='Mr']['Age']), df_train['Age'])

df_train['Age'] = np.where(((df_train['title']=='Master') & (df_train['Age'].isna())), np.nanmedian(df_train[df_train['title']=='Master']['Age']), df_train['Age'])

# For test data



df_test['Age'] = np.where(((df_test['title']=='Mrs') & (df_test['Age'].isna())), np.nanmedian(df_test[df_test['title']=='Mrs']['Age']), df_test['Age'])

df_test['Age'] = np.where(((df_test['title']=='Miss') & (df_test['Age'].isna())), np.nanmedian(df_test[df_test['title']=='Miss']['Age']), df_test['Age'])

df_test['Age'] = np.where(((df_test['title']=='Mr') & (df_test['Age'].isna())), np.nanmedian(df_test[df_test['title']=='Mr']['Age']), df_test['Age'])

df_test['Age'] = np.where(((df_test['title']=='Master') & (df_test['Age'].isna())), np.nanmedian(df_test[df_test['title']=='Master']['Age']), df_test['Age'])

df_train.isna().sum()
df_test.isna().sum()
df_train.columns
df_test.columns
tr_df_reducd = df_train.drop(['Name','PassengerId','Ticket','Cabin','Cab_alpha','title'], axis = 1)

test_df_reducd = df_test.drop(['Name','PassengerId','Ticket','Cabin','Cab_alpha','title'], axis = 1)
test_df_reducd.columns
y = tr_df_reducd['Survived']

x = tr_df_reducd.drop(['Survived'], axis=1)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(x,y)
# Scaling attributes using Standard Scaler



num_cols = ['Age','Fare']



X_train_scaled = X_train.copy()

X_val_scaled = X_val.copy()

X_test_scaled = test_df_reducd.copy()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train[num_cols])

X_train_scaled[num_cols] = scaler.transform(X_train_scaled[num_cols])

X_val_scaled[num_cols] = scaler.transform(X_val_scaled[num_cols])

X_test_scaled[num_cols] = scaler.transform(X_test_scaled[num_cols])
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_train_scaled,y_train)

train_pred = log_reg.predict(X_train_scaled)

validation_pred = log_reg.predict(X_val_scaled)

test_pred = log_reg.predict(X_test_scaled)
print(test_pred)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score



print("Train data")

print("Accuracy score: ", accuracy_score(y_train, train_pred))

print("f1 score: ", f1_score(y_train, train_pred))

print("recall score: ", recall_score(y_train, train_pred))

print("precision: ", precision_score(y_train, train_pred))

print("   ")

print("Validation data")

print("Accuracy score: ", accuracy_score(y_val, validation_pred))

print("f1 score: ", f1_score(y_val, validation_pred))

print("recall score: ", recall_score(y_val, validation_pred))

print("precision: ", precision_score(y_val, validation_pred))
pd.crosstab(y_train, train_pred)
pd.crosstab(y_val, validation_pred)