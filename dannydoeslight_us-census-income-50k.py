import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df32K = pd.read_csv('/kaggle/input/uci-us-census-income-original-dataset/adult.data')

df32K.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', '50K']
df16K = pd.read_csv('/kaggle/input/uci-us-census-income-original-dataset/adult.test')

df16K = df16K.reset_index()

df16K.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 

                 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 

                 'native_country', '50K']
df48K = df32K.append(df16K, ignore_index=True, sort=False) 
df48K.native_country.value_counts()/len(df48K)
# As you can see United States count for almost 90% of the native_country data,.. 

# ..and won't help that much with our analysis, for which I will drop this column.

# Will drop as well 'fnlwgt' which is an US State weigh and 'education_num' which is the duplicate of 'education'. 

df48K = df48K.drop(['native_country', 'fnlwgt', 'education_num'], axis=1) 

# here you have details for final weight fnlwgt: https://www.kaggle.com/uciml/adult-census-income
# will have just a little bit of fun with the features, just to simplify things a bit:

df48K['marital_status'] = df48K['marital_status'].replace({'Never-married','Divorced', 'Separated', 'Widowed'}, 'Single', regex=True)

df48K['marital_status'] = df48K['marital_status'].replace({'Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'}, 'Married', regex=True)

                                                          

df48K['education'] = df48K['education'].replace({'Preschool','1st-4th','5th-6th', '7th-8th'}, 'Elementary-School', regex=True)

df48K['education'] = df48K['education'].replace({'9th','10th', '11th', '12th', 'HS-grad'}, 'High-School', regex=True)

df48K['education'] = df48K['education'].replace({'Masters', 'Doctorate'}, 'Advanced-Studies', regex=True)

df48K['education'] = df48K['education'].replace({'Bachelors', 'Some-college'}, 'College', regex=True)

df48K['education'] = df48K['education'].replace({'Prof-school', 'Assoc-acdm', 'Assoc-voc'}, 'Professional-School', regex=True)



df48K['workclass'] = df48K['workclass'].replace({'Self-emp-inc', 'Self-emp-not-inc'}, 'SelfEmployed', regex=True)

df48K['workclass'] = df48K['workclass'].replace({'Local-gov', 'State-gov', 'Federal-gov'}, 'Gov-job', regex=True)

df48K['workclass'] = df48K['workclass'].replace({'Without-pay','Never-worked'}, 'Unemployed', regex=True)



df48K['50K'] = df48K['50K'].replace({'<=50K.'}, '<=50K', regex=True)

df48K['50K'] = df48K['50K'].replace({'>50K.'}, '>50K', regex=True)
# having a quick look at our proud cleaning:

df48K.head()
df48K['50K'].value_counts(dropna=False)/len(df48K)

# Looks like our set is imbalanced, as most of the data sets out there in the wild.
# Plotting time! Let`s see how things are:

facet = sns.FacetGrid(df48K, hue="50K", aspect=4)

facet.map(sns.kdeplot,'age', shade= True)

facet.set(xlim=(0, df48K['age'].max()))

facet.add_legend()

# Looks like the income peak is between mid 30`s to late 40`s.
facet = sns.FacetGrid(df48K, hue="50K", aspect=4)

facet.map(sns.kdeplot,'hours_per_week', shade= True)

facet.set(xlim=(0, df48K['hours_per_week'].max()))

facet.add_legend()

# clearly that once you work more than 40 hours/week you get a higher income
sns.countplot('50K', hue='education', data=df48K)

# College and Advanced-Studies do get more jobs that pay higher than 50K.
sns.countplot('50K', hue='sex', data=df48K)

# the gender gap of the 90`s..
sns.countplot('50K', hue='race', data=df48K)
# There are a few cells that hold the value: '?' 

# - which either you can replace it with NaN's - though I don`t recommend since It might capture valuable information

# - or you add it to the value that occurs the most or whatever else makes sense from case to case 

plt.figure(figsize=(10,8))

sns.countplot('50K', hue='occupation', data=df48K)

# this shows that most of the '?' are in the <50K .. for which I will add them to 'Other-service' which shows similar behavior and high numbers. 
df48K['occupation'] = df48K['occupation'].replace({r'\?', 'Other-service'}, 'Other-service', regex=True)
plt.figure(figsize=(5,8))

sns.countplot('50K', hue='occupation', data=df48K)

# I think it looks better now..
sns.countplot('50K', hue='workclass', data=df48K)

# will deal with '?' same as above
df48K['workclass'] = df48K['workclass'].replace({r'\?', 'Private'}, 'Private', regex=True)
sns.countplot('50K', hue='workclass', data=df48K) 

# better, no? 
sns.countplot('50K', hue='marital_status', data=df48K)

# seems that stability and responsibilities offered by having a family could increase your income.
plt.figure(figsize=(20,8))

sns.countplot('occupation', hue='workclass', data=df48K)

# interesting to see the dynamics between workclasses, this will definitely help our analysis.
# preparing the data for spliting into train, validation and test with train_test_split

y = df48K[['50K']]  #pt arrray .values

X = df48K.drop(['50K'], axis=1)

X = pd.get_dummies(X, drop_first=True)

y = pd.get_dummies(y, drop_first=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=9)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=9)
# depending on the model you use you might need a little bit of normalization and standardization:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #copy=True, with_mean=True, with_std=True

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)
# Now let`s pick some models and see which performs best.

# I won`t use any for loops or GridSearchCV since would be too cumbersome for this notebook.. 

# ..though I have had on my own system and got to some conclusions that I will input bellow.

# Let`s look at the first model: knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=28)

knn.fit(X_train_scaled, y_train)
# Model Evaluation:

# a) On train data

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

print(knn.score(X_train_scaled, y_train))

y_pred = knn.predict(X_train_scaled)

print(confusion_matrix(y_pred, y_train))  

print(classification_report(y_pred, y_train, target_names=[">50K", "<50K"])) 
# b) On New, unseen data

print(knn.score(X_val_scaled, y_val))

y_pred = knn.predict(X_val_scaled)

print(confusion_matrix(y_pred, y_val)) 

print(classification_report(y_pred, y_val, target_names=[">50K", "<50K"]))
y_pred = knn.predict(X_test_scaled)

print(confusion_matrix(y_pred, y_test))

print(classification_report(y_pred, y_test, target_names=[">50K", "<50K"]))
# Now let`s try: LogisticRegression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train_scaled, y_train)

print(logreg.score(X_train_scaled, y_train))

y_pred = logreg.predict(X_train_scaled)

print(confusion_matrix(y_pred, y_train)) 

print(classification_report(y_pred, y_train))
print(logreg.score(X_val_scaled, y_val))

y_pred = logreg.predict(X_val_scaled)

print(confusion_matrix(y_pred, y_val)) 

print(classification_report(y_pred, y_val))

y_pred = logreg.predict(X_test_scaled)

print(confusion_matrix(y_pred, y_test))

print(classification_report(y_pred, y_test))
# Last but not least, let`s try: random forest

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=500, random_state=9, verbose=1)

forest.fit(X_train_scaled, y_train) # if you don`t like the overfit feel free to use max_features and max_depth, you might even get a slightly higher f1 ;) 

print(forest.score(X_train_scaled, y_train))

y_pred = forest.predict(X_train_scaled)

print(confusion_matrix(y_pred, y_train)) 

print(classification_report(y_pred, y_train, target_names=[">50K", "<50K"]))
# on unseen data

print(forest.score(X_val_scaled, y_val))

y_pred = forest.predict(X_val_scaled)

print(confusion_matrix(y_pred, y_val)) 

print(classification_report(y_pred, y_val))
y_pred = forest.predict(X_test_scaled)

print(confusion_matrix(y_pred, y_test))

print(classification_report(y_pred, y_test, target_names=[">50K", "<50K"]))
# Let's have a look at the scores on X_val:

# knn: 0.8372

# logreg: 0.8509

# randomforest: 0.8467



#.. as well at f1-scores for X_test:

# knn: 0.59           TP: 614

# logreg: 0.63        TP: 668

# randomforest: 0.63  TP: 687 



# Conclusion: if I want high score on X_val I`ll go with logreg or..

# (f1 being same) if looking for as many true positives as possible (without modifying the recall) will go with random forest... 