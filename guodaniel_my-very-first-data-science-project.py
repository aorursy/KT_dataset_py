import pandas as pd 

import numpy as np

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

import seaborn as sns
#read excel file

file = '../input/all.xlsx'

df = pd.read_excel(file, index_col="Jahr")
df.tail()
df.isnull().sum()

df.info()
#show all empty cells (amount=26)

df[df.Zweck.isnull()]
# shows most frequent value

df['Zweck'].value_counts()
df['Zweck'].fillna('Kosten für die Beschäftigung von Übungsleitern')

#print(df.isnull().sum())

print(df)
#list only the numerical ones:

list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
# plot every numeric data

df_num.hist(figsize=(10, 6), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
encode = LabelEncoder()

df['Politikbereich'] = encode.fit_transform(df['Politikbereich'])
df['Politikbereich']
# Deciding the inital threshold to be 0.05% of dataset size

tot_instances = df.shape[0]

threshold = tot_instances*0.0005

print ('The minimum count threshold is: '+str(threshold))
# Apply the count threshold to all the categorical values

obj_columns = list(df.select_dtypes(include=['object']).columns)    # Get a list of all the columns' names with object dtype

del obj_columns[0:4]

df = df.apply(lambda x: x.mask(x.map(x.value_counts())<threshold, 'RARE') if x.name in obj_columns else x)
df_encoded = pd.get_dummies(data=df, columns=obj_columns)

df_encoded.dtypes
df_encoded.head()
df.columns
df['Art'] = [x.replace('institutionelle Förderung','Institutionelle Förderung') for x in df['Art']]
labels = df.Art.value_counts().index

colors = ['pink','orange']

explode = [0,0]

sizes = df.Art.value_counts().values



# visual 

plt.figure(0,figsize = (6,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Art der Ausgabe',color = 'blue',fontsize = 15)

plt.show()
df_bet=df.groupby(["Jahr"])["Betrag"].sum()

#df_bet.drop(df.columns[0])

df_bet.head()
df_med_by_year = df_by_year.sum()

df_rat_by_year = df_med_by_year['Betrag']

plt.scatter(df_rat_by_year.index, df_rat_by_year)

plt.xlabel('Year')

plt.ylabel('Expenses')
plt.scatter(df.index,df['Betrag'],df['Politikbereich'])

plt.xlabel('Year')

plt.ylabel('Expenses')
sns.barplot(x=df.index, y=df['Betrag'])
sns.swarmplot(x=insurance_data['smoker'],

              y=insurance_data['charges'])
# Split the dataset and remove unnecessary columns

X_orig = df_encoded.drop(['Name','Geber','Art','Jahr','Anschrift','Politikbereich'], axis=1)

Y_orig = df['Politikbereich']     # Set the target column
# Split the data set to train and test sets

X_train, X_test, y_train, y_test = train_test_split(X_orig, Y_orig, test_size=0.3, random_state=42)
model = XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=3, 

                      gamma=5)
model.fit(X_train, y_train)
# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
#cross_val_score(LogisticRegression(), X_orig,Y_orig,cv=7)
kfold = StratifiedKFold(n_splits=10, random_state=7)

results=cross_val_score(model, X_orig,Y_orig,cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#randomforest

res =cross_val_score(RandomForestClassifier(n_estimators=400), X_orig,Y_orig,cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train,y_train)

rf.score(X_test,y_test)