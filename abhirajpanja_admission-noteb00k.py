#Importing the necessary packages



import collections



import matplotlib.pyplot as plt



import numpy as np

import pandas as pd



import scipy

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, plot_roc_curve, accuracy_score, mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

data.head()
data.describe()
data.isnull().sum()
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')
df = data.drop('Serial No.', axis=1)

df.rename(columns={'Chance of Admit ': 'Chance of Admit'}, inplace=True)

df.columns
df.head()
plt.hist(df['Chance of Admit'])

plt.xlabel("Chance of Admit")

plt.ylabel("Count")

plt.show()
sns.pairplot(df)
sns.kdeplot(df['Chance of Admit'], df['GRE Score'], cmap='Blues', shade=True, shade_lowest=False)
sns.kdeplot(df['Chance of Admit'], df['University Rating'], cmap='Blues', shade=True, shade_lowest=False)
sns.scatterplot(df['University Rating'], df['GRE Score'])
for i in np.arange(0, 1, 0.18):

    #print(i)

    print(i, df[df['Chance of Admit'] > i].shape[0]/len(df))
df['Label'] = np.where(df['Chance of Admit']>0.72, 1, 0)
print(df.Label.value_counts())

df.sample(10)
df.shape
#Checking feature importance with DTree classifier

# define the model

model = DecisionTreeClassifier()



X = df.drop(columns = ['Chance of Admit', 'Label'])

y = df['Label']



# fit the model

model.fit(X, y)



feature_importance = model.feature_importances_



for i,v in enumerate(feature_importance):

    print('Feature: %0d,  Score: %.5f' % (i,v))

    

feature_importance = pd.Series(feature_importance, index=X.columns)

feature_importance.plot(kind='barh')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
def plot_roc(false_positive_rate, true_positive_rate, roc_auc):

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
parameters = [

    {

        'penalty' : ['l1', 'l2', 'elasticnet'],

        'C' : [0.1, 0.4, 0.5],

        'random_state' : [0]

    }

]

model_logit = LogisticRegression()

gscv = GridSearchCV(model_logit, parameters, scoring='accuracy')

gscv.fit(X_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_train), y_train))

print(confusion_matrix(gscv.predict(X_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_test), y_test))

print(confusion_matrix(gscv.predict(X_test), y_test))



cvs = cross_val_score(estimator=model_logit, X=X_train, y=y_train, cv=12)

print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
lr_model = LogisticRegression(C= 0.1, penalty= 'l2', random_state= 0)

lr_model.fit(X_train, y_train)



y_pred = lr_model.predict(X_test)

y_pred_proba = lr_model.predict_proba(X_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print(cm)

#print("Accurecy is : ", (cm[0,0]+cm[1,1])/cm.sum())
parameters = [

    {

        'criterion' : ['gini', 'entropy'],

        'max_depth' : [3, 4, 5],

        'min_samples_split' : [10, 20, 5],

        'random_state': [0],

        

    }

]

model_dt = DecisionTreeClassifier()

gscv = GridSearchCV(model_dt, parameters, scoring='accuracy')

gscv.fit(X_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_train), y_train))

print(confusion_matrix(gscv.predict(X_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_test), y_test))

print(confusion_matrix(gscv.predict(X_test), y_test))



cvs = cross_val_score(estimator=model_logit, X=X_train, y=y_train, cv=12)

print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
dt_model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 3, min_samples_split= 10, random_state= 0)

dt_model.fit(X_train, y_train)



y_pred = dt_model.predict(X_test)

y_pred_proba = dt_model.predict_proba(X_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print(cm)

#print("Accurecy is : ", (cm[0,0]+cm[1,1])/cm.sum())
parameters = [

    {

        'n_estimators': np.arange(10, 40, 5),

        'criterion' : ['gini', 'entropy'],

        'max_depth' : [3, 4, 5],

        'min_samples_split' : [10, 20, 5],

        'random_state': [0],

        

    }

]

model_rf = RandomForestClassifier()

gscv = GridSearchCV(model_rf, parameters, scoring='accuracy')

gscv.fit(X_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_train), y_train))

print(confusion_matrix(gscv.predict(X_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_test), y_test))

print(confusion_matrix(gscv.predict(X_test), y_test))



cvs = cross_val_score(estimator=model_logit, X=X_train, y=y_train, cv=12)

print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
rf_model = RandomForestClassifier(criterion= 'entropy', max_depth= 4, min_samples_split= 10,n_estimators= 10, random_state= 0)

rf_model.fit(X_train, y_train)



y_pred = rf_model.predict(X_test)

y_pred_proba = rf_model.predict_proba(X_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print(cm)

#print("Accurecy is : ", (cm[0,0]+cm[1,1])/cm.sum())
parameters = [

    {

        'learning_rate': [0.01, 0.02, 0.002],

        'n_estimators': np.arange(10, 40, 5),

        'max_depth' : [3, 4, 5],

        'min_samples_split' : [10, 20, 5],

        'random_state': [0],

        

    }

]

model_gbc = GradientBoostingClassifier()

gscv = GridSearchCV(model_gbc, parameters, scoring='accuracy')

gscv.fit(X_train, y_train)



print('Best parameters set:')

print(gscv.best_params_)

print()



print("*"*50)

print("Train classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_train), y_train))

print(confusion_matrix(gscv.predict(X_train), y_train))



print()

print("*"*50)

print("Test classification report: ")

print("*"*50)

print(classification_report(gscv.predict(X_test), y_test))

print(confusion_matrix(gscv.predict(X_test), y_test))



cvs = cross_val_score(estimator=model_logit, X=X_train, y=y_train, cv=12)

print()

print("*"*50)

print(cvs.mean())

print(cvs.std())
gbc_model = GradientBoostingClassifier(learning_rate= 0.01, max_depth= 3, min_samples_split= 20,n_estimators= 20, random_state= 0)

gbc_model.fit(X_train, y_train)



y_pred = gbc_model.predict(X_test)

y_pred_proba = gbc_model.predict_proba(X_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



plot_roc(false_positive_rate, true_positive_rate, roc_auc)



print('Accurancy Score :',accuracy_score(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print(cm)

#print("Accurecy is : ", (cm[0,0]+cm[1,1])/cm.sum())
y_proba=rf_model.predict(X_test)

np.sqrt(mean_squared_error(y_proba, y_test))
