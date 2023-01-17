import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve

import scipy.stats as stats

from matplotlib import pylab
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df
df['Outcome'].value_counts()
df.isnull().sum()
def diagnostic_plots(df, variable):

    

    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)

    df[variable].hist()



    plt.subplot(1, 2, 2)

    stats.probplot(df[variable], dist="norm", plot=pylab)



    plt.show()
diagnostic_plots(df, 'Pregnancies')
df['Pregnancies'] = df.Pregnancies**(1/1.4)



diagnostic_plots(df, 'Pregnancies')
diagnostic_plots(df, 'Glucose')
df['Glucose'] = df.Glucose**(0.95)



diagnostic_plots(df, 'Glucose')
diagnostic_plots(df, 'BloodPressure')
df['BloodPressure'] = df.BloodPressure**1.4



diagnostic_plots(df, 'BloodPressure')
diagnostic_plots(df, 'SkinThickness')
diagnostic_plots(df, 'Insulin')
df['Insulin'] = df.Insulin**0.4



diagnostic_plots(df, 'Insulin')
diagnostic_plots(df, 'BMI')
diagnostic_plots(df, 'DiabetesPedigreeFunction')
df['DiabetesPedigreeFunction'] = df.DiabetesPedigreeFunction**0.1



diagnostic_plots(df, 'DiabetesPedigreeFunction')
diagnostic_plots(df, 'Age')
df['Age']= np.log(df.Age)



diagnostic_plots(df, 'Age')
for feature in df.columns[:-1]:

    IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)

    upper_bond = df[feature].quantile(0.75) + (IQR * 1.5)

    lower_bond = df[feature].quantile(0.25) - (IQR * 1.5)

    

    df[feature] = np.where(df[feature]>upper_bond,upper_bond,df[feature])

    df[feature] = np.where(df[feature]<lower_bond,lower_bond,df[feature])

    
for feature in df.columns[:-1]:

    df[f'{feature}_zero'] = np.where(df[feature]==0,1,0)

    df[feature] = np.where((df[feature]==0) & (df['Outcome']==0),df.groupby('Outcome')[feature].median()[0],df[feature])

    df[feature] = np.where((df[feature]==0) & (df['Outcome']==1),df.groupby('Outcome')[feature].median()[1],df[feature])
X = df.drop('Outcome', axis=1)

X = StandardScaler().fit_transform(X)

y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=0)
y_train.value_counts()
y_test.value_counts()
model = SVC()



parameters = [{'kernel': ['rbf'],

               

               'gamma': [1e-3, 1e-4],

               

               'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5,scoring='roc_auc')

grid.fit(X, y)
grid.best_estimator_
roc_auc = np.mean(cross_val_score(grid, X, y, cv=5, scoring='roc_auc'))

print('Score: {}'.format(roc_auc))
model = SVC(C=1000, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',

    max_iter=-1, probability=True, random_state=None, shrinking=True,

    tol=0.001, verbose=False)
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
print(classification_report(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
roc_auc_score(y_test,y_predicted)
plt.figure(figsize=(7,5))

fpr, tpr, thresh = roc_curve(y_test, y_pred_proba[:,1])

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr,color='blue')

plt.plot([0, 1], [0, 1],'r--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
