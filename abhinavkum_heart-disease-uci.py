import numpy as np

import pandas as pd 





from matplotlib import pyplot as plt

import seaborn as sns

sns.set()





import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn 



from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, confusion_matrix, auc
data = pd.read_csv('../input/heart.csv')

data.head()
data.info()
data.loc[data['sex']==0, 'sex'] = 'female'

data.loc[data['sex']==1, 'sex'] = 'male'



data.loc[data['cp'] == 0, 'cp'] = 'typical angina'

data.loc[data['cp'] == 1, 'cp'] = 'atypical angina'

data.loc[data['cp'] == 2, 'cp'] = 'non-anginal pain'

data.loc[data['cp'] == 3, 'cp'] = 'asymptomatic'



data.loc[data['restecg'] == 0, 'restecg'] = 'normal'

data.loc[data['restecg'] == 1, 'restecg'] = 'ST-T wave abnormality'

data.loc[data['restecg'] == 2, 'restecg'] = 'left ventricular hypertrophy'



data.loc[data['slope'] == 0, 'slope'] = 'upsloping'

data.loc[data['slope'] == 1, 'slope'] = 'flat'

data.loc[data['slope'] == 2, 'slope'] = 'downsloping'



data.loc[data['thal'] == 0, 'thal'] = 'normal'     # no info about 0 so we assume its normal

data.loc[data['thal'] == 1, 'thal'] = 'normal'

data.loc[data['thal'] == 2, 'thal'] = 'fixed defect'

data.loc[data['thal'] == 3, 'thal'] = 'reversable defect'

listed = ['sex', 'cp', 'restecg', 'slope', 'thal']

for i in listed:

    print('\n',i, '\n')

    print(data[i].value_counts())

data.columns.to_series().groupby(data.dtypes).groups
data.target.value_counts().plot(kind = 'pie', autopct = '%0.1f%%', explode = [0,0.05], shadow = True)
sns.distplot(data.loc[data['target'] == 1, 'age'])
checkname = {

    'age' : 'AGE',

    'trestbps' : ' resting blood pressure (mm Hg)',

    'chol' : 'Cholesterol (mg/dl)',

    'fbs' : 'Fasting blood sugar',

    'thalach' : 'maximum heart rate achieved',

    'exang' :  ' Exercise induced angina ',

    'ca' : 'number of major vessels'

}
sns.catplot(x = 'sex',y = 'age', hue = 'target', col = 'thal', kind = 'violin', data = data)

plt.xlabel('Sex')

plt.ylabel(checkname['age'])

plt.show()
sns.catplot(x = 'sex',y = 'age', hue = 'target', col = 'restecg', kind = 'violin', data = data)

plt.xlabel('Sex')

plt.ylabel(checkname['age'])

plt.show()
sns.catplot(x = 'sex',y = 'age', hue = 'target', col = 'cp', kind = 'violin', col_wrap=2, data = data)

plt.xlabel('Sex')

plt.ylabel(checkname['age'])

plt.show()
sns.catplot(x = 'sex',y = 'trestbps', hue = 'target',  kind = 'violin',data = data)

plt.xlabel('Sex')

plt.ylabel('Resting blood pressure')

plt.show()
sns.catplot(x = 'sex',y = 'chol', hue = 'target',  kind = 'violin',data = data)

plt.xlabel('Sex')

plt.ylabel('cholesterol mg/dl')

plt.show()
sns.catplot(x = 'sex',y = 'fbs', hue = 'target',  kind = 'violin',data = data)

plt.xlabel('Sex')

plt.ylabel(checkname['fbs'])

plt.show()
sns.catplot(x = 'sex',y = 'thalach', hue = 'target',  kind = 'violin',data = data)

plt.xlabel('Sex')

plt.ylabel(checkname['thalach'])

plt.show()
sns.relplot(x="age", y="thalach", hue = 'target', col = 'thal',  kind = 'scatter', data=data)
sns.relplot(x="age", y="thalach", hue = 'target', col = 'cp',  kind = 'scatter', col_wrap = 2, height = 3, data = data)
sns.relplot(x="chol", y="thalach", hue = 'target', size = 'exang', col = 'thal',  kind = 'scatter', data=data)

data = pd.get_dummies(data, drop_first=True)
data.shape
X = data.iloc[:,:-1]

y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=42)
def learning_curves(estimator,name,  X = X , y  = y, train_sizes = np.linspace(0.1, 1, 5), cv=5, score = 'accuracy'):

    train_sizes, train_scores, validation_scores = learning_curve(estimator, X, y, train_sizes = train_sizes,cv = cv,

                                                                  scoring = score)

    

    train_scores_mean = (train_scores).mean(axis = 1)

    validation_scores_mean = (validation_scores).mean(axis = 1)



    plt.plot(train_sizes, train_scores_mean, label = 'Training error')

    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

    plt.ylabel(score, fontsize = 10)

    plt.xlabel('Training set size', fontsize = 10)

    title = 'LC for a ' + name

    plt.title(title, fontsize = 12, y = 1.03)

    plt.legend()

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)

y_pred_quant = model_rf.predict_proba(X_test)[:, 1]

cm  = confusion_matrix(y_test, y_pred)

print(cm)
for i in np.linspace(0.20,0.50, num = 10):

    print('\n for threshold {:0.2f} \n'.format(i))

    y_pred = (y_pred_quant>i).astype(int)

    cm  = confusion_matrix(y_test, y_pred)

    print(cm)
y_pred = (y_pred_quant>0.38).astype(int)



total=sum(sum(cm))



sensitivity = cm[0,0] / (cm[0,0] + cm[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm[1,1] / (cm[1,1] + cm[0,1])

print('Specificity : ', specificity)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr, tpr)
learning_curves(model_rf, 'Random Forest', score = 'accuracy')
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf_random = RandomizedSearchCV(estimator = model_rf, param_distributions = random_grid, n_iter = 100,

                               cv = 3, verbose=2, random_state=42, n_jobs = -1)



rf_random.fit(X_train, y_train)
rf_random.best_params_
model2_rf = rf_random.best_estimator_

new_score = model2_rf.fit(X_train, y_train).score(X_test, y_test)

old_score = model_rf.fit(X_train, y_train).score(X_test, y_test)

print('so improvement is {:0.2f}%'.format(100*(new_score-old_score)/old_score))
model_rf.fit(X_train, y_train).score(X_test, y_test)
from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [False],

    'max_depth': [25, 30, 35, 40],

    'max_features': ['auto', 'sqrt'],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [1400, 1600, 1800, 2000]

}



grid_search = GridSearchCV(estimator = model_rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2).fit(X_train, y_train)
model3_rf = grid_search.best_estimator_

new_score3 = model3_rf.fit(X_train, y_train).score(X_test, y_test)

new_score2 = model2_rf.fit(X_train, y_train).score(X_test, y_test)

print('so improvement is {:0.2f}%'.format(100*(new_score3-new_score2)/new_score2))
y_pred = model3_rf.predict(X_test)

y_pred_quant = model3_rf.predict_proba(X_test)[:, 1]

cm  = confusion_matrix(y_test, y_pred)

print(cm)
for i in np.linspace(0.20,0.5, num = 10):

    print('\n for threshold {:0.2f} \n'.format(i))

    y_pred = (y_pred_quant>i).astype(int)

    cm  = confusion_matrix(y_test, y_pred)

    print(cm)
y_pred = (y_pred_quant > 0.47).astype(int)



total=sum(sum(cm))



sensitivity = cm[0,0] / (cm[0,0] + cm[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm[1,1] / (cm[1,1] + cm[0,1])

print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr, tpr)
learning_curves(model3_rf, 'Random Forest', score = 'accuracy')