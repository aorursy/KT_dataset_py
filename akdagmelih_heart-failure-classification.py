import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df = data.copy()



display(df.head())

df.info()
sns.countplot(df['DEATH_EVENT'], palette=['blue', 'red'])

plt.title('Target Feature Counts', fontsize=20);
plt.pie(df['sex'].value_counts().values, 

        labels=['Men', 'Women'], 

        colors=['cyan', 'pink'], 

        autopct='%1.f%%', 

        shadow=True, 

        startangle=45, 

        textprops={'fontsize':25});
sns.countplot(df['smoking'],

              palette=['orange', 'brown'])

plt.title('Smokers and Non-smokers Counts', fontsize=20);
print('Age Statistics of the Patients' + '\n\n' + str(df.age.describe()))
sns.boxplot(df.age)

plt.title('Age Statistics of the Patients', fontsize=20);
plt.figure(figsize=(15,6))

sns.countplot(df['age'], hue=df['smoking'], palette=['blue', 'red'], alpha=0.7)

plt.title("Age and Smoking", fontsize=20)

plt.xticks(rotation=90)

plt.yticks(list(range(0,27,3)))

plt.grid();
plt.figure(figsize=(11,11))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title('Correlation of the Features', fontsize=20);
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_scaled.head()
X = df_scaled.drop('DEATH_EVENT', axis=1).values

y = df_scaled['DEATH_EVENT'].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)



print('X_train : ', X_train.shape)

print('y_train : ', y_train.shape)

print('X_test  : ', X_test.shape)

print('y_test  : ', y_test.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



# Defining the model creation and evaluation function, so we don't have to write it again and again.



def model_and_eval(max_features, n_estimators, random_state):

    rf = RandomForestClassifier(max_features=max_features, random_state=random_state)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    

    # Printing the model scores:

    print('Mean accuracy  : %.2f' % accuracy_score(y_test, y_pred))

    print('Mean precision : %.2f' % precision_score(y_test, y_pred))

    print('Mean recall    : %.2f' % recall_score(y_test, y_pred))

    print('Mean f1 score  : %.2f' % f1_score(y_test, y_pred))

    

    # Creating the confusion matrix:

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')

    plt.title('Confusion Matrix of the Model', color='navy', fontsize=15)

    plt.xlabel('Predicted Values')

    plt.ylabel('Actual Values');
model_and_eval(max_features='auto', n_estimators=100, random_state=10)
from sklearn.model_selection import GridSearchCV



param_grid = {'n_estimators' : [10, 15, 20, 30, 50, 100, 150], # Number of decision trees

              'max_features' : [0.5, 2, 5, 10, 12]}            # Number of features to consider at each split



rf = RandomForestClassifier(random_state=10)



gs = GridSearchCV(rf, param_grid, cv=10, n_jobs=-1)



gs.fit(X_train, y_train)



print('Best Parameter ', gs.best_params_)
model_and_eval(max_features=gs.best_params_['max_features'], n_estimators=gs.best_params_['n_estimators'], random_state=10)
rf = RandomForestClassifier()

rf.fit(X_train, y_train)



ft_imp = pd.Series(rf.feature_importances_, index=df.iloc[:,:12].columns).sort_values()



ft_imp.plot(kind='barh')

plt.title('Feature Importance', fontsize=20);
X_new = df_scaled[['time', 'serum_creatinine', 'ejection_fraction', 'age', 'creatinine_phosphokinase', 'platelets', 'serum_sodium', 'smoking']]



X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=10)
from sklearn.model_selection import GridSearchCV



n_estimators = list(range(1, 101))



param_grid = {'n_estimators' : n_estimators,

              'max_features' : [2, 5, 10, 12]}



rf = RandomForestClassifier(random_state=42)



gs = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)



gs.fit(X_train, y_train)



scores = gs.cv_results_['mean_test_score']



print('Best Parameter ', gs.best_params_)
best_x = gs.best_params_['n_estimators']

best_y = gs.cv_results_['mean_test_score'][gs.best_params_['n_estimators']-1]



plt.figure(figsize=(15,5))

sns.lineplot(n_estimators, scores[:100], color='navy')

plt.plot(best_x, best_y, marker='o', markersize=8, color="red", label='best_param')

plt.xlabel('n_estimators')

plt.ylabel('Accuracy')

plt.title('Random Forest n_estimators and Accuracy Plot', fontsize=20)

plt.xticks(np.arange(0, 100, 5), rotation=45)

plt.grid();
model_and_eval(max_features=2 , n_estimators=25 , random_state=10)