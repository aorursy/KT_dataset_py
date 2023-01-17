import pandas as pd

from sklearn import tree,model_selection,metrics,neighbors,linear_model

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
diabetes_data.shape
diabetes_data.describe()
def clean_data(data):

    data.loc[data['Glucose']==0,'Glucose'] = data.Glucose.mean()

    data.loc[data['BloodPressure']==0,'BloodPressure'] = data.BloodPressure.mean()

    data.loc[data['SkinThickness']==0,'SkinThickness'] = data.SkinThickness.median()

    data.loc[data['Insulin']==0,'Insulin'] = data.Insulin.median()

    data.loc[data['BMI']==0,'BMI'] = data.BMI.median()



clean_data(diabetes_data)
diabetes_data.isnull().any()
diabetes_data.head()
diabetes_data.hist(figsize=(12,10))

plt.show()
# value counts in Pregnancy column (normalized)

diabetes_data['Pregnancies'].value_counts(normalize=True).plot(kind='bar')
# Plotting the Outcome column that shows if the person has diabetes or not

diabetes_data.Outcome.value_counts(normalize=True).plot(kind='bar')
diabetes_data.Pregnancies[diabetes_data.Outcome==1].value_counts(normalize=True).plot(kind='bar')
preg_values = set(diabetes_data.Pregnancies.values)
for x in preg_values:

    diabetes_data.Outcome[diabetes_data.Pregnancies == x].value_counts(normalize=True).plot(kind='bar')

    print('Pregnancies =',x,', n =',(diabetes_data.Pregnancies == x).sum())

    plt.show()
diabetes_data['Glucose'].plot(kind = 'hist')
sns.pairplot(diabetes_data,hue='Outcome')
sns.heatmap(diabetes_data.corr(),annot=True)

plt.show()

plt.scatter(diabetes_data.Glucose,diabetes_data.Outcome)
X = diabetes_data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y = diabetes_data.Outcome
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.33,random_state = 4)
log_reg = linear_model.LogisticRegression()

log_reg.fit(X_train,y_train)
log_pred = log_reg.predict(X_test)
metrics.accuracy_score(y_test,log_pred)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
metrics.accuracy_score(y_test,knn_pred)
decision_tree = tree.DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)
tree_pred = decision_tree.predict(X_test)
metrics.accuracy_score(y_test,tree_pred)
scoreList = []

for i in [log_reg,knn,decision_tree]:

    score = model_selection.cross_val_score(i,X,y,scoring='accuracy',cv=50)

    scoreList.append(score.mean())
scoreList
x = diabetes_data[diabetes_data.columns[:8]]



pd.Series(decision_tree.feature_importances_,index=X.columns).sort_values(ascending=False)