import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#reading the dataset using pandas
data = pd.read_csv('../input/pimadatacsv/pima-data.csv')
data.head()

#finding out missing values

data.isnull().sum()
diabetes_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(diabetes_map)
data.head()
#checking whether the target value data is balanced or imbalanced

count_diabetes = pd.value_counts(data['diabetes'])
count_diabetes
count_diabetes.plot(kind='bar')

#finding the correlation between dependent & independent variables

data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot = True, cmap = "RdYlGn")
print("number of (0) values present in num_preg: {0}". format(len(data.loc[data['num_preg'] == 0 ])))
print("number of (0) values present in glucose_conc: {0}". format(len(data.loc[data['glucose_conc']==0])))
print("number of (0) values present in diastolic_dp: {0}". format(len(data.loc[data['diastolic_bp']==0])))
print("number of (0) values present in thickness: {0}".format(len(data.loc[data['thickness']== 0 ])))
print("number of (0) values present in insulin:{0}". format(len(data.loc[data['insulin']==0])))
print("number of (0) values present in bmi: {0}". format(len(data.loc[data['bmi']==0])))
print("number of (0) values present in diab_pred: {0}". format(len(data.loc[data['diab_pred']==0])))
print("number of (0) values present in age:{0}". format(len(data.loc[data['age']==0])))
print("number of (0) values present in skin: {0}". format(len(data.loc[data['skin']==0])))
#handling missing(0) values
from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values = 0, strategy = "mean" )

filled_data = fill_values.fit_transform(data.iloc[:,0:9])
filled_data.shape
X = pd.DataFrame(data=filled_data, columns=['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age','skin'])
X.head()
Y = data.iloc[:,9:10]
Y.head()
#creating the train test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , train_size = 0.70, random_state = 0)
#Training the data with randomforestclassifier

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=200,criterion='entropy')

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

score = accuracy_score(Y_test,Y_pred)
score

print(confusion_matrix(Y_test,Y_pred))

print(classification_report(Y_test,Y_pred))
parameters = {"n_estimators":[100,200,300,400], "criterion":['gini','entropy'], "max_depth":[2,4,6,8,10,12],
              "min_samples_split":[2,4,5,6,8],"min_samples_leaf":[1,2,3,4,5,6,7],"max_features":['auto','sqrt','log2']}
   

from sklearn.model_selection import RandomizedSearchCV

random_classifier = RandomForestClassifier()

random_search = RandomizedSearchCV(random_classifier,param_distributions=parameters,n_iter=5,cv=5) 

random_search.fit(X_train,Y_train)
random_search.best_params_
random_search
best_parameters = random_search.best_estimator_
Y_pred_values = best_parameters.predict(X_test)
print("CONFUSION MATRIX")
print(confusion_matrix(Y_pred_values,Y_test))
print("AccuarcyScore:{}".format(accuracy_score(Y_pred_values,Y_test)))

print("Classification Report")
print(classification_report(Y_pred_values,Y_test))

