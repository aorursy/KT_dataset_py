import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

# Read your datasets from the installed directory

df.head() #prints the first 5 rows of the dataset
df.info()
df.describe()
df.shape
from pandas_profiling import ProfileReport

report = ProfileReport(df,title='Summary Report of Student Placements')

report
greater_70 = (df.ssc_p > 70) & (df.hsc_p > 70) & (df.mba_p >70)
df_70 = df[greater_70]

df_70.shape
plt.hist(df_70.salary,bins=20)

plt.show()
df['workex'].value_counts() #Since its a categorical variable it does not require normalization
df['specialisation'].value_counts()
df['degree_t'].value_counts()
df['hsc_s'].value_counts()
df['status'].value_counts()
df['gender'].value_counts()
import matplotlib.pyplot as plt 
plt.hist(df['salary'],bins=20)

plt.show()
plt.scatter(df['ssc_p'],df['salary'])

plt.xlabel('Percentage in SSC')

plt.ylabel('Salary Offered')

plt.title('Salary offered wrt SSC Percentage')

plt.show()
plt.scatter(df['hsc_p'],df['salary'])

plt.xlabel('Percentage in HSC')

plt.ylabel('Salary Offered')

plt.title('Salary offered wrt HSC Percentage')

plt.show()
plt.scatter(df['degree_p'],df['salary'])

plt.xlabel('Percentage in Degree')

plt.ylabel('Salary Offered')

plt.title('Salary offered wrt Degree Percentage')

plt.show()
plt.scatter(df['mba_p'],df['salary'])

plt.xlabel('Percentage in MBA')

plt.ylabel('Salary Offered')

plt.title('Salary offered wrt MBA Percentage')

plt.show()
plt.hist(df['salary'],bins=20)

plt.show()
from scipy import stats



degree_p = stats.norm.rvs(df['degree_p'])

ssc_p = stats.norm.rvs(df['ssc_p'])

hsc_p = stats.norm.rvs(df['hsc_p'])

mba_p = stats.norm.rvs(df['mba_p'])

salary = stats.norm.rvs(df['salary'])

etest_p = stats.norm.rvs(df['etest_p'])

print("Stat for degree:", stats.shapiro(degree_p)) # Null Accepted

print("Stat for ssc:", stats.shapiro(ssc_p)) # Null Accepted

print("Stat for hsc:", stats.shapiro(hsc_p)) # Null Rejected

print("Stat for mba:", stats.shapiro(mba_p)) # Null Accepted

print("Stat for salary:", stats.shapiro(salary)) # Null Accepted 

print("Stat for etest:", stats.shapiro(etest_p)) # Null Rejected
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

columns = df[['ssc_p','hsc_p','degree_p','mba_p','etest_p']]

x_scaled = pd.DataFrame(scaler.fit_transform(columns))

x_scaled.columns = ['ssc_p','hsc_p','degree_p','mba_p','etest_p']

x_scaled.reset_index(drop=True, inplace=True)

x_scaled
x_cat = df[['gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation']]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x_cat['gender'] = le.fit_transform(x_cat.gender)

x_cat['ssc_b'] = le.fit_transform(x_cat.ssc_b)

x_cat['hsc_b'] = le.fit_transform(x_cat.hsc_b)

x_cat['hsc_s'] = le.fit_transform(x_cat.hsc_s)

x_cat['degree_t'] = le.fit_transform(x_cat.degree_t)

x_cat['specialisation'] = le.fit_transform(x_cat.specialisation)

x_cat.reset_index(drop=True, inplace=True)

x_cat
x = pd.concat([x_cat,x_scaled],join='outer',axis=1)

x.isnull().sum()

x
y = le.fit_transform(df.status)
from sklearn.model_selection import train_test_split as tts



x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

lrscore = lr.score(x_test,y_test)

lrscore
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()

knc.fit(x_train,y_train)

y_pred = knc.predict(x_test)

kncscore = knc.score(x_test,y_test)

kncscore
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

dtr = DecisionTreeClassifier(random_state=42)

dtr.fit(x_train,y_train)

y_pred = dtr.predict(x_test)



dtcscore =  metrics.accuracy_score(y_test,y_pred)

print(f'Decision Tree Classification Score = {dtcscore:4.1f}%\n')

print(f'Classification Report:\n {metrics.classification_report(y_test, y_pred)}\n')
from sklearn.ensemble import RandomForestClassifier



rfr = RandomForestClassifier(n_estimators=10,random_state=42)

rfr.fit(x_train,y_train)



from sklearn import metrics



predicted = rfr.predict(x_test)

rfcscore =  metrics.accuracy_score(y_test, predicted)

print(f'Random Forest Classification Score = {rfcscore:4.1f}%\n')

print(f'Classification Report:\n {metrics.classification_report(y_test, predicted)}\n')
from sklearn.linear_model import RidgeClassifier

rc = RidgeClassifier(random_state=42)

rc.fit(x_train,y_train)

l_pred = rc.predict(x_test)

rcscore = rc.score(x_test,y_test)

rcscore
from sklearn.linear_model import SGDClassifier

SGDC = SGDClassifier(random_state=42)

SGDC.fit(x_train,y_train)

result = SGDC.predict(x_test)

sgdcscore = SGDC.score(x_test,y_test)

sgdcscore
from sklearn.linear_model import Perceptron

p = Perceptron(random_state=42)

p.fit(x_train,y_train)

result = p.predict(x_test)

pscore = p.score(x_test,y_test)

pscore
from sklearn.linear_model import PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(random_state=42)

pac.fit(x_train,y_train)

result_pac = pac.predict(x_test)

pacscore = pac.score(x_test,y_test)

pacscore
from sklearn.svm import SVC

svc = SVC(random_state=42)

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

svcscore = svc.score(x_test,y_test)

svcscore
from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(random_state=43)

bc.fit(x_train,y_train)

y_pred = bc.predict(x_test)

bcscore = bc.score(x_test,y_test)

bcscore
d = {'Algorithms Used': ['Logistic Regression','K Neighbors Classifier','Decision Tree Classifier','Random Forest Classifier',

                         'Ridge Classifier','Stochastic Gradient Descent','Perceptron','Passive Aggressive Classifier',

                        'Support Vector Classifier','Bagging Classifier'],

    'Accuracy Achieved': [lrscore,kncscore,dtcscore,rfcscore,rcscore,sgdcscore,pscore,pacscore,svcscore,bcscore]}
Accuracy_df = pd.DataFrame(d)

Accuracy_df = Accuracy_df.sort_values(by=['Accuracy Achieved'],ascending=False)

Accuracy_df
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.feature_selection import RFE



pac = PassiveAggressiveClassifier(random_state=42)

rfe = RFE(pac,1)

rfe.fit(x_train,y_train)

for var, name in sorted(zip(rfe.ranking_,x), key=lambda x: x[0]):

    print(f'{name:>18} rank = {var}')
from sklearn import metrics

import seaborn as sns
matrix = metrics.confusion_matrix(y_test,result_pac)

report = metrics.classification_report(y_test,result_pac)

print(f'Classification Report:\n {metrics.classification_report(y_test,result_pac)}\n')
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from time import time

from sklearn.linear_model import PassiveAggressiveClassifier



# Start clock

start = time()



pac = PassiveAggressiveClassifier()

skf = StratifiedKFold(n_splits=10)



fit_intercept = [True]

validation_fraction = [0.1,0.2,0.3,0.4,0.5,0.6]

loss = ['hinge','squared_hinge']

random_state = [42,33]

class_weight = ['weight','balanced',None]



# Create a dictionary of hyperparameters and values

params = {'fit_intercept':fit_intercept, 'validation_fraction':validation_fraction,'loss':loss,'random_state':random_state,'class_weight':class_weight}



# Number of random parameter samples

num_samples = 20



# Run randomized search

rscv = RandomizedSearchCV(pac, param_distributions=params, n_iter=num_samples, random_state=23)



# Fit grid search estimator and display results

rscv.fit(x_train, y_train)



print(f'Compute time = {time() - start:4.2f} seconds', end='')

print(f' for {num_samples} parameter combinations')
# Get best esimtator

be = rscv.best_estimator_



# Display parameter values

print(f'Best fit_intercept={be.get_params()["fit_intercept"]:5.4f}')

print(f'Best validation_fraction={be.get_params()["validation_fraction"]}')

print(f'Best loss={be.get_params()["loss"]}')

print(f'Best Class_Weight={be.get_params()["class_weight"]}')



# Display best score

print(f'Best CV Score = {rscv.best_score_:4.3f}')