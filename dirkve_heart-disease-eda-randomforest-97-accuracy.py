# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

from scipy.stats import norm, shapiro

from sklearn.metrics import accuracy_score

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv(r'/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
data.describe()
color = plt.get_cmap('RdYlGn') 

color.set_bad('lightblue')



corrmat = data.corr()

f, ax = plt.subplots(figsize=(18, 12))

sns.heatmap(corrmat, vmax=.8, annot=True, cmap=color);
# Feature Selection

sns.set_style("darkgrid")



x = data.iloc[:, :-1]

y = data.iloc[:,-1]



model = ExtraTreesClassifier()

model.fit(x,y)

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(12).plot(kind='barh')

plt.show()
features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']



highly_correlated_features = ['age', 'anaemia', 'creatinine_phosphokinase','high_blood_pressure',

                             'serum_creatinine', 'ejection_fraction', 'serum_sodium', 'time']



corr_highly_correlated_features = ['age', 'anaemia', 'creatinine_phosphokinase','high_blood_pressure',

                             'serum_creatinine', 'ejection_fraction', 'serum_sodium', 'time', 'DEATH_EVENT']



corr_features = ['age', 'anaemia', 'creatinine_phosphokinase','high_blood_pressure',

                             'serum_creatinine', 'ejection_fraction', 'serum_sodium', 'time', 'DEATH_EVENT']



training_features = ['serum_creatinine', 'ejection_fraction', 'time', 'age']

skew_feats = data[features].skew().sort_values(ascending=False)

skewness = pd.DataFrame({'Skew':skew_feats})

skewness
#'creatinine_phosphokinase' histogram and normal probability plot

sns.distplot(data['creatinine_phosphokinase'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['creatinine_phosphokinase'], plot=plt)
#Remove skewness by logistic transformation

data['creatinine_phosphokinase'] = np.log(data['creatinine_phosphokinase'])



#adjusted histogram and normal probability plot

sns.distplot(data['creatinine_phosphokinase'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['creatinine_phosphokinase'], plot=plt)
#'serum_creatinine' histogram and normal probability plot

sns.distplot(data['serum_creatinine'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['serum_creatinine'], plot=plt)
#Remove skewness by logistic transformation

data['serum_creatinine'] = np.log(data['serum_creatinine'])



#adjusted histogram and normal probability plot

sns.distplot(data['serum_creatinine'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['serum_creatinine'], plot=plt)
skew_feats = data[features].skew().sort_values(ascending=False)

skewness = pd.DataFrame({'Skew':skew_feats})

skewness
color = plt.get_cmap('RdYlGn') 

color.set_bad('lightblue')



corrmat = data[corr_highly_correlated_features].corr()

f, ax = plt.subplots(figsize=(18, 12))

sns.heatmap(corrmat, vmax=.8, annot=True, cmap=color);
#Splitting data into training and validation data.

data_train, data_test = train_test_split(data, test_size=0.2, random_state=2698)
X_train = data_train[highly_correlated_features]

y_train = data_train.DEATH_EVENT

X_val = data_test[highly_correlated_features]

y_val = data_test.DEATH_EVENT
#Finding the optimum number of n_estimators

estimatorList = []

best_estimators = 0

best_performer = 0



for estimators in range(1,45):

    classifier = RandomForestClassifier(n_estimators = estimators, random_state=1, criterion='gini', max_features='auto')

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)

    estimatorList.append(accuracy_score(y_val,y_pred))

    

    if accuracy_score(y_val,y_pred) > best_performer:

        best_estimators = estimators

        best_performer = accuracy_score(y_val,y_pred)

        

print(f"Optimal n_estimators hyperparameter value: {best_estimators}")

print(f"Optimal accuracy: {best_performer}")



plt.plot(list(range(1,45)), estimatorList)

plt.show()
#Construct RandomForestClassifier supplied with auto-generated n_estimators value.

Model = RandomForestClassifier(n_estimators = best_estimators, criterion='gini', random_state=1)



#Fit training data

Model.fit(X_train, y_train)
#Generate predications for validation data.

predictions = Model.predict(X_val)



#Calculate mean average error.

train_mae = mean_absolute_error(y_val, predictions)
#Model accuracy summary



print(f"model train_mae: {train_mae}")

print(f"accuracy score: {accuracy_score(predictions, data_test.DEATH_EVENT)}")
#confusion matrix

plt.figure(figsize = (10,10))

cm = confusion_matrix(y_val, predictions)

sns.heatmap(cm,cmap= "Blues", linecolor = 'black', linewidth = 1, annot = True, fmt='', 

            xticklabels = ['True','False'], yticklabels = ['True','False'])

plt.xlabel("Predicted")

plt.ylabel("Actual")