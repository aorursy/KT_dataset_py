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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv(r'../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

dataset.head()#to read dataset
dataset.isnull().any()#checking for any missing or null values
f, axes = plt.subplots(5, 3, figsize=(24, 36), sharex=False, sharey=False)#time spent to learn and visualise data(learnt from another tutorial notebook)
sns.swarmplot(x="EducationField", y="MonthlyIncome", data=dataset, hue="Gender", ax=axes[0,0])
axes[0,0].set( title = 'Monthly income against Educational Field')

sns.pointplot(x="PerformanceRating", y="JobSatisfaction", data=dataset, hue="Gender", ax=axes[0,1])
axes[0,1].set( title = 'Job satisfaction against Performance Rating')

sns.barplot(x="NumCompaniesWorked", y="PerformanceRating", data=dataset, ax=axes[0,2])
axes[0,2].set( title = 'Number of companies worked against Performance rating')

sns.barplot(x="JobSatisfaction", y="EducationField", data=dataset, ax=axes[1,0])
axes[1,0].set( title = 'Educational Field against Job Satisfaction')

sns.barplot(x="YearsWithCurrManager", y="JobSatisfaction", data=dataset, ax=axes[1,1])
axes[1,1].set( title = 'Years with current Manager against Job Satisfaction')

sns.pointplot(x="JobSatisfaction", y="MonthlyRate", data=dataset, ax=axes[1,2])
axes[1,2].set( title = 'Job Satisfaction against Monthly rate')

sns.barplot(x="WorkLifeBalance", y="DistanceFromHome", data=dataset, ax=axes[2,0])
axes[2,0].set( title = 'Distance from home against Work life balance')

sns.pointplot(x="OverTime", y="WorkLifeBalance", hue="Gender", data=dataset, jitter=True, ax=axes[2,1])
axes[2,1].set( title = 'Work life balance against Overtime')

sns.pointplot(x="OverTime", y="RelationshipSatisfaction", hue="Gender", data=dataset, ax=axes[2,2])
axes[2,2].set( title = 'Overtime against Relationship satisfaction')

sns.pointplot(x="MaritalStatus", y="YearsInCurrentRole", hue="Gender", data=dataset, ax=axes[3,0])
axes[3,0].set( title = 'Marital Status against Years in current role')

sns.pointplot(x="Age", y="YearsSinceLastPromotion", hue="Gender", data=dataset, ax=axes[3,1])
axes[3,1].set( title = 'Age against Years since last promotion')

sns.pointplot(x="OverTime", y="PerformanceRating", hue="Gender", data=dataset, ax=axes[3,2])
axes[3,2].set( title = 'Performance Rating against Overtime')

sns.barplot(x="Gender", y="PerformanceRating", data=dataset, ax=axes[4,0])
axes[4,0].set( title = 'Performance Rating against Gender')

sns.barplot(x="Gender", y="JobSatisfaction", data=dataset, ax=axes[4,1])
axes[4,1].set( title = 'Job satisfaction against Gender')

sns.countplot(x="Attrition", data=dataset, ax=axes[4,2])
axes[4,2].set( title = 'Attrition distribution')
data = {'y_Actual':    [],
        'y_Predicted': []
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

if 'EmployeeNumber' in dataset:
    del dataset['EmployeeNumber']
    
if 'EmployeeCount' in dataset:
    del dataset['EmployeeCount']
    
if 'Over18' in dataset:
    del dataset['Over18']
    
if 'StandardHours' in dataset:
    del dataset['StandardHours']
    
dataset.describe(include='all')
dataset.keys()
features = dataset[dataset.columns.difference(['Attrition'])]
output = dataset.iloc[:, 1]

'''

'''

categorical = []
for col, value in features.iteritems():
    if value.dtype == 'object':
        categorical.append(col)


numerical = features.columns.difference(categorical)

print(categorical)
# get the categorical dataframe, and one hot encode it
features_categorical = features[categorical]
features_categorical = pd.get_dummies(features_categorical, drop_first=True)
features_categorical.head()

features_categorical.head()

features_train, features_test, attrition_train, attrition_test = train_test_split(features, output, test_size = 0.3, random_state = 0)


features_numerical = features[numerical]
features = pd.concat([features_numerical, features_categorical], axis=1)

features.head()



random_forest = RandomForestClassifier(n_estimators = 800, criterion = 'entropy', random_state = 0)
random_forest.fit(features_train, attrition_train)
attrition_pred = random_forest.predict(features_test)
print("Accuracy: " + str(accuracy_score(attrition_test, attrition_pred) * 100) + "%") 
def plot_feature_importances(importances, features):
    # get the importance rating of each feature and sort it
    indices = np.argsort(importances)

    # make a plot with the feature importance
    plt.figure(figsize=(12,14), dpi= 80, facecolor='w', edgecolor='k')
    plt.grid()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], height=0.8, color='mediumvioletred', align='center')
    plt.axvline(x=0.03)
    plt.yticks(range(len(indices)), list(features))
    plt.xlabel('Relative Importance')
    plt.show()

plot_feature_importances(random_forest.feature_importances_, features)