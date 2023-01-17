from IPython.display import Image

Image(filename='../input/heartdises/heart.jpg', width="800", height='50')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from pandas_profiling import ProfileReport

from IPython.display import Image

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve



sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data.head()
data.isnull().sum()
data.dtypes
data.describe()
data.describe().T
profile = ProfileReport(data)

profile
data.corr()
#visualize the correlation

plt.figure(figsize=(15,10))

sns.heatmap(data.corr(), annot=True, cmap = 'Wistia')

plt.show()
plt.figure(figsize=(15,8))

num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]

sns.pairplot(data[num], kind='scatter', diag_kind='hist')

plt.show()
# plot histograms for each variable

data.hist(figsize = (15, 12))

plt.show()
# visualising the Age in the dataset

plt.subplots(figsize=(15,5))

data['age'].value_counts(normalize = True)

data['age'].value_counts(dropna = False).plot.bar(color = 'cyan')

plt.title('Visualizing the Age')

plt.xlabel('Age')

plt.ylabel('count')

plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(15,8))

plt.title('Heart Disease Frequency for Male and Female Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
plt.subplots(figsize=(15,5))

sns.distplot(data['age'], color = 'cyan')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
plt.subplots(figsize=(20,5))

plt.subplot(1, 4, 1)

sns.distplot(data['age'])



plt.subplot(1, 4, 2)

sns.distplot(data['sex'])



plt.subplot(1, 4, 3)

sns.distplot(data['cp'])



plt.subplot(1, 4, 4)

sns.distplot(data['trestbps'])



plt.show()
# visualising the number of male and female in the dataset

plt.subplots(figsize=(15,5))

data['sex'].value_counts(normalize = True)

data['sex'].value_counts(dropna = False).plot.bar(color = 'cyan')

plt.title('Comparison of Males and Females')

plt.xlabel('gender')

plt.ylabel('count')

plt.show()
female_count = len(data[data.sex == 0])

male_count = len(data[data.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((female_count / (len(data.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((male_count / (len(data.sex))*100)))
# data.groupby('target').mean()
# Prepare Data

df = data.groupby('cp').size()

# Make the plot with pandas

df.plot(kind='pie', subplots=True, figsize=(15, 8))

plt.title("Pie Chart of Vehicle Class - Bad")

plt.ylabel("")

plt.show()
plt.subplots(figsize=(15,5))

sns.countplot(x = 'cp', data = data,  hue = 'sex', palette = 'bright')

plt.show()
plt.subplots(figsize=(15,5))

sns.countplot(x = 'cp', data = data,  hue = 'target', palette = 'bright')

plt.show()
data.isnull().sum()
#### Visualizing the null values using missingo function

import missingno as msno

msno.matrix(data)
from IPython.display import Image

Image(filename='../input/standard-deviation/std.png', width="800", height='50')
Image(filename='../input/boxplot/box plot2.png', width="800", height='50')
Image(filename='../input/boxplot/box.png', width="800", height='50')
data['age'].describe()
sns.boxplot(x=data["age"])

plt.show()
data['trestbps'].describe()
sns.boxplot(x=data["trestbps"])

plt.show()
data['chol'].describe()
sns.boxplot(x=data["chol"])

plt.show()
data['thalach'].describe()
sns.boxplot(x=data["thalach"])

plt.show()
data['oldpeak'].describe()
sns.boxplot(x=data["oldpeak"])

plt.show()
data.head()
f, ax = plt.subplots(figsize=(13, 7))

ax = sns.scatterplot(x="age", y="trestbps", data=data)

plt.show()   #################################################  There is no correlation between age and trestbps variable.
f, ax = plt.subplots(figsize=(13, 7))

ax = sns.regplot(x="age", y="trestbps", data=data)

plt.show()    ###################################  Our Graph shows the Linear Regression is the not good fit for our data.
f, ax = plt.subplots(figsize=(13, 7))

ax = sns.regplot(x="age", y="chol", data=data)

plt.show()   #####################################   This Graph is a slighly positive correlation between age and chol variables
f, ax = plt.subplots(figsize=(13, 7))

ax = sns.regplot(x="chol", y="thalach", data=data)

plt.show()                     ######################   there is no correlation between chol and thalach variable
f, ax = plt.subplots(figsize=(13, 7))

ax = sns.scatterplot(x="chol", y = "thalach", data=data)

plt.show()
# let's change the names of the  columns for better understanding



data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
data['sex'][data['sex'] == 0] = 'female'

data['sex'][data['sex'] == 1] = 'male'



data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'

data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'

data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'

data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'



data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'

data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'

data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'



data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'

data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'



data['st_slope'][data['st_slope'] == 1] = 'upsloping'

data['st_slope'][data['st_slope'] == 2] = 'flat'

data['st_slope'][data['st_slope'] == 3] = 'downsloping'



data['thalassemia'][data['thalassemia'] == 1] = 'normal'

data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'

data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'
data['sex'] = data['sex'].astype('object')

data['chest_pain_type'] = data['chest_pain_type'].astype('object')

data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')

data['rest_ecg'] = data['rest_ecg'].astype('object')

data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')

data['st_slope'] = data['st_slope'].astype('object')

data['thalassemia'] = data['thalassemia'].astype('object')
data.dtypes
data.head()
data = pd.get_dummies(data, drop_first=True)
data.head()
##  Spiliting the dataset into dependent & Independent Variables



X = data.drop('target', axis = 1)



Y = data['target']
# splitting the sets into training and test sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# getting the shapes

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
# Random Forest Classifier



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = RandomForestClassifier(n_estimators = 50, max_depth = 5)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

y_pred_quant = model.predict_proba(x_test)[:, 1]

y_pred = model.predict(x_test)



# evaluating the model

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))
# classification report

cp = classification_report(y_test, y_pred)

print(cp)
# cofusion matrix

plt.subplots(figsize=(15,5))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.tree import export_graphviz



estimator = model.estimators_[1]

feature_names = [i for i in x_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values





export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)
from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'heart.png', '-Gdpi=50'])



from IPython.display import Image

Image(filename = 'heart.png')
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])



plt.rcParams['figure.figsize'] = (15, 8)

plt.title('ROC curve for diabetes classifier', fontweight = 30)

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()