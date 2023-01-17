# Ignore Warnings

import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
# Clear memory -  https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-reset

# Resets the namespace by removing all names defined by the user

# -f : force reset without asking for confirmation.

%reset -f
# Call data manipulation libraries

import pandas as pd

import numpy as np
# Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns
# Read the train Data File 5 rows × 142 columns

train = pd.read_csv("../input/train.csv")

train.head()
# Read the test Data File  5 rows × 142 columns

test = pd.read_csv("../input/test.csv")

test.head()
train.info()
test.info()
#https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough

train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'orange', 

                                                                             figsize = (8, 6),

                                                                            edgecolor = 'k', linewidth = 2);

plt.xlabel('Number of Unique Values'); plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');
from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



# Color mapping

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



# Iterate through the float columns

for i, col in enumerate(train.select_dtypes('float')):

    ax = plt.subplot(4, 2, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
train.select_dtypes('object').head()
mapping = {"yes": 1, "no": 0}



# Apply same operation to both train and test

for df in [train, test]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)



train[['dependency', 'edjefa', 'edjefe']].describe()
target_values = train['Target'].value_counts()

target_values = pd.DataFrame(target_values)

target_values['Household_type'] = target_values.index

mappy = {4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"}

target_values['Household_type'] = target_values.Household_type.map(mappy)

target_values
sns.set(style = 'whitegrid', font_scale=1.4)

fig = plt.subplots(figsize=(15, 8))

ax = sns.barplot(x = 'Household_type', y = 'Target', data = target_values, palette='Accent', ci = None).set_title('Distribution of Poverty in Households')

train['Target'].value_counts().plot(kind='pie',  autopct='%1.1f%%')
#sns.countplot(x="v2a1",data=train)

sns.set(style = 'whitegrid', font_scale=1.4)

fig = plt.subplots(figsize=(15, 8))



sns.countplot(x="rooms", hue= "Target", data=train, palette="Accent").set_title('# of Rooms in Households for Diff Proverty Class')
sns.set(style = 'whitegrid', font_scale=1.4)

fig = plt.subplots(figsize=(15, 8))



sns.countplot(x="r4h3", hue= "Target", data=train, palette="Accent").set_title('# of Males in Households for Diff Proverty Class')
sns.set(style = 'whitegrid', font_scale=1.4)

fig = plt.subplots(figsize=(15, 8))



sns.countplot(x="refrig", hue= "Target", data=train, palette="Accent").set_title('# of Refrigrator in Households for Diff Proverty Class')
# Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
# Number of missing in each column - Test Data

missing = pd.DataFrame(test.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(test)



missing.sort_values('percent', ascending = False).head(10)
train[['meaneduc', 'SQBmeaned']].describe()
#train

train['meaneduc'].fillna(train['meaneduc'].mean(), inplace = True)

train['SQBmeaned'].fillna(train['SQBmeaned'].mean(), inplace = True)

#the same for test

test['meaneduc'].fillna(test['meaneduc'].mean(), inplace = True)

test['SQBmeaned'].fillna(test['SQBmeaned'].mean(), inplace = True)

train['rez_esc'].fillna(0, inplace = True)

train['v18q1'].fillna(0, inplace = True)

train['v2a1'].fillna(0, inplace = True)
def extract_features(df):

    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']

    df['rent_to_rooms'] = df['v2a1']/df['rooms']

    df['rent_to_bedrooms'] = df['v2a1']/df['bedrooms']

    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household

    df['tamhog_to_bedrooms'] = df['tamhog']/df['bedrooms']

    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household

    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household

    df['r4t3_to_bedrooms'] = df['r4t3']/df['bedrooms']

    df['rent_to_r4t3'] = df['v2a1']/df['r4t3']

    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1'])

    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']

    df['hhsize_to_bedrooms'] = df['hhsize']/df['bedrooms']

    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']

    df['qmobilephone_to_r4t3'] = df['qmobilephone']/df['r4t3']

    df['qmobilephone_to_v18q1'] = df['qmobilephone']/df['v18q1']

    



extract_features(train)

extract_features(test)
train.shape,test.shape
# Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(20)
train['qmobilephone_to_v18q1'].fillna(0, inplace = True)





test['qmobilephone_to_v18q1'].fillna(0, inplace = True)

# Splitting data into dependent and independent variable

# X is the independent variables matrix

X = train.drop('Target', axis = 1)



# y is the dependent variable vector

y = train.Target
X.drop(['Id','idhogar'], inplace = True, axis=1)

X.drop(['qmobilephone_to_v18q1'], inplace = True, axis=1)

X.shape
X.describe()
# Scaling Features

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



X_ss = ss.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)

X_PCA = pca.fit_transform(X)
# split into train/test and resample the data

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, random_state=1)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

rf = rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

y_pred
print('    Accuracy Report: Random Forest Model\n', classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3, random_state=42)

dt = dt.fit(X_train, y_train)

y_pred1 = dt.predict(X_test)

y_pred1
print('    Accuracy Report: Decision Tree Model\n', classification_report(y_test, y_pred1))
from sklearn.ensemble import GradientBoostingClassifier as gbm



gbc = gbm()

gbc = gbc.fit(X_train, y_train)

y_pred2 = gbc.predict(X_test)

y_pred2
print('    Accuracy Report: Gradient Boost Model\n', classification_report(y_test, y_pred2))

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=4)

kn = kn.fit(X_train, y_train)
y_pred3 = kn.predict(X_test)

y_pred3
print(' Accuracy Report: K Neighbors Model\n', classification_report(y_test, y_pred3))
from xgboost.sklearn import XGBClassifier as XGB

xgb = XGB()

xgb = xgb.fit(X_train, y_train)
y_pred4 = xgb.predict(X_test)

y_pred4
print('Accuracy Report: XGB Model\n', classification_report(y_test, y_pred4))
import lightgbm as lgb

lightgbm = lgb.LGBMClassifier()

lightgbm = lightgbm.fit(X_train, y_train)
y_pred5 = lightgbm.predict(X_test)

y_pred5
print('Accuracy Report: Light GBM Model\n', classification_report(y_test, y_pred5))