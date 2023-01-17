import os
from os.path import join

import pandas as pd
import numpy as np
import missingno as msno

import seaborn as sns
sns.set(color_codes=True)

input_dir = '../input'

data = pd.read_csv(join(input_dir,'wine.data.txt'), names =['Class','Alcohol','Malic Acid','Ash','Alcalinity of ash',
                                                               'Magnesium','Total phenols','Flavanoids',
                                                               'Nonflavanoid phenols','Proanthocyanins',
                                                               'Color intensity','Hue','OD280/OD315 of diluted wines',
                                                               'Proline'])
df_data = pd.DataFrame(data)
df_label = df_data['Class']
df_data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_data,df_label, test_size=0.33, random_state=42)
df_label.head()
class_1 = list()
class_2 = list()
class_3 = list()
for row in df_data.values :
    if row[0] == 1:
        class_1.append(row)
    elif row[0] == 2:
        class_2.append(row)
    else :
        class_3.append(row)
df_class_1 = pd.DataFrame(class_1, columns = df_data.columns)
df_class_2 = pd.DataFrame(class_2, columns = df_data.columns)
df_class_3 = pd.DataFrame(class_3, columns = df_data.columns)
del df_class_1['Class']
del df_class_2['Class']
del df_class_3['Class']
del df_data['Class']
msno.matrix(df_data)
means = list()
means.append(df_class_1.mean())
means.append(df_class_2.mean())
means.append(df_class_3.mean())
df_means = pd.DataFrame(means, columns = df_class_1.columns,
                        index = range(1,4))
df_means
medians = list()
medians.append(df_class_1.median())
medians.append(df_class_2.median())
medians.append(df_class_3.median())
df_medians = pd.DataFrame(medians, columns = df_class_1.columns,
                        index = range(1,4))
df_medians
sns.kdeplot(df_class_1['Alcohol'], label = 'class_1')
sns.kdeplot(df_class_2['Alcohol'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Alcohol'], bw=2, label='class_3').set_title('Alcohol Dist')
print('Mean')
print(df_means['Alcohol'].values)
print('Median')
print(df_medians['Alcohol'].values)
sns.kdeplot(df_class_1['Malic Acid'], label = 'class_1')
sns.kdeplot(df_class_2['Malic Acid'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Malic Acid'], bw=2, label='class_3').set_title('Malic Acid Dist')
print('Mean')
print(df_means['Malic Acid'].values)
print('Median')
print(df_medians['Malic Acid'].values)
sns.kdeplot(df_class_1['Ash'], label = 'class_1')
sns.kdeplot(df_class_2['Ash'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Ash'], bw=2, label='class_3').set_title('Ash Dist')
print('Mean')
print(df_means['Ash'].values)
print('Median')
print(df_medians['Ash'].values)
sns.kdeplot(df_class_1['Alcalinity of ash'], label = 'class_1')
sns.kdeplot(df_class_2['Alcalinity of ash'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Alcalinity of ash'], bw=2, label='class_3').set_title('Alcalinity of ash Dist')
print('Mean')
print(df_means['Alcalinity of ash'].values)
print('Median')
print(df_medians['Alcalinity of ash'].values)
sns.kdeplot(df_class_1['Magnesium'], label = 'class_1')
sns.kdeplot(df_class_2['Magnesium'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Magnesium'], bw=2, label='class_3').set_title('Magnesium Dist')
print('Mean')
print(df_means['Magnesium'].values)
print('Median')
print(df_medians['Magnesium'].values)
sns.kdeplot(df_class_1['Total phenols'], label = 'class_1')
sns.kdeplot(df_class_2['Total phenols'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Total phenols'], bw=2, label='class_3').set_title('Total phenols Dist')
print('Mean')
print(df_means['Total phenols'].values)
print('Median')
print(df_medians['Total phenols'].values)
sns.kdeplot(df_class_1['Flavanoids'], label = 'class_1')
sns.kdeplot(df_class_2['Flavanoids'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Flavanoids'], bw=2, label='class_3').set_title('Flavanoids Dist')
print('Mean')
print(df_means['Flavanoids'].values)
print('Median')
print(df_medians['Flavanoids'].values)
sns.kdeplot(df_class_1['Nonflavanoid phenols'], label = 'class_1')
sns.kdeplot(df_class_2['Nonflavanoid phenols'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Nonflavanoid phenols'], bw=2, label='class_3').set_title('Nonflavanoid phenols Dist')
print('Mean')
print(df_means['Nonflavanoid phenols'].values)
print('Median')
print(df_medians['Nonflavanoid phenols'].values)
sns.kdeplot(df_class_1['Proanthocyanins'], label = 'class_1')
sns.kdeplot(df_class_2['Proanthocyanins'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Proanthocyanins'], bw=2, label='class_3').set_title('Proanthocyanins Dist')
print('Mean')
print(df_means['Proanthocyanins'].values)
print('Median')
print(df_medians['Proanthocyanins'].values)
sns.kdeplot(df_class_1['Color intensity'], label = 'class_1')
sns.kdeplot(df_class_2['Color intensity'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Color intensity'], bw=2, label='class_3').set_title('Color intensity Dist')
print('Mean')
print(df_means['Color intensity'].values)
print('Median')
print(df_medians['Color intensity'].values)
sns.kdeplot(df_class_1['Hue'], label = 'class_1')
sns.kdeplot(df_class_2['Hue'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Hue'], bw=2, label='class_3').set_title('Hue Dist')
print('Mean')
print(df_means['Hue'].values)
print('Median')
print(df_medians['Hue'].values)
sns.kdeplot(df_class_1['OD280/OD315 of diluted wines'], label = 'class_1')
sns.kdeplot(df_class_2['OD280/OD315 of diluted wines'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['OD280/OD315 of diluted wines'], bw=2, label='class_3').set_title('OD280/OD315 of diluted wines Dist')
print('Mean')
print(df_means['OD280/OD315 of diluted wines'].values)
print('Median')
print(df_medians['OD280/OD315 of diluted wines'].values)
sns.kdeplot(df_class_1['Proline'], label = 'class_1')
sns.kdeplot(df_class_2['Proline'], bw=.2, label='class_2')
sns.kdeplot(df_class_3['Proline'], bw=2, label='class_3').set_title('Proline Dist')
print('Mean')
print(df_means['Proline'].values)
print('Median')
print(df_medians['Proline'].values)
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
clf1.score(X_test,y_test)
from sklearn.svm import SVC
clf2 = SVC(kernel='linear')
clf2.fit(X_train, y_train)
clf2.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(max_depth = 5)
clf3.fit(X_train, y_train)
clf3.score(X_test,y_test)
from sklearn.linear_model import RidgeClassifier
clf4 = RidgeClassifier()
clf4.fit(X_train, y_train)
clf4.score(X_test,y_test)
from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('lr', clf1),('svm',clf2),('RF', clf3),('Ridge',clf4)])
eclf.fit(X_train, y_train)
eclf.score(X_test,y_test)