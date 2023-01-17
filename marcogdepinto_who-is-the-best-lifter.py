import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
meets = pd.read_csv('../input/meets.csv')
openpl = pd.read_csv('../input/openpowerlifting.csv')
meets.head(10)
openpl.head(10)
merged = pd.merge(meets, openpl, on='MeetID', how='left')
merged.head(5)
merged.info()
merged.describe()
print(merged['Equipment'].value_counts())
def strapsInWraps(x):
    if x == 'Straps':
        return 'Wraps'
    return x
merged['Equipment'] = merged['Equipment'].apply(strapsInWraps)
print(merged['Equipment'].value_counts())
plt.figure(figsize=(10,7))
merged['Sex'].value_counts().plot(kind='bar')
plt.title('Gender division in the dataframe',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print('Percentage of Male lifters: {}%\n'.format(round(len(merged[merged['Sex']=='M'])/len(merged)*100),4))
print(merged['Sex'].value_counts())
g = sns.FacetGrid(merged,hue='Sex',size=6,aspect=2,legend_out=True)
g = g.map(plt.hist,'BodyweightKg',bins=50,alpha=.6)
plt.title('Bodyweight Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.legend(loc=1)
plt.show()
def age_calculate(x):
    if(x < 10.0):
        return "05-10"
    if(x >= 10.0 and x < 20.0):
        return "10-20"
    if(x >= 20.0 and x < 30.0):
        return "20-30"
    if(x >= 30.0 and x < 40.0):
        return "30-40"
    if(x >= 40.0 and x < 50.0):
        return "40-50"
    if(x >= 50.0 and x < 60.0):
        return "50-60"
    if(x >= 60.0 and x < 70.0):
        return "60-70"
    if(x >= 70.0 and x < 80.0):
        return "70-80"
    if(x >= 80.0 and x < 90.0):
        return "80-90"
    else:
        return "90-100"
    
merged['ageCategory'] = pd.DataFrame(merged.Age.apply(lambda x : age_calculate(x)))
firstPlace = merged[(merged.Place == '1')]
firstPlaceMale = firstPlace[(firstPlace.Sex == 'M')]
firstPlaceFemale = firstPlace[(firstPlace.Sex == 'F')]
firstPlaceMale.head(5)
firstPlaceFemale.head(5)
firstPlaceMale.isnull().any()
firstPlaceMale = firstPlaceMale[np.isfinite(firstPlaceMale['Age'])]
uniqueAgeValuesMale = firstPlaceMale.Age.unique() 
uniqueAgeValuesMale
firstPlaceMale[firstPlaceMale['Age'] == 9.5]
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(firstPlaceMale['ageCategory'], palette="muted")
plt.title('Distribution of Age for Male Athletes (winners)')
firstPlaceFemale = firstPlaceFemale[np.isfinite(firstPlaceFemale['Age'])]
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(firstPlaceFemale['ageCategory'], palette="Set1")
plt.title('Distribution of Age for Female Athletes (winners)')
plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestSquatKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Squat per Age Category')
firstPlace['BestSquatKg'] = firstPlace['BestSquatKg'].abs()
firstPlace['BestBenchKg'] = firstPlace['BestBenchKg'].abs()
firstPlace['BestDeadliftKg'] = firstPlace['BestDeadliftKg'].abs()
plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestSquatKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Squat per Age Category')

plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestBenchKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Bench per Age Category')

plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestDeadliftKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Deadlift per Age Category')
merged['Year'] = pd.DatetimeIndex(merged['Date']).year  
merged.head()
plt.figure(figsize=(20, 10))
sns.set(style="ticks", rc={"lines.linewidth": 5})
sns.lineplot('Year', 'Age', data=merged)
plt.title('Variation of Age for Athletes over time')
sns.set(style="darkgrid")
plt.figure(figsize=(30, 10))
sns.countplot(x='Year', data=merged, palette='Set2')
plt.title('Variation of the number of athletes over time')
merged.head(5)
merged['Place'].dtype
merged['Place'] = pd.to_numeric(merged.Place, errors='coerce').fillna(0, downcast='infer')
def is_Winner(x):
    if(x == 1):
        return 1
    else:
        return 0
    
merged['isWinner'] = pd.DataFrame(merged.Place.apply(lambda x : is_Winner(x)))
final_data = merged.drop(['MeetID', 'MeetPath', 'Federation', 
                          'MeetName', 'Date', 'MeetName', 
                          'Name', 'WeightClassKg', 'Division', 'Squat4Kg', 
                          'Bench4Kg', 'Deadlift4Kg', 'Place', 'ageCategory'], axis=1)
final_data.head(5)
catData = pd.get_dummies(final_data, columns=['Sex', 'MeetCountry', 'MeetState', 'MeetTown', 'Equipment'])
catData.head(5)
catData.dtypes
catData['Age'] = catData['Age'].fillna(catData['Age'].mean())
catData['BodyweightKg'] = catData['BodyweightKg'].fillna(catData['BodyweightKg'].mean())
catData['BestSquatKg'] = catData['BestSquatKg'].fillna(catData['BestSquatKg'].mean())
catData['BestBenchKg'] = catData['BestBenchKg'].fillna(catData['BestBenchKg'].mean())
catData['BestDeadliftKg'] = catData['BestDeadliftKg'].fillna(catData['BestDeadliftKg'].mean())
catData['TotalKg'] = catData['TotalKg'].fillna(catData['TotalKg'].mean())
catData['Wilks'] = catData['Wilks'].fillna(catData['Wilks'].mean())
catData['Year'] = catData['Year'].fillna(catData['Year'].mean())
from sklearn.model_selection import train_test_split
X = catData.drop('isWinner',axis=1)
y = catData['isWinner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
merged.head(5)