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

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline
df=pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")
df.head()
df.info()
df.shape
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print('Workclass, ? numbers: ',df['workclass'][df['workclass']=='?'].count())

print('Percentage: {0:.2f}%'.format(df['workclass'][df['workclass']=='?'].count()/(48842)*100))
print('Occupation, ? numbers: ',df['occupation'][df['occupation']=='?'].count())

print('Percentage: {0:.2f}%'.format(df['occupation'][df['occupation']=='?'].count()/(48842)*100))
print('Native Country, ? numbers: ',df['native-country'][df['native-country']=='?'].count())

print('Percentage: {0:.2f}%'.format(df['native-country'][df['native-country']=='?'].count()/(48842)*100))
import statistics 

from statistics import mode 

  

def most_common(List): 

    return(mode(List))
print(most_common(df['workclass']))

print(most_common(df['occupation']))

print(most_common(df['native-country']))
df['workclass']=df['workclass'].replace('?','Private')

df['occupation']=df['occupation'].replace('?','Prof-specialty')

df['native-country']=df['native-country'].replace('?','United-States')
df.head()
cols =['workclass', 'education','marital-status', 'occupation',

               'relationship','race', 'gender', 'native-country', 'income'] 

for i in cols:

    print(i,':')

    print('')

    print(df[i].value_counts())

    print('')
#education category

df.education=df.education.replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],'left')

df.education=df.education.replace('HS-grad','school')

df.education=df.education.replace(['Assoc-voc','Assoc-acdm','Prof-school','Some-college'],'higher')

df.education=df.education.replace('Bachelors','undergrad')

df.education=df.education.replace('Masters','grad')

df.education=df.education.replace('Doctorate','doc')
#marital status

df['marital-status']=df['marital-status'].replace(['Married-civ-spouse','Married-AF-spouse'],'married')

df['marital-status']=df['marital-status'].replace(['Never-married','Divorced','Separated','Widowed',

                                                   'Married-spouse-absent'], 'not-married')
#income

df.income=df.income.replace('<=50K', 0)

df.income=df.income.replace('>50K', 1)
df.head()
df.describe()
plt.figure(figsize=(20,7))

sns.countplot(x='age',data=df)

sns.despine()

plt.title('Age Distribution')
px.histogram(df,x='age',color='gender',nbins=40)
px.pie(df,values='educational-num',names='education',title='Percentage of Education',

      color_discrete_sequence=px.colors.qualitative.G10)
plt.figure(figsize=(15,7))

sns.countplot(x='workclass',data=df)

sns.despine()
plt.figure(figsize=(20,7))

sns.countplot(x='occupation',data=df)

sns.despine()
# fig=px.bar(df,x='occupation')

# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

# fig.show()
# fig=px.bar(df,x='relationship')

# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

# fig.show()
# fig=px.bar(df,x='race')

# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

# fig.show()
sns.countplot(y="gender",data=df)
# fig=px.bar(df,x='native-country')

# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

# fig.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
df1=df.copy()

df1=df1.apply(LabelEncoder().fit_transform)

df1.head()
std_sclr=StandardScaler().fit(df1.drop('income',axis=1))
X=std_sclr.transform(df1.drop('income',axis=1))

y=df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
forest=RandomForestClassifier(n_estimators=5,random_state=0)
forest.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
importance=forest.feature_importances_
for i,v in enumerate(importance):

    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.figure(figsize=(10,5))

plt.bar([x for x in range(len(importance))], importance)

plt.title('Feature Importance')

plt.show()
gbrt=GradientBoostingClassifier(max_depth=1,learning_rate=1,random_state=0)

gbrt.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
importance=gbrt.feature_importances_
for i,v in enumerate(importance):

    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.figure(figsize=(10,5))

plt.bar([x for x in range(len(importance))], importance)

plt.title('Gradient Boosting Classifier Feature Importances')

plt.show()