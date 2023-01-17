import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')
del df['Name']
df.head(3)

df.Survived.value_counts()
df.groupby(['Fare']).Survived.agg(['mean','count']).sort_values('count',ascending=False)
df.isna().sum()
#sns.heatmap(df.isnull(),yticklabels=False ,cbar=False)
sns.heatmap(df.corr())
df.corr()[['Survived']].sort_values('Survived',ascending=False)
df.Fare.apply(np.log1p).plot.hist(bins=50)
print(df.Fare)
df.plot.scatter(x='Survived' , y='Fare', alpha= 0.2)
sns.violinplot(x='Survived',y='Fare',data = df)
features_df = df.drop('Survived', axis=1 )
num_features = features_df.select_dtypes(np.number)
num_features.describe()
feature = num_features.Pclass
value =  feature[feature < feature.quantile(.95)]
value = (value - value.mean())/value.std()
value.plot.hist(bins= 20)
value_1 =  num_features.Age
value_1 = (value_1 - value_1.mean())/value_1.std()
value_1.plot.hist(bins=20)
cat_features = df.select_dtypes(['object'])
cat_features.sample(5)
pd.get_dummies(cat_features)
features = pd.concat([num_features,pd.get_dummies(cat_features)],axis=1)
target = df['Survived']
features= features.fillna(0)
from sklearn.model_selection import train_test_split
x_train,x_test , y_train,y_test = train_test_split(features, target, test_size =0.25)
x_train.sample(5)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

baseline =  DummyClassifier(strategy='most_frequent')
model = RandomForestClassifier()

baseline.fit(x_train,y_train)
model.fit(x_train,y_train)
baseline_pred = baseline.predict(x_test)
model_pred =  model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

print('baseline :',classification_report(y_test,baseline.predict(x_test)))
dtCM = confusion_matrix(y_test, baseline_pred)

fig, ax = plot_confusion_matrix(conf_mat=dtCM ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.show()


print('model :',classification_report(y_test,model.predict(x_test)))
dtCM = confusion_matrix(y_test, model_pred)

fig, ax = plot_confusion_matrix(conf_mat=dtCM ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.show()
