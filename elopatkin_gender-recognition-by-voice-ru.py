import pandas as pd



df = pd.read_csv('../input/voicegender/voice.csv')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns



label = df['label'].map({ 'male': 0, 'female': 1 }, na_action='ignore')

corr_data = df.drop('label', axis=1).corrwith(label).abs().sort_values()



plt.figure(figsize=(12, 8))

sns.barplot(x=corr_data.values, y=corr_data.index)

plt.title('Correlation with gender')

plt.xlabel('correlation (abs)')

plt.show()
plt.figure(figsize=(10, 6))

sns.swarmplot(x=df['label'], y=df['meanfun'])

plt.title('meanfun distribution by gender')

plt.show()
sns.jointplot(x=df['IQR'], y=df['Q25'], kind='kde')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics



y = df['label'].map({ 'male': 0, 'female': 1 }, na_action='ignore')

X = df[['meanfun', 'IQR', 'Q25', 'sp.ent', 'sd', 'sfm', 'centroid', 'meanfreq', 'median']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)



print('Accuracy is', metrics.accuracy_score(y_test, y_pred))
y = df['label'].map({ 'male': 0, 'female': 1 }, na_action='ignore')

X = df[['IQR', 'Q25', 'sp.ent', 'sd', 'sfm', 'centroid', 'meanfreq', 'median']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)



print('Accuracy is', metrics.accuracy_score(y_test, y_pred))