import numpy as np 

import pandas as pd 



df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
print(df.shape)
df.describe()
df[df['age'] > 90]
df['DEATH_EVENT'].value_counts(normalize=True)
import seaborn as sns



ax = sns.countplot(x="DEATH_EVENT", data=df)
df.isna().sum() 
corr = df.drop('DEATH_EVENT', axis=1).apply(lambda x: x.corr(df['DEATH_EVENT']))

corr
corr.apply(abs).sort_values(ascending=False)
X = df[['time','serum_creatinine','ejection_fraction', 'age']]

y = df['DEATH_EVENT'].values
X
from sklearn.model_selection import train_test_split

# rando_state value from @Nayan Sakhiya

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=2698)
print(x_train.shape,x_test.shape)
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score
xgb = XGBClassifier(n_estimators=300)

xgb.fit(x_train, y_train)



y_pred = xgb.predict(x_test)



print('F1 score: ',f1_score(y_test, y_pred))



print('Accuracy score: ',accuracy_score(y_test, y_pred))
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(xgb, x_test, y_test, cmap='Blues')