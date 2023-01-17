import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





data = pd.read_csv('../input/bank-note-authentication-uci-data/BankNote_Authentication.csv')

data.head()
data.isnull().sum()
print(f"So, the data has no missing values.\n\n\nNo. of observations in our dataset is {data.shape[0]}")
display("Let's have a look on relationships between features of our data")

sns.pairplot(data,hue='class');
sns.countplot(x='class',data=data)

plt.title('Classes (Authentic 1 vs Fake 0)');
data['class'].value_counts()
from sklearn.utils import resample,shuffle

df_majority = data[data['class']==0]

df_minority = data[data['class']==1]

df_minority_upsampled = resample(df_minority,replace=True,n_samples=762,random_state = 123)

balanced_df = pd.concat([df_minority_upsampled,df_majority])

balanced_df = shuffle(balanced_df)

balanced_df['class'].value_counts()
#importing librarires needed for modelling

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
scaler = StandardScaler()

scaled_features = scaler.fit_transform(balanced_df.drop('class',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=balanced_df.columns[:-1])



X = df_feat

y = balanced_df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc_preds = rfc.predict(X_test)

print(classification_report(y_test,rfc_preds))

print(confusion_matrix(y_test,rfc_preds))
score=cross_val_score(rfc,X,y,cv=5)

(100*score.mean()).round(2)
my_pipeline= Pipeline([('scaler',StandardScaler()),('rfc',RandomForestClassifier())])

my_pipeline.fit(X_train,y_train)

pp_preds = my_pipeline.predict(X_test)

print(classification_report(y_test,pp_preds))

print(confusion_matrix(y_test,pp_preds))
score1=cross_val_score(my_pipeline,X,y,cv=5)

(100*score1.mean()).round(2)