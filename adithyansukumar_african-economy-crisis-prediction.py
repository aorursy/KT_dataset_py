import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
df.head()
df.info()
df.describe().transpose()
df.isnull().sum()
sns.countplot(df['banking_crisis'])
sns.scatterplot(x='year',y='systemic_crisis',data=df)
sns.scatterplot(x='domestic_debt_in_default',y='year',data=df)
sns.scatterplot(x='sovereign_external_debt_default',y='year',data=df)
sns.countplot(df['cc3'])
sns.heatmap(df.corr(),annot=True)

sns.jointplot(x='year',y='exch_usd',data=df)

sns.pairplot(df)
df['banking_crisis'].value_counts()
df1=df[df['banking_crisis']=='no_crisis'].head(100)
df2=df[df['banking_crisis']=='crisis'].head(94)
new_df=pd.concat([df1,df2]).sample(frac=1)
new_df.head()
new_df.drop(['cc3','country'],axis=1,inplace=True)
new_df['banking_crisis']=pd.get_dummies(new_df['banking_crisis'],drop_first=False)
new_df['banking_crisis']
from sklearn.preprocessing import StandardScaler
x=new_df.drop('banking_crisis',axis=1)
y=new_df['banking_crisis']
scaler=StandardScaler()
x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
predictions=rfc.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))
