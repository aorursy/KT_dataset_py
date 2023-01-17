# importing basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # for visualization

# reading data

raw_data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print("First 5 rows of dataset: ")

raw_data.head(5)
raw_data.isna().sum().any()
raw_data.Class.value_counts()
from sklearn.utils import resample,shuffle

df_majority = raw_data[raw_data['Class']==0]

df_minority = raw_data[raw_data['Class']==1]

df_majority_downsampled = resample(df_majority,replace=False,n_samples=492,random_state = 123)

balanced_df = pd.concat([df_minority,df_majority_downsampled])

balanced_df = shuffle(balanced_df)

balanced_df.Class.value_counts()
sns.distplot(balanced_df.Amount,color='green');
sns.boxenplot(balanced_df.Time,palette='Set1');
balanced_df.describe()
from sklearn.preprocessing import StandardScaler



X = balanced_df.drop('Class',axis=1)

y = balanced_df.Class

scaled_X = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(scaled_X,y,test_size=0.3,shuffle=True,random_state=42)

x_train.shape,x_test.shape
# importing classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import f1_score,accuracy_score



classifiers = {

    'Logistic Regression' : LogisticRegression(),

    'Decision Tree' : DecisionTreeClassifier(),

    'Random Forest' : RandomForestClassifier(),

    'Support Vector Machines' : SVC(),

    'K-nearest Neighbors' : KNeighborsClassifier(),

    'XGBoost' : XGBClassifier()

}

results=pd.DataFrame(columns=['Accuracy in %','F1-score'])

for method,func in classifiers.items():

    func.fit(x_train,y_train)

    pred = func.predict(x_test)

    results.loc[method]= [100*np.round(accuracy_score(y_test,pred),decimals=4),

                         round(f1_score(y_test,pred),2)]

results