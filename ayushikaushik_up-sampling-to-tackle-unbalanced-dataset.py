#import basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')
raw_data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

# Check the data

raw_data.info()
raw_data.head()
sns.countplot(raw_data.quality);
data.corr()['quality'].sort_values()[:-1]
data = raw_data.copy()

plt.figure(figsize=(12,12))

sns.heatmap(data.corr(),annot=True);
def quality_trans(x):

    if x<6:

        return 0

    else:

        return 1

data.quality = data.quality.map(quality_trans)

sns.countplot(data.quality);
data.quality.value_counts()
from sklearn.utils import resample,shuffle

df_majority = data[data['quality']==1]

df_minority = data[data['quality']==0]

df_minority_upsampled = resample(df_minority,replace=True,n_samples=855,random_state = 123)

balanced_df = pd.concat([df_minority_upsampled,df_majority])

balanced_df = shuffle(balanced_df)

balanced_df.quality.value_counts()
balanced_df.describe()
sns.boxplot(balanced_df['residual sugar']);
len(balanced_df[balanced_df['residual sugar']>4])
sns.boxplot(balanced_df['volatile acidity']);
# standardization

from sklearn.preprocessing import StandardScaler

X = balanced_df.drop('quality',axis=1)

y = balanced_df.quality

scaled_X = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(scaled_X,y,test_size=0.3,shuffle=True,random_state=42)

x_train.shape,x_test.shape
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
#Now lets try to do some evaluation for random forest model using cross validation.

from sklearn.model_selection import cross_val_score

rfc_eval = cross_val_score(estimator = RandomForestClassifier(), X = x_train, y = y_train, cv = 10)

rfc_eval.mean()