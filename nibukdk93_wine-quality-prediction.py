# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
red_df=pd.read_csv('/kaggle/input/wine-quality-selection/winequality-red.csv')

#white_df=pd.read_csv('/kaggle/input/wine-quality-selection/winequality-white.csv')
red_df.describe()
red_df.info()
red_df.head(2)
import seaborn as sns

import matplotlib.pyplot as plt 





plt.rcParams["patch.force_edgecolor"] = True

sns.set_style('darkgrid')
wine_quality = {

    3:'Three',

    4:'Four',

    5:'Five',

    6:'Six',

    7:'Seven',

    8:'Eight'

}
red_df['quality']= red_df['quality'].replace(wine_quality)
red_df['quality'].unique()
sns.pairplot(red_df)
red_df.loc[:,'citric acid':'total sulfur dioxide'].describe()
plt.figure(figsize=(15,15))

sns.boxplot(data = red_df.iloc[:,2:])

plt.ylim(0,100)
sns.distplot(red_df['total sulfur dioxide'], bins=50)
#red_df[red_df['total sulfur dioxide']>100]['total sulfur dioxide'].count()
iq1 = red_df.quantile(0.25)

iq3 = red_df.quantile(0.75)

IQR  = iq3- iq1
IQR
red_df.skew()
print(red_df[['residual sugar', 'chlorides', 'free sulfur dioxide',

       'total sulfur dioxide','sulphates']].quantile(0.10))
print(red_df[['residual sugar', 'chlorides', 'free sulfur dioxide',

       'total sulfur dioxide','sulphates']].quantile(0.90))
#red_df[red_df['residual sugar'] <= 3.6]['residual sugar']
red_df['residual sugar'] = red_df['residual sugar'].apply(lambda x: 3.6 if (x >3.6) else x)

red_df['chlorides'] = red_df['chlorides'].apply(lambda x: 0.109 if (x >0.109) else x)

red_df['free sulfur dioxide'] = red_df['free sulfur dioxide'].apply(lambda x: 31.000 if (x >31.000) else x)

red_df['total sulfur dioxide'] = red_df['total sulfur dioxide'].apply(lambda x: 93.200 if (x >93.200) else x)

red_df['sulphates'] = red_df['sulphates'].apply(lambda x: 0.850 if (x >0.850) else x)

red_df.skew()
red_df.info()
corr_matrix = red_df.corr()
sns.heatmap(corr_matrix, cmap='magma',annot=True, lw=2, linecolor='white')
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
red_df.iloc[:,:11] = scaler.fit_transform(red_df.iloc[:,:11])
sns.countplot(red_df['quality'])

red_df['quality'].value_counts()
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X = red_df.iloc[:,:11]

y= red_df['quality']
X_res, y_res = smote.fit_resample(X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report: \n", classification_report(y_test, y_pred))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
params = {

    'n_neighbors' :[3,5,7,9,11,13,15,19],

    'weights':['uniform','distance']

}
grs_cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=10,verbose=2)
grs_cv.fit(X_res,y_res)
print(grs_cv.best_params_)

print(grs_cv.best_score_)
cls_2 = KNeighborsClassifier(n_neighbors=3,weights='distance')

cls_2.fit(X_train, y_train)



y_pred_2 = cls_2.predict(X_test)

print("Classification Report: \n", classification_report(y_test, y_pred_2))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_2))