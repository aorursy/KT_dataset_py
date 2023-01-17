# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

%matplotlib inline



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.shape
#Check for the null value

df.isnull().sum()
bins = (2,6.5,8)

labels = ['bad','good']

df['quality'] = pd.cut(df['quality'],bins=bins,labels=labels)
def unistats(df):

    output_df = pd.DataFrame(columns=['Count','Missing','NUnique','Unique','Dtype', 'Numeric','Mode','Mean','Min','25%','Median','75%','Max','Std', 'Skew', 'Kurt'])





    for col in df:

        if pd.api.types.is_numeric_dtype(df[col]):

            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].unique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]),

                                  df[col].mode().values[0], df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75),

                                  df[col].max(), df[col].std(),df[col].skew(), df[col].kurt()]

        else:

            output_df.loc[col] = [df[col].count(),df[col].isnull().sum(),df[col].nunique(), df[col].unique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]),

                                  df[col].mode().values[0],'','','','','','','','','']

    return output_df.sort_values(by=['Numeric','Skew', 'NUnique'], ascending=False)
unistats(df)
def univaritePlot(df,col,vartype):

    if vartype==0:

        sns.set(style="darkgrid")

        fig, ax=plt.subplots(nrows =1,ncols=2,figsize=(20,8))

        ax[0].set_title(col.upper() + " DISTRIBUTION PLOT")

        sns.distplot(df[col],ax=ax[0])

        ax[1].set_title(col.upper() + " BOX PLOT")

        sns.boxplot(data =df, x=col,ax=ax[1],orient='v')

        plt.show()

    if vartype==1:

        fig, ax = plt.subplots()

        fig.set_size_inches(len(df[col].unique())+10 , 7)

        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index) 

        for p in ax.patches:

            percentage = '{:.1f}%'.format(100 * p.get_height()/len(df))

            x = p.get_x() + p.get_width() / 2 - 0.05

            y = p.get_y() + p.get_height()

            ax.annotate(percentage, (x, y), size = 12)
univaritePlot(df=df,col='quality',vartype=1)
univaritePlot(df=df,col='chlorides',vartype=0)
univaritePlot(df=df,col='density',vartype=0)
univaritePlot(df=df,col='residual sugar',vartype=0)
# Find the correlation between variables

df.corr()
#sns.pairplot(df,hue='quality')

#plt.show()
# Encode the class bad as 0 and good as a 1

df['quality'] = df['quality'].map({'bad':0, 'good':1})
df.head()
df['quality'].value_counts()
y = df['quality']

X = df.drop(['quality'], axis=1)
# Split the data into train and test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("X_train Shape : ", X_train.shape)

print("X_test Shape : ", X_test.shape)



y_train_imb = (y_train != 0).sum()/(y_train == 0).sum()

y_test_imb = (y_test != 0).sum()/(y_test == 0).sum()

print("Imbalance in Train Data : ", y_train_imb)

print("Imbalance in Test Data : ", y_test_imb)
# Balancing DataSet

from imblearn.over_sampling import SMOTE



sm = SMOTE()

X_train,y_train = sm.fit_sample(X_train,y_train)
print("X_train Shape", X_train.shape)

print("y_train Shape", y_train.shape)



imb = (y_train != 0).sum()/(y_train == 0).sum()

print("Imbalance in Train Data : ",imb)
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score

from imblearn.metrics import sensitivity_specificity_support

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold



lr = LogisticRegression()





lr.fit(X_train,y_train)

preds = lr.predict(X_test)


print("Accuracy Score:",accuracy_score(preds,y_test))

print("classification Report:\n",classification_report(preds,y_test))

print("confusion Matrix:\n",confusion_matrix(preds,y_test))
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, preds, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

print("AUC:    \t", round(roc_auc_score(y_test, preds),2))
svc= SVC()

svc.fit(X_train,y_train)
preds1= svc.predict(X_test)
print("Accuracy Score:",accuracy_score(preds1,y_test))

print("classification Report:\n",classification_report(preds1,y_test))

print("confusion Matrix:\n",confusion_matrix(preds1,y_test))
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, preds1, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

print("AUC:    \t", round(roc_auc_score(y_test, preds1),2))
# random forest - the class weight is used to handle class imbalance - it adjusts the cost function

forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9}, n_jobs = -1)



# hyperparameter space

params = {"criterion": ['gini', 'entropy'], "max_features": ['auto', 0.4]}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

model = GridSearchCV(estimator=forest, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)
# fit model

model.fit(X_train, y_train)
print("Best AUC: ", model.best_score_)

print("Best hyperparameters: ", model.best_params_)
# predict churn on test data

y_pred = model.predict(X_test)



# create onfusion matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy Score:",accuracy_score(y_pred,y_test))



print("classification Report:\n",classification_report(y_pred,y_test))

# check sensitivity and specificity

sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')



# check area under curve

y_pred_prob = model.predict_proba(X_test)[:, 1]

print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))