# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows",None)

pd.set_option("display.max_columns",None)
dfx = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
dfx.shape
df = dfx.drop_duplicates()

df.shape
df.head()
df.isnull().sum()
df.info()
df["Churn"] = df["Churn"].replace(("Yes","No"),(1,0))
target = df["Churn"]

import seaborn as sns

sns.countplot(target)
print((target[target == 1].value_counts()))

print((target[target == 0].value_counts()))
df["customerID"].nunique()
df = df.drop(["customerID","Churn"],axis=1)
df_num = [col for col in df.columns if df[col].dtype != 'object']

df_num = df[df_num]
df_num.head()
df_cat = df.drop(df_num.columns,axis=1)
df_cat.head()
for col in df_cat.columns:

    if col != 'TotalCharges':

        print(col, ":", df_cat[col].value_counts())

        print("\n")
df[df["TotalCharges"]== " "]
df_num["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df_cat = df_cat.drop("TotalCharges",axis=1)
df_num.info()
#df_num["TotalCharges"] = df_num["TotalCharges"].fillna(df_num["TotalCharges"].median())

from sklearn.impute import KNNImputer



imputer = KNNImputer(n_neighbors=3)

imputer_array = imputer.fit_transform(df_num)

df_num = pd.DataFrame(imputer_array, columns  = df_num.columns)
df_num.head()
def tenure_bin(df_num) :

    if df_num["tenure"] <= 12 :

        return "Tenure_0-12"

    elif df_num["tenure"] > 12 and df_num["tenure"] <= 24:

        return "Tenure_12-24"

    elif df_num["tenure"] > 24 and df_num["tenure"] <= 36 :

        return "Tenure_24-36"

    elif df_num["tenure"] > 36 and df_num["tenure"] <= 48 :

        return "Tenure_36-48"

    elif df_num["tenure"] > 48 and df_num["tenure"] <= 60 :

        return "Tenure_48-60"

    elif df_num["tenure"] > 60 :

        return "Tenure_gt_60"

df_num["tenure_bin"] = df_num.apply(lambda df_num:tenure_bin(df_num),

                                      axis = 1)
df_num["y"] = target
j_df = pd.DataFrame()



j_df['yes'] = df_num[df_num['y'] == 1]['tenure_bin'].value_counts()

j_df['no'] = df_num[df_num['y'] == 0]['tenure_bin'].value_counts()



j_df.plot.bar(title = 'Job and deposit')
df_num["tenure_bin"] = df_num["tenure_bin"].replace(["Tenure_0-12","Tenure_12-24",

                                                     "Tenure_24-36","Tenure_36-48","Tenure_48-60","Tenure_gt_60"],[5,4,3,2,1,0])
del df_num["tenure"]

del df_num["y"]
df_num.head()
sns.distplot(df_num["TotalCharges"])
sns.distplot(df_num["MonthlyCharges"])
#sns.pairplot(df,vars = ['tenure','MonthlyCharges','TotalCharges'], hue="y")
df_cat = df_cat.replace(["No internet service","No phone service"],'No')
label=[]

for col in df_cat.columns:

    if df_cat[col].nunique() > 2:

        label.append(col)
label
df_dum = df_cat.drop(label,axis=1)

df_dum = df_dum.replace(["Yes","No"],[1,0])

df_dum = df_dum.replace(["Female","Male"],[0,1])

df_dum.head()
df_la = df_cat[label]

df_la.head()
df_la["InternetService"] = df_la["InternetService"].replace(["Fiber optic","DSL","No"],[2,1,0])

df_la["Contract"] = df_la["Contract"].replace(["Month-to-month","One year","Two year"],[2,1,0])

df_la["PaymentMethod"] = df_la["PaymentMethod"].replace(["Electronic check",

                                                       "Mailed check","Bank transfer (automatic)",

                                                      "Credit card (automatic)"],[3,2,1,0])

df_la.head()
# Label encoder order is alphabetical

#from sklearn.preprocessing import LabelEncoder

#labelencoder_X = LabelEncoder()

#df_l = pd.DataFrame()

#for var in df_cat.columns:

 #   df_l[var]= labelencoder_X.fit_transform(df_cat[var]) 
final = pd.concat([df_num,df_la,df_dum],axis=1)
print(final.shape,final.columns)
final.head()
#del final["Partner"]

#del final["Dependents"]

#del final["OnlineSecurity"]

#del final["OnlineBackup"]

#del final["DeviceProtection"]
final.head()
#final["churn1"] = target
#plt.figure(figsize=(15,8))

#final.corr()['churn1'].sort_values(ascending = False).plot(kind='bar')
print(final.shape,len(target))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final, target, test_size = 0.30, random_state = 101)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import KFold,GridSearchCV,StratifiedKFold,RandomizedSearchCV

from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import precision_score,recall_score,f1_score

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
def metrics(y,yhat):

    print(f"Precision Score: {precision_score(y,yhat) * 100:.2f}%")

    print(f"Recall Score: {recall_score(y,yhat) * 100:.2f}%")

    print(f"F1 score: {f1_score(y,yhat) * 100:.2f}%")
def plot_confusion(actual,pred):

    conf_matrix = confusion_matrix(actual, pred)

    f, ax = plt.subplots(figsize=(4, 4))

    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)

    plt.title("Confusion Matrix", fontsize=20)

    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)

    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(['0', '1'], fontsize=16, rotation=360)

    ax.set_yticklabels(['0', '1'], fontsize=16, rotation=360)

    plt.show()
lr = LogisticRegression(class_weight='balanced')

lr.fit(X_train, y_train)

lrprd = lr.predict(X_test)

print(classification_report(y_test,lrprd))

metrics(y_test,lrprd)

plot_confusion(y_test,lrprd)
#data['Churn'] = data['Churn'].map(lambda s :1  if s =='Yes' else 0)
#learning_rate=0.032,max_depth=4,n_estimators=100,

lgb = LGBMClassifier(class_weight={0:0.21,1:0.79},learning_rate=0.13099999999999,max_depth=18,n_estimators=250)

lgb.fit(X_train, y_train)

lgbprd = lgb.predict(X_test)

print(classification_report(y_test,lgbprd))

metrics(y_test,lgbprd)

plot_confusion(y_test,lgbprd)
plt.figure(figsize=(10,10))

sns.barplot(y= final.columns,x = lgb.feature_importances_)
LG = LGBMClassifier(class_weight={0:0.21,1:0.79},learning_rate=0.13099999999999,max_depth=18,n_estimators=250)

LG.fit(final, target)

prd = LG.predict(final)

print(classification_report(target,prd))

metrics(target,prd)

plot_confusion(target,prd)