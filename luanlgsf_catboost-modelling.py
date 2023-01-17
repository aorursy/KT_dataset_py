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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



import re



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

df = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv('../input/titanic/test.csv')



df.head()
df.shape
100.0 * df.isnull().sum() / len(df)
df.info()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer



class DataProcesser(BaseEstimator, TransformerMixin):

    def __init__(self, cols_drop = ["PassengerId", "Name", "Cabin", "Ticket"]):

        self.cols_drop = cols_drop

    

    def drop_columns(self, df):

        return df.drop(columns=self.cols_drop, axis="columns")

    

    def binarize(self, df):

        def bin_embarked(name:str):

            if name == "S":

                return 0

            elif name == "C":

                return 1

            elif name == "Q":

                return 2

            else:

                return -1





        df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "female" else 0)

        df["Embarked"] = df["Embarked"].apply(bin_embarked)

        

        return df

    

    def fillna(self, df, cols):

        for c in cols:

            df[c] = df[c].fillna(value=df[c].mean())

        return df



    def process_data(self, X, verbose=0):



        df = X.copy()

        

        r_common = re.compile("(Master\s+)|(Mr.?\s+)|(Miss\s+)|(Mrs\s+)|(Ms\s+)|(Mx\s+)|(M\s+)")

        r_formal = re.compile("(Sir\s+)|(Gentleman\s+)|(Sire\s+)|(Mistress\s+)|(Madam\s+)|(Ma'am\s+)|(Dame\s+)|(Lord\s+)|(Lady\s+)|(Esq\.\s+)|(Excellency\s+)|(Honour\s+)|(Honourable\s+)")

        r_academic = re.compile("(Dr\.\s+)|(Professor\s+)|(QC\s+)|(Counsel\s+)|(CI\s+)|(Eur\sIng\s+)|(Chancellor\s+)|(Principal\s+)|(Principal\s+)|(Dean\s+)|(Rector\s+)|(Executive\s+)")



        def honorific(name:str):

            if r_common.search(name):

                return 1

            if r_formal.search(name):

                return 2

            if r_academic.search(name):

                return 3

            else:

                return 0

            

        df["honorific"] = df["Name"].apply(honorific)

        df = self.binarize(df)

        df = self.fillna(df, cols=["Embarked", "Age"])

        df = self.drop_columns(df)

        

        return df





    def fit(self, X, y=None, verbose=0):

        return self



    def transform(self, X, y=None, verbose=0):

        """transform data, processing each row on dataframe X. y is not used and is there just for compatibility."""

        return self.process_data(X, verbose=verbose)
processer = DataProcesser()
Y = df["Survived"].values

X = df.drop(columns="Survived")



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)



print("ratio: ", len(y_test) / (len(y_test) + len(y_train)))
from sklearn.pipeline import Pipeline

from xgboost import XGBRFClassifier, plot_importance

from lightgbm import LGBMClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score





cate_features_names = ["Pclass", "Sex", "Embarked", "honorific"]



clf_list = [("RF", RandomForestClassifier()),

            ("XGBoost", XGBRFClassifier()),

            ("LGBMClassifier", LGBMClassifier()),

           ]



pipelines = []

for clf in clf_list:



    pipelines.append(

        Pipeline([

            ("processer", processer),

            ("clf", clf[1])



        ])

    )

    

for i, pipeline in enumerate(pipelines):

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"accuracy for {clf_list[i][0]}: ", accuracy_score(y_test, y_pred))
X_proc_train = processer.fit_transform(X_train, y_train)

100.0 * X_proc_train.isnull().sum() / len(X_proc_train)
from catboost import CatBoostClassifier, Pool



pipeline =  Pipeline([

            ("processer", processer),

#             ("imputer", imputer)



        ])
X_proc_train.head()
clf = CatBoostClassifier()



clf.fit(X_proc_train, 

        y_train,

        plot=True)
y_pred = clf.predict(pipeline.fit_transform(X_test))

print(f"accuracy: ", accuracy_score(y_test, y_pred))
import shap

shap.initjs()
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Pool(X_proc_train, y_train))
shap.summary_plot(shap_values, X_proc_train, plot_type="bar")
import seaborn as sns

sns.barplot(x="Pclass", y="Survived", data=df)
# summarize the effects of all the features

shap.summary_plot(shap_values, X_proc_train)
from sklearn.model_selection import cross_val_score



clf = CatBoostClassifier()

print(cross_val_score(clf, X_proc_train, y_train, cv=3))
cats = ["Sex", "Pclass", "Embarked", "honorific"]



clf = CatBoostClassifier()



clf.fit(X_proc_train, y_train, cat_features=cats)
y_pred = clf.predict(pipeline.fit_transform(X_test))

print(f"accuracy: ", accuracy_score(y_test, y_pred))
clf = CatBoostClassifier(cat_features=cats, verbose=0)

print(cross_val_score(clf, X_proc_train, y_train, cv=3))
cats = ["Sex", "Pclass", "Embarked", "honorific"]



clf = CatBoostClassifier(verbose=0)



clf.fit(X_proc_train, y_train, cat_features=cats)

print(f"accuracy: ", accuracy_score(y_test, y_pred))
print(df_test.shape)

df_test.head()
X_val_proc = processer.fit_transform(df_test)

preds = clf.predict(X_val_proc)

preds[:10]
print(df_test.shape)

print(X_val_proc.shape)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':preds})



#Visualize the first 5 rows

submission.head(20)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'titanic_predictions_2.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)