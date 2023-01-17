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
data = pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
data.info()
data.head()

import matplotlib.pyplot as plt



plt.matshow(data.corr())

plt.show()
print(data.corr())
print(data.describe())


from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
y_data = data["fraudulent"]
X_data = data.dropna(subset=["fraudulent"])
X_data_num = X_data.drop("job_id",axis=1)

X_data_num = X_data_num.drop("title",axis=1)

X_data_num = X_data_num.drop("location",axis=1)

X_data_num = X_data_num.drop("company_profile",axis=1)

X_data_num = X_data_num.drop("description",axis=1)

X_data_num = X_data_num.drop("requirements",axis=1)

X_data_num = X_data_num.drop("benefits",axis=1)

X_data_num = X_data_num.drop("employment_type",axis=1)

X_data_num = X_data_num.drop("required_experience",axis=1)

X_data_num = X_data_num.drop("required_education",axis=1)

X_data_num = X_data_num.drop("industry",axis=1)

X_data_num = X_data_num.drop("function",axis=1)

X_data_num = X_data_num.drop("department",axis=1)

X_data_num = X_data_num.drop("salary_range",axis=1)

X_data_num = X_data_num.drop("fraudulent",axis=1)
X_data_num


try:

    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

except ImportError:

    from sklearn.preprocessing import Imputer as SimpleImputer



imputer = SimpleImputer(strategy="most_frequent")
imputer.fit(X_data_num)
imputed_X_data_num = imputer.transform(X_data_num)
imputed_X_data_num
imputed_X_data_num.shape
X_data_cat = X_data.drop("job_id", axis=1)

X_data_cat = X_data_cat.drop("title",axis=1)

X_data_cat = X_data_cat.drop("location",axis=1)

X_data_cat = X_data_cat.drop("salary_range",axis=1)

X_data_cat = X_data_cat.drop("company_profile",axis=1 )

X_data_cat = X_data_cat.drop("description" ,axis=1 )

X_data_cat = X_data_cat.drop("requirements",axis=1 )

X_data_cat = X_data_cat.drop("benefits",axis=1 )

X_data_cat = X_data_cat.drop("telecommuting",axis=1 )

X_data_cat = X_data_cat.drop("has_company_logo",axis=1 )

X_data_cat = X_data_cat.drop("has_questions",axis=1)

X_data_cat = X_data_cat.drop("fraudulent",axis=1)
X_data_cat
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer( strategy='most_frequent')

imp_mean.fit(X_data_cat)

imputed_X_data_cat = imp_mean.transform(X_data_cat)



imputed_X_data_cat[0]

X_data_cat_encoded= []


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for i in range(17880):

    X_data_cat_encoded.append(encoder.fit_transform(imputed_X_data_cat[i]))

    

X_data_cat_encoded

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

X_data_cat_1hot = encoder.fit_transform(X_data_cat_encoded)

X_data_cat_1hot
X_data_cat_tr=X_data_cat_1hot.toarray()
X_data_cat_tr
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="most_frequent")),

        ('std_scaler', StandardScaler()),

    ])



X_data_num_tr = num_pipeline.fit_transform(X_data_num)
X_data_num_tr
X_data_prepared = np.append(X_data_num_tr,X_data_cat_tr,axis =1)
X_data_prepared.shape
X_data_num_tr.shape
X_data_cat_tr.shape
from sklearn.model_selection import train_test_split



X_train,X_test ,y_train, y_test = train_test_split(X_data_prepared, y_data,test_size=0.2,random_state=42 )

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



log_clf = LogisticRegression(solver="liblinear", random_state=42)

score = cross_val_score(log_clf, X_train, y_train, cv=3, verbose=3)

score.mean()



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

log_clf = LogisticRegression(solver="liblinear", random_state=42)

log_clf.fit(X_train, y_train)



y_pred = log_clf.predict(X_test)



print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
from sklearn.metrics import accuracy_score

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_pred)))