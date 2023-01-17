# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "svm"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
kiva = pd.read_csv("C:/kiva_loans.csv")
mpi = pd.read_csv("C:/MPI_subnational.csv")
kiva.info()
kiva.describe()
mpi.info()
mpi.rename(columns = {'Sub-national region':'region'}, inplace = True)
combo = pd.merge(kiva,mpi)
combo.info()
#combo.to_csv("combo.csv", sep=',')
corr_matrix = combo.corr()
corr_matrix["MPI Regional"].sort_values(ascending=False)
combo["funded_amount"].equals(combo["loan_amount"])#Funded_amount turns out to not be the same as loan_amount
combo = combo.drop(["date", "posted_time", "disbursed_time", "term_in_months", "use", "country_code", "partner_id", "funded_time",
            "currency", "lender_count", "tags", "date", "ISO country code", "Country", "id", ], axis = 1)#"Intensity of deprivation Regional"
data = pd.get_dummies(combo, columns = ["activity", "sector", "country", "region", "borrower_genders", "repayment_interval", "World region"])
data.info()
data.describe()
mpi_info = data['MPI Regional']
mpi_info.value_counts()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data["MPI Regional"])
le.classes_
data["MPI Regional"] = le.transform(data["MPI Regional"])
data["MPI Regional"]
data['funded_amount'] = data["funded_amount"].astype(int)
data['loan_amount'] = data['loan_amount'].astype(int)
data['Headcount Ratio Regional'] = data['Headcount Ratio Regional'].astype(int)
sample_incomplete_rows = data[data.isnull().any(axis=1)]
sample_incomplete_rows
data = data.dropna()
sample_incomplete_rows = data[data.isnull().any(axis=1)]
sample_incomplete_rows
data.info()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=.5, random_state = 42)
train_x = train_set.drop("MPI Regional", axis = 1)
train_y = train_set["MPI Regional"]
test_x = test_set.drop("MPI Regional", axis = 1)
test_y = test_set["MPI Regional"]
train_y.shape
train_y.value_counts()
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_features=2, max_leaf_nodes=16, n_jobs=-1, class_weight = 'balanced', random_state = 42)
rnd_clf.fit(train_x,train_y)
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(rnd_clf, train_x, train_y, cv=3)
from sklearn.metrics import precision_score, recall_score
precision_score(train_y, y_train_pred, average = 'weighted')
recall_score(train_y,y_train_pred, average = 'weighted')
y_test_pred = cross_val_predict(rnd_clf, test_x, test_y, cv=3)
precision_score(test_y, y_test_pred, average = 'weighted')
recall_score(test_y,y_test_pred, average = 'weighted')
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(train_x,train_y)
feature_importances = pd.DataFrame(rnd_clf.feature_importances_,
                                   index = train_x.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print (feature_importances)