# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing libraries and important stuff
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
#import mglearn #helper library
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_moons
import math

#def roundup(x):
#    return int(math.ceil(x / 10.0)) * 100

from pyzipcode import ZipCodeDatabase

%matplotlib inline

zcdb = ZipCodeDatabase()

from google.cloud import bigquery
#from bq_helper import BigQueryHelper


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
census_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="census_bureau_usa")
bq_assistant = BigQueryHelper("bigquery-public-data", "census_bureau_usa")
#bq_assistant.list_tables()
#bq_assistant.head("population_by_zip_2010", num_rows=3)

#bq_assistant.head("population_by_zip_2010")
#bq_assistant.head("population_by_zip_2000")

query2 = """SELECT
  zipcode,
  pop_2000,
  pop_2010,
  pop_chg,
  pop_pct_chg
FROM (
  SELECT
    r1.zipcode AS zipcode,
    r2.population AS pop_2000,
    r1.population AS pop_2010,
    r1.population - r2.population AS pop_chg,
    ROUND((r1.population - r2.population)/NULLIF(r2.population,0) * 100, 2) AS pop_pct_chg, 
    ABS((r1.population - r2.population)/NULLIF(r2.population,0)) AS abs_pct_chg
  FROM
    `bigquery-public-data.census_bureau_usa.population_by_zip_2010` AS r1
  INNER JOIN
    `bigquery-public-data.census_bureau_usa.population_by_zip_2000` AS r2
  ON
    r1.zipcode = r2.zipcode WHERE --following criteria selects total population without breaking down by age/gender
    r1.minimum_age IS NULL
    AND r2.minimum_age IS NULL
    AND r1.maximum_age IS NULL
    AND r2.maximum_age IS NULL
    AND r1.gender = ''
    AND r2.gender = ''
    AND r1.population <> r2.population 
    AND r1.population > 1000)
WHERE 
    pop_pct_chg > 55.0
    AND pop_2010 > 10000
ORDER BY
  abs_pct_chg ASC
        """
response2 = census_data.query_to_pandas_safe(query2)
response2.head(100000)
#response2.info()

response2.zipcode = response2.zipcode.str[:2]
#df.round({'A': 1, 'C': 2})
#response2.pop_2000 = response2.pop_2000.round();
#response2.round({'pop_2000': 2, 'pop_2010': 2})
#response2.pop_2000 = round(response2.pop_2000)
#response2.pop_2000 = round((response2.pop_2000), 1000)
#response2.pop_2010 = response2.pop_2010.str[:3]
#response2.pop_chg = response2.pop_chg.str[:3]

response2.pop_pct_chg = round(response2.pop_pct_chg)

print(response2)
X = response2.drop('zipcode',axis = 1)
#print(X)
y = response2['zipcode']
#print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print("Accuracy on decision tree training set: {:.3f}".format(dtree.score(X_train, y_train)))
print("Accuracy on decision tree test set: {:.3f}".format(dtree.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=2000)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
print("Accuracy on random forest training set: {:.3f}".format(rfc.score(X_train, y_train)))
print("Accuracy on random forest test set: {:.3f}".format(rfc.score(X_test, y_test)))

#CITATIONS (also mentioned in the report):
#A lot of the learning and code came from https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/ that I followed to be able to do this assignment.
#I think I learned a few things on the bootcamp that I wouldn't have and coded along the way once I learned a topic in relation to this assignment.
#I also found code online for the BigQuery query at https://www.kaggle.com/paultimothymooney/how-to-query-the-usa-census-dataset