import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
avocado = pd.read_csv('../input/avocado.csv')
avocado.shape
avocado.head()
avocado.columns
avocado.drop('Unnamed: 0',axis=1,inplace=True)
avocado.describe()
avocado.info()
def low_cardinality_cols(dataframe):
    low_card_cols = [cname for cname in dataframe.columns if (dataframe[cname].nunique()<55 and dataframe[cname].dtype=='object')]
    return (low_card_cols)
                                                              
low_cardinality_cols(avocado)
def cols_with_missing_values(dataframe):
    cols_missing_data = [cname for cname in dataframe.columns 
                        if dataframe[cname].isnull().any()]
    return (cols_missing_data)
cols_with_missing_values(avocado)
avocado['region'].unique() # we have a total column, we can delete those records
avocado[ avocado['region'] == 'TotalUS'].head()
avocado = avocado[ avocado['region'] != 'TotalUS']
# checking if the records are removed
avocado[ avocado['region'] == 'TotalUS']
# adding new columns
avocado['small Hass'] = avocado['4046']
avocado['large Hass'] = avocado['4225']
avocado['extra large Hass'] = avocado['4770']
avocado.columns
# removing the number columns
avocado.drop(['4046','4225','4770'],axis=1,inplace=True)
avocado.columns
# get the values for the region column
region_dummies =   pd.get_dummies(data=avocado['region'])
# similar for the year column
year_dummies =   pd.get_dummies(data=avocado['year'])
# join the dataframes on index
avocado =   avocado.join(other=region_dummies,on=region_dummies.index,how='inner')
avocado.drop('key_0',axis=1,inplace=True)
avocado  = avocado.join(other=year_dummies,on=year_dummies.index,how='inner')
# check the new shape
avocado.shape
# check the new columns
avocado.columns
# create the feature
X = avocado.drop(['key_0','Total Volume','Total Bags','Date', 'year','type','region'],axis=1)
y = avocado['type']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

