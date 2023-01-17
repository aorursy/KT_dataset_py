# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# read data

training_features_data = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_features.csv",

                    sep=',')





training_set_labels = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_labels.csv",

                    sep=',')





test_features_data = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/test_set_features.csv",

                    sep=',')

test_features_data.shape
training_set_labels.shape
training_features_data=training_features_data.fillna(training_features_data.mean())





training_features_data=training_features_data.fillna('out-of-category')
#no missing values are left 

training_features_data.isna().sum()
#encoding categorical features  --> (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(training_features_data)

training_features_data_arr=enc.transform(training_features_data)
col_names_list=training_features_data.columns



encoded_categorical_df=pd.DataFrame(training_features_data_arr, columns=col_names_list)
#normalization(bet. 0-1)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(encoded_categorical_df)

normalized_arr=scaler.transform(encoded_categorical_df)
normalized_df=pd.DataFrame(normalized_arr, columns=col_names_list)
normalized_df.info()
test_features_data.info()
test_features_data=test_features_data.fillna(test_features_data.mean())





test_features_data=test_features_data.fillna('out-of-category')
#no missing values are left 

test_features_data.isna().sum()
test_features_data.describe()
#encoding categorical features  --> (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(test_features_data)

test_features_data_arr=enc.transform(test_features_data)
col_names_list=test_features_data.columns



test_encoded_categorical_df=pd.DataFrame(test_features_data_arr, columns=col_names_list)
test_encoded_categorical_df.info()
#normalization(bet. 0-1)



test_normalized_arr=scaler.transform(test_encoded_categorical_df)
test_normalized_df=pd.DataFrame(test_normalized_arr, columns=col_names_list)
test_normalized_df.describe()
#import sklearn methods 

from sklearn.metrics import roc_curve, classification_report, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
training_set_labels.head()
# split df to X and Y

y = training_set_labels.loc[:, 'seasonal_vaccine'].values

X = normalized_df

y
# display test scores and return result string and indexes of false samples

def display_test_scores(test, pred):

    str_out = ""

    str_out += ("TEST SCORES\n")

    str_out += ("\n")



    #print AUC score

    auc = roc_auc_score(test, pred)

    str_out += ("AUC: {:.4f}\n".format(auc))

    str_out += ("\n")

    

    false_indexes = np.where(test != pred)

    return str_out, false_indexes



from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(random_state=0, n_estimators=100)



rfr.fit(X, y)



# prediction results

y_pred = rfr.predict(test_normalized_df)



# print accuracy metrics

#results, false = display_test_scores(y, y_pred)

#print(results)

y_pred
#pred sonuçlarını dosyaya yazdırma



df_pred_h1n1=pd.DataFrame(y_pred, columns=['h1n1_vaccine'])

df_pred_h1n1["respondent_id"] = df_pred_h1n1.index
df_pred_h1n1
y
y_pred