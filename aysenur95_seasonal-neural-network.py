# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# read data

training_features_data = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_features.csv",

                    sep=',')





test_features_data = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/test_set_features.csv",

                    sep=',')







training_set_labels = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_labels.csv",

                    sep=',')



#eliminate null values



#for float types

training_features_data=training_features_data.fillna(training_features_data.mean())



#for string types

training_features_data=training_features_data.fillna('out-of-category')
#check no missing values are left 

training_features_data.isna().sum()
#encoding categorical features (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(training_features_data)

training_features_data_arr=enc.transform(training_features_data)



col_names_list=training_features_data.columns

encoded_categorical_df=pd.DataFrame(training_features_data_arr, columns=col_names_list)
#normalization(make all values bet. 0-1)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(encoded_categorical_df)

normalized_arr=scaler.transform(encoded_categorical_df)



normalized_df=pd.DataFrame(normalized_arr, columns=col_names_list)
#check if data types are correct or not 

normalized_df.info()
#check types of test dataset

test_features_data.info()
#eliminate null values



#for float types

test_features_data=test_features_data.fillna(test_features_data.mean())



#for string types

test_features_data=test_features_data.fillna('out-of-category')
#check no missing values are left 

test_features_data.isna().sum()
#encoding categorical features  (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()

enc.fit(test_features_data)

test_features_data_arr=enc.transform(test_features_data)



col_names_list=test_features_data.columns

test_encoded_categorical_df=pd.DataFrame(test_features_data_arr, columns=col_names_list)
#check data types

test_encoded_categorical_df.info()
#normalization(bet. 0-1)



#using minmax scaler(look up)

test_normalized_arr=scaler.transform(test_encoded_categorical_df)

test_normalized_df=pd.DataFrame(test_normalized_arr, columns=col_names_list)
# split df to X and Y

y = training_set_labels.loc[:, 'seasonal_vaccine'].values

X = normalized_df
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split# split data into 80-20 for training set / test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)



# cross-validation with 5 splits

cv = StratifiedShuffleSplit(n_splits=5, random_state = 42)
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.model_selection import GridSearchCV
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
# NN with 1 layer*****BEST

nn_1 = MLPRegressor(tol=1e-5, hidden_layer_sizes=10, random_state=0, solver='adam', activation='relu', max_iter=1000, batch_size=2048)



nn_1.fit(X, y)



# prediction results

y_pred = nn_1.predict(test_normalized_df)



# NN with 1 layer

nn_1 = MLPRegressor(tol=1e-5, hidden_layer_sizes=10, random_state=0, solver='adam', activation='logistic', max_iter=1000, batch_size=512)



nn_1.fit(X, y)



# prediction results

y_pred_logistic_nn = nn_1.predict(test_normalized_df)

import numpy as np



np.sum(np.logical_or(np.array(y_pred_logistic_nn) > 1, np.array(y_pred_logistic_nn) < 0), axis=0)
y_pred_logistic_nn = 1/(1+np.exp(-y_pred_logistic_nn))

#pred sonuçlarını dosyaya yazdırma



df_pred_seasonal_vaccine=pd.DataFrame(y_pred_logistic_nn, columns=['seasonal_vaccine'])

df_pred_seasonal_vaccine["respondent_id"] = df_pred_seasonal_vaccine.index



df_pred_seasonal_vaccine=df_pred_seasonal_vaccine[['respondent_id', 'seasonal_vaccine']]



df_pred_seasonal_vaccine.to_csv('/kaggle/working/df_seasonal_nn_log.csv', columns=['respondent_id', 'seasonal_vaccine'], 

                            index=False, sep=',')
df_pred_seasonal_vaccine.head()
df_pred_h1n1 = pd.read_csv("../input/h1n1-nn-log/df_h1n1_nn_log_son.csv",

                    sep=',')



df_pred_h1n1.head()
df_final = df_pred_h1n1.merge(df_pred_seasonal_vaccine, on="respondent_id", how = 'inner')
df_final['respondent_id'] = df_final['respondent_id'].astype(int) + 26707
#pred sonuçlarını dosyaya yazdırma



#df_final=df[['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine' ]]



df_final.to_csv('/kaggle/working/df_nn_log.csv', columns=['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine' ], 

                            index=False, sep=',')
df_final.head()