# Perform necessary imports

import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, MaxPooling2D, CuDNNLSTM

from tensorflow.keras.utils import to_categorical



import numpy as np

import pandas as pd



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif



from tsfresh import extract_features

from tsfresh.utilities.dataframe_functions import impute



import pickle

from functools import reduce
import pandas as pd

import numpy as np



from tsfresh import extract_features

from tsfresh.feature_extraction import ComprehensiveFCParameters

from tsfresh.feature_extraction import MinimalFCParameters

from tsfresh.utilities.dataframe_functions import impute



import pickle



if __name__ == '__main__':

    # read train and test files

    df_train=pd.read_csv('../input/X_train.csv')

    df_train.drop(['row_id'],axis=1,inplace=True)

    df_train=extract_features(df_train,column_id = "series_id", chunksize=7, default_fc_parameters=None, column_sort = "measurement_number", impute_function=impute, n_jobs=7)

    df_train.to_csv('train_feature.csv',index=False)

    del df_train



    df_test=pd.read_csv('../input/X_test.csv')

    df_test.drop(['row_id'],axis=1,inplace=True)

    df_test=extract_features(df_test,column_id = "series_id", chunksize=7, default_fc_parameters=None, column_sort = "measurement_number", impute_function=impute, n_jobs=7)

    df_test.to_csv('test_feature.csv',index=False)

    del df_test




# # Function to read files and store in dataframe

# def read_files():



# 	global df_X

# 	global df_y

# 	# Read the dataset and view

# 	df_X=pd.read_csv('train_feature.csv')

# 	df_y=pd.read_csv('../input/y_train.csv',index_col='series_id')



# 	# Drop unecessary columns

# 	print(len(df_X.columns))

# 	# df_X.drop(['row_id','measurement_number'],axis=1,inplace=True)



# 	#Encode the labels

# 	le=LabelEncoder()

# 	le.fit(df_y['surface'])

# 	pickle.dump(le, open("label_encoder.pickle", "wb"))

# 	df_y['surface']=le.transform(df_y['surface'])





# # Feature engg using tsfresh

# def feature_engg(df):



# 	return df



# # Function to predict

# def predict(x,model_file):



# 	file = open(model_file,'rb')

# 	model = pickle.load(file)

# 	file.close()



# 	y_pred = model.predict(x)

# 	return y_pred





# # Function to write submission to file

# def test_file():



# 	# Read the test file

# 	df_X_pred=pd.read_csv('test_feature.csv')



# 	# Perform feature engg

# 	df_X_pred=feature_engg(df_X_pred)



# 	# Normalise

# 	df_X_pred=minmax.transform(df_X_pred)



# 	# Feature selection

# 	df_X_pred=selector.transform(df_X_pred)



# 	y_pred=predict(df_X_pred,'randomforest.pickle')





# 	file = open('label_encoder.pickle','rb')

# 	le= pickle.load(file)

# 	file.close()



# 	predictions=le.inverse_transform(y_pred)



# 	# Save to file

# 	submission_df=pd.DataFrame(predictions)

# 	submission_df.to_csv('submission.csv',index_label=['series_id'],header=['surface'])

# 	print('Submission written to file')









# read_files()



# df_X_final=feature_engg(df_X)

# print(df_X_final.head())



# # Split the dataset

# df_X_train,df_X_test,df_y_train,df_y_test=train_test_split(df_X_final,df_y['surface'],test_size=0.4)



# # Normalize the X dataset

# minmax=MinMaxScaler()

# df_X_train=minmax.fit_transform(df_X_train)



# # Perform feature selection

# selector=SelectKBest(mutual_info_classif,k='all')

# selector.fit(df_X_train,df_y_train)

# df_X_train=selector.transform(df_X_train)



# # Perform necessary operations on test set



# # Normalise

# df_X_test=minmax.transform(df_X_test)



# # Perform feature selection

# df_X_test=selector.transform(df_X_test)





# #Training

# classifier = RandomForestClassifier(n_estimators = 1000, max_depth=20, random_state=42, verbose=1, n_jobs=-1)

# classifier.fit(df_X_train,df_y_train)



# f = open('randomforest.pickle','wb')

# pickle.dump(classifier,f)

# f.close()

# print('Training done')



# # Check accuracy on train set

# y_pred=predict(df_X_train,'randomforest.pickle')

# print(classification_report(df_y_train, y_pred))

# print(accuracy_score(df_y_train,y_pred)*100)



# # Check accuracy on test set

# y_pred=predict(df_X_test,'randomforest.pickle')

# print(classification_report(df_y_test, y_pred))

# print(accuracy_score(df_y_test,y_pred)*100)



# test_file()