import warnings



import numpy as np

import pandas as pd



import xgboost as xgb



import category_encoders as ce

warnings.filterwarnings('ignore')



from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split



from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize, Normalizer,MinMaxScaler



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
df = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
df.head()
df_test.head()
#Creating the object for onehot encoding

onehot_encoder = OneHotEncoder()



#function to return the onehot encoded dataframe

def one_hot_encodingg(temp_col):

    integer_encoded = temp_col.reshape(len(temp_col), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    Mc=onehot_encoded.tocoo()

    mc = Mc.toarray()

    return mc



#function for renaming the columns

def rename_col(new_df,string):    

    new_df.columns = [string+str(i) for i in range(len(new_df.columns))]

    return new_df

    

#converting categorical dataframe to one hot encoded dataframes    

def convert_to_one_hot(temp_col,string):

    temp_col = np.array(list(temp_col))

    temp_col = temp_col.reshape(-1,1)

    new_df = one_hot_encodingg(temp_col)

    new_df = pd.DataFrame(new_df)

    new_df = rename_col(new_df,string)

    return new_df
#concating the two dataframes

def concatenate_dataframes(df1,df2):

    df1 = df1.reset_index()

    df2 = df2.reset_index()

    df_c =  pd.concat([df1,df2] , axis = 1)

    df_c = df_c.drop(columns = ['index'])

    return df_c



#concating onehot encoded dataframes with the original dataframe

def encode_concate(df, df_col, string):

    df_one_hot = convert_to_one_hot(df_col, string)

    df_concat = concatenate_dataframes(df,df_one_hot)

    df_concat = df_concat.drop(columns = [string])

    return df_concat 



#extracting class from the original dataframe and dropping that column

def Y_train_extract(df):

    y_train = df.Class

    df1 = df.drop(columns = 'Class')

    return y_train,df1



#doing explorating data analysis

def eda(df):

    #encoding into onehot of the columns 'V2', 'V3', 'V4', 'V9'

    df = encode_concate(df,df.V2,'V2')

    df = encode_concate(df,df.V3,'V3')

    df = encode_concate(df,df.V4,'V4')

    df = encode_concate(df,df.V9,'V9')

    

    #dropping the column 'V11' as some of its category are not there in training data

    df = df.drop(columns = 'V11')



    #dropping columns of 'V11' from test data which are not there in training data

    try:

        df = df.drop(columns = ['V119','V1110','V1111'])

    except:

        pass

    

#     std = StandardScaler()

#     df_for_x = df.drop(columns=['Class'])

#     normalize = Normalizer().fit(df)

#     X = normalize.transform(df)

#     X = std.fit_transform(df)

#     return X, df

    return df



#dropping the unnamed column from the data

def drop_unecessary(df):

    df = df.drop(columns = ['Unnamed: 0'])

    return df
y_train, df_new = Y_train_extract(df)



df_new = drop_unecessary(df_new)

# X_train, df_new = eda(df_new)

X_train = eda(df_new)

y_train = np.array(y_train)



df_test_prime = drop_unecessary(df_test)

# test, df_test_new = eda(df_test_prime)

test = eda(df_test_prime)

test = np.array(test)
sm = SMOTE(random_state=12, ratio = 1)

X_train, y_train = sm.fit_sample(X_train, y_train)
seed = 7

test_size = 0.25

X_train_prime, X_test_prime, y_train_prime, y_test_prime = train_test_split(X_train, y_train, test_size=test_size, random_state=seed, shuffle = True)
model = xgb.XGBClassifier(n_estimators=320, n_jobs=-1)

model.fit(X_train_prime, y_train_prime)
y_pred = model.predict(X_test_prime)

predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test_prime, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))





y_true = y_test_prime

y_scores = predictions

roc_auc_score(y_true, y_scores)
test_pred = model.predict_proba(test)

test_prime = pd.DataFrame(test_pred)

test_prime.columns = ['a', 'b']

sample = pd.read_csv('../input/webclubrecruitment2019/SAMPLE_SUB.csv')

sample.PredictedValue = test_prime['b']

sample.to_csv('predfinal_5.csv',index= False)

sample.head()
# model_1 = pd.read_csv('../input/f-ka-baap/f_ka_baap.csv')

# model_2 = pd.read_csv('../input/f-ka-baap-2/f_ka_baap_2.csv')

# model_3 = pd.read_csv('../input/final-pred/predfinal_5.csv')

# model_4 = pd.read_csv('../input/final-pred/predfinal_2.csv')

# model_5 = pd.read_csv('../input/final-pred/predfinal_3.csv')

# model_6 = pd.read_csv('../input/final-pred/predfinal_4.csv')



# ensemble_model = model_1.copy()

# ensemble_model['PredictedValue'] = (model_1['PredictedValue'] + model_2['PredictedValue'] + model_3['PredictedValue'] + model_4['PredictedValue'] + model_5['PredictedValue'] + model_6['PredictedValue'])/6

# ensemble_model.to_csv('final_pred.csv', index = False)



# ensemble_model.head()