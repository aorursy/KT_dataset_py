# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler  

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.preprocessing import LabelEncoder



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split, KFold, cross_validate



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import roc_auc_score as auc





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")





print("Shape of test_data "+str(test_data.shape))



print("Shape of train_data "+str(train_data.shape))



y_train = train_data['target']

train_data_id = train_data['id']



test_data_id = test_data['id']



train_data.drop(['target','id'], axis=1,inplace=True)

test_data.drop('id', axis=1,inplace=True)



print("New shape of test_data "+str(test_data.shape))



print("New shape of train_data "+str(train_data.shape))



print("Contents of train_data\n")





train_data.head()
print("Contents of test_data\n")





test_data.head()
train_data.columns
test_data.columns
missing_val_count_by_col = train_data.isnull().sum()



print("Columns in train_data with missing values, and the number of missing values")

print(missing_val_count_by_col[missing_val_count_by_col > 0])
X_train = train_data



print("Shape of y_train before train/validation split is"+str(y_train.shape))

print("Shape of X_train before train/validation split is"+str(X_train.shape))

print("\n")

#Split training set into a training set and validation set



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, test_size=0.1, random_state=0)



print("Shape of y_train is"+str(y_train.shape))

print("Shape of X_train is"+str(X_train.shape))



print("Shape of y_valid is"+str(y_valid.shape))

print("Shape of X_valid is"+str(X_valid.shape))
#Printing out the number of unique values for each column in the training data

for col_name in X_train.keys():

    print("Column " + col_name + " has " + str( len(X_train[col_name].unique()) ) + " unique values")
print(X_train["nom_6"].value_counts().sort_values(ascending=False))

print(X_train["nom_7"].value_counts().sort_values(ascending=False))

print(X_train["nom_8"].value_counts().sort_values(ascending=False))

print(X_train["nom_9"].value_counts().sort_values(ascending=False))
%%time



label_X_train = X_train.copy()

label_X_valid = X_valid.copy()

label_test_data = test_data.copy()



sk_label_encoder = LabelEncoder()



for mycol in ["ord_0","ord_1","ord_2","ord_2","ord_3","ord_4","ord_5"]:

    label_X_train[mycol] = sk_label_encoder.fit_transform(label_X_train[mycol])

    label_X_valid[mycol] = sk_label_encoder.transform(label_X_valid[mycol])

    label_test_data[mycol] = sk_label_encoder.transform(label_test_data[mycol])

    
%%time

label_X_train.head()
%%time

label_X_valid.head()
%%time

label_test_data.head()
low_cardinality_nom_cols = []

high_cardinality_nom_cols = []





for nom_col in range(10):

    nom_col_name = "nom_"+str(nom_col)

    if label_X_train[nom_col_name].nunique() < 10:

        low_cardinality_nom_cols.append(nom_col_name)

    else:

        high_cardinality_nom_cols.append(nom_col_name)



print("Nominal columns low cardinality (<=10):", low_cardinality_nom_cols)

print("Nominal columns with high cardinality (>10):", high_cardinality_nom_cols)
#combining everything into a single data frame so as to apply a uniform encoding across train, validation, test data sets

#If this is not OK please provide your feedback, with references as to why (tyvm)

label_X_train["kind"] = "train"

label_X_valid["kind"] = "valid"

label_test_data["kind"] = "test"



big_df = pd.concat([label_X_train, label_X_valid, label_test_data], sort=False ).reset_index(drop=True)



print("big_df shape is "+str(big_df.shape))
%%time

for col in low_cardinality_nom_cols:

    temp_df_to_concat = pd.get_dummies(big_df[col], prefix=col)

    big_df = pd.concat([big_df, temp_df_to_concat], axis=1)

    big_df.drop([col],axis=1, inplace=True)





for col in high_cardinality_nom_cols:

        big_df[f"hash_{col}"] = big_df[col].apply( lambda x: hash(str(x)) % 5000)

        



#Not sure if I can run this over all of big_df. In the example the coder runs it over df_train only



#Just modify training or validation data set



big_df_train_valid = big_df.loc[ (big_df["kind"] == "train") | (big_df["kind"]=="valid") ]

big_df_test = big_df.loc[big_df["kind"] == "test"]



for col in high_cardinality_nom_cols:

    enc_nom_1 =  (big_df_train_valid.groupby(col).size() ) / len(big_df_train_valid)

    big_df_train_valid[f"freq_{col}"] = big_df_train_valid[col].apply( lambda x : enc_nom_1[x])



for col in high_cardinality_nom_cols:

    enc_nom_1 =  (big_df_test.groupby(col).size() ) / len(big_df_test)

    big_df_test[f"freq_{col}"] = big_df_test[col].apply( lambda x : enc_nom_1[x])

    

label_X_train = big_df_train_valid.loc[ big_df["kind"]=="train" ]

label_X_valid = big_df_train_valid.loc[ big_df["kind"]=="valid" ]

label_test_data = big_df_test.loc[ big_df["kind"]=="test" ]



label_X_train.drop("kind", axis=1, inplace=True)

label_X_valid.drop("kind", axis=1, inplace=True)

label_test_data.drop("kind", axis=1, inplace=True)
label_X_train.head()
label_X_valid.head()
label_test_data.head()
print("shape of label_test_data "+str(label_test_data.shape))

print("shape of label_X_train "+str(label_X_train.shape))

print("shape of label_X_valid "+str(label_X_valid.shape))



del big_df

del big_df_test

del big_df_train_valid
%%time



#More encoding. Borrowed idea from another notebook. Trying other things were too slow



binary_dict = {"T":1, "F":0, "Y":1, "N":0}





label_X_train["bin_3"] = label_X_train["bin_3"].map(binary_dict)

label_X_train["bin_4"] = label_X_train["bin_4"].map(binary_dict)



label_X_valid["bin_3"] = label_X_valid["bin_3"].map(binary_dict)

label_X_valid["bin_4"] = label_X_valid["bin_4"].map(binary_dict)



label_test_data["bin_3"] = label_test_data["bin_3"].map(binary_dict)

label_test_data["bin_4"] = label_test_data["bin_4"].map(binary_dict)



label_X_train.drop(high_cardinality_nom_cols, axis=1, inplace=True)

label_X_valid.drop(high_cardinality_nom_cols, axis=1, inplace=True)

label_test_data.drop(high_cardinality_nom_cols, axis=1, inplace=True)



label_X_train.head()
print("Rows of label_X_train "+str(label_X_train.shape[0]))

print("Rows of y_train "+str(y_train.shape[0]))

print("Rows of label_X_valid "+str(label_X_valid.shape[0]))

print("Rows of y_valid "+str(y_valid.shape[0]))



ada_boost_model = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=0.05, base_estimator=DecisionTreeClassifier(max_depth=10))
#using a StandardScaler as the sklearn documents suggest scaling the inputs

neural_model = MLPClassifier(hidden_layer_sizes=(96,96,48,48,24,12,6,3,1), 

                             solver="adam", 

                             batch_size="auto", 

                             #learning_rate="adaptive",

                             learning_rate_init=0.002,

                             max_iter=200,

                             n_iter_no_change=10,

                             random_state=1,

                             verbose=True

                            )



NNPipeline = Pipeline([("scaler",StandardScaler()), ("NN",neural_model)])
gradient_boost_model = GradientBoostingClassifier(n_estimators=50)



#I was thinking of doing my own train/valid split

#but realized with the cross_val_score() function, I need to

#recombine my train/validation sets and let the internal functionality of cross_val_score()

#do this splitting for me. So that's why I'm recombining them below :\

new_X_train = pd.concat([label_X_train, label_X_valid], axis=0)

new_y_train = pd.concat([y_train, y_valid], axis=0)

new_X_train_scaled = pd.DataFrame()

my_columns= new_X_train.columns
%%time

n_folds = 7



kfold = KFold(n_splits=n_folds, shuffle=False, random_state=42)



cv_results = cross_val_score(gradient_boost_model, new_X_train.values, new_y_train,

                            cv=kfold, scoring='roc_auc', n_jobs=-1)



print("gradient_boost_model average results",cv_results.mean())



cv_results = cross_val_score(ada_boost_model, new_X_train.values, new_y_train,

                            cv=kfold, scoring='roc_auc', n_jobs=-1)



print("ada_boost_model average results",cv_results.mean())



#cv_results = cross_val_score(NNPipeline, new_X_train.values, new_y_train,

#                            cv=kfold, scoring='roc_auc', n_jobs=-1)



#print("NNPipeline average results",cv_results.mean())
%%time

gradient_boost_model.fit(new_X_train, new_y_train)
y_test_pred = gradient_boost_model.predict(label_test_data)



myscore = gradient_boost_model.score(label_test_data)
label_test_data.head()
y_test_pred
y_test_pred.shape
import matplotlib.pyplot as plt

%matplotlib inline



plt.hist(y_test_pred, density=True, bins=2)

#plt.xticks(x+0.5,['0','1'])

plt.ylabel("number of predictions")

plt.xlabel("values")



submission = pd.DataFrame({'id':test_data_id, 'target':y_test_pred})

submission.head()
submission.to_csv('submission.csv', index=False)