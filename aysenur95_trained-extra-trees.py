# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
#read data

train_features_data = pd.read_csv('../input/hr-dataset/train_LZdllcl.csv')

test_features_data = pd.read_csv('../input/hr-dataset/test_2umaH9m.csv')
train_features_data.info()
train_features_data.drop(['employee_id'], axis="columns", inplace=True)

train_features_data.head()
cat_fetaures_col = []

for column in train_features_data.columns:

    if train_features_data[column].dtype == object:

        cat_fetaures_col.append(column)

        print(f"{column} : {train_features_data[column].unique()}")

        print(train_features_data[column].value_counts())

        print("-------------------------------------------")

#numeric-cat ==> discrete

disc_feature_col = []

for column in train_features_data.columns:

    if train_features_data[column].dtypes != object and train_features_data[column].nunique() <= 30:

        print(f"{column} : {train_features_data[column].unique()}")

        print(train_features_data[column].value_counts())

        disc_feature_col.append(column)

        print("-------------------------------------------")

        

disc_feature_col.remove('is_promoted')
cont_feature_col=[]

for column in train_features_data.columns:

    if train_features_data[column].dtypes != object and train_features_data[column].nunique() > 30:

        print(f"{column} : Minimum: {train_features_data[column].min()}, Maximum: {train_features_data[column].max()}")

        cont_feature_col.append(column)

        print("-------------------------------------------")
#there are missing values for "education" and "previous_year_rating" cols.

train_features_data.isnull().sum()
#eliminate null values(fill with mode of that column)



for column in train_features_data.columns:

    train_features_data[column].fillna(train_features_data[column].mode()[0], inplace=True)

print(train_features_data['education'].mode()[0])

print(train_features_data['previous_year_rating'].mode()[0])
#there are no missing values in our dataset anymore!!!

train_features_data.isnull().sum()
#outlier analysis using box-plot(continuos data can have outliers(aykırı değerler))



sns.set(style="whitegrid",font_scale=1)

plt.figure(figsize=(7,7))

sns.boxplot(data=train_features_data[cont_feature_col])

plt.xticks(rotation=30)

plt.title("Box plot of continuos features")

plt.show()
# find the IQR



q1 = train_features_data[cont_feature_col].quantile(.25)

q3 = train_features_data[cont_feature_col].quantile(.75)

IQR = q3-q1



print("         IQR")

print("------------------------------\n")

print(IQR)

print("         q1")

print("------------------------------\n")

print(q1)

print("         q3")

print("------------------------------\n")

print(q3)





lower_bound = q1 - 1.5*IQR

upper_bound = q3 + 1.5*IQR

print("\n--------lower bounds--------")

print(lower_bound)

print("\n--------upper bound---------")

print(upper_bound)

print(lower_bound)

print("-------------------")

print(lower_bound[0])

print(lower_bound[1])

print(lower_bound[2])
outliers_df = np.logical_or((train_features_data[cont_feature_col] < lower_bound), (train_features_data[cont_feature_col] > upper_bound)) 

outliers_df
outlier_total=[]

outlier_percentage=[]



for col in list(train_features_data[cont_feature_col].columns):

    try:

        outlier_total.append(outliers_df[col].value_counts()[True])

        outlier_percentage.append((outliers_df[col].value_counts()[True] / outliers_df[col].value_counts().sum())*100)

    except:

        outlier_total.append(0)

        outlier_percentage.append(0)

        

print(outlier_total)

print(outlier_percentage)



outlier_number_df=pd.DataFrame(zip(list(outliers_df.columns), outlier_total,outlier_percentage), columns=['name', 'total', 'outlier(%)'])

#outlier_df.set_index('name', inplace=True)

outlier_number_df
outlier_det_age_df=train_features_data[cont_feature_col]['age']

print(type(outlier_det_age_df))



outlier_det_los_df=train_features_data[cont_feature_col]['length_of_service']

outlier_det_age_df
print(outlier_det_age_df.head())

print(outlier_det_los_df.head())
outlier_age=train_features_data[cont_feature_col]['age'] > upper_bound[0]

print(outlier_age.head())



outlier_los=train_features_data[cont_feature_col]['length_of_service'] > upper_bound[1]

print(outlier_los.head())



#fill outliers with mean



outlier_det_age_df[outlier_age]=upper_bound[0]

print(outlier_det_age_df[outlier_age])



print("=======================")

outlier_det_los_df[outlier_los]=upper_bound[1]

print(outlier_det_los_df[outlier_los])

#update original train set with fixed outlier values



train_features_data['age']=outlier_det_age_df

train_features_data['length_of_service']=outlier_det_los_df
#there is NO outlier anymore!!!(see the boxplot)



sns.set(style="whitegrid",font_scale=1)

plt.figure(figsize=(7,7))

sns.boxplot(data=train_features_data[cont_feature_col])

plt.xticks(rotation=30)

plt.title("Box plot of continuos features")

plt.show()
#encode ediyoruzzz!!!



#encoding categorical features (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(train_features_data)

train_features_data_arr=enc.transform(train_features_data)



col_names_list=train_features_data.columns

encoded_categorical_df=pd.DataFrame(train_features_data_arr, columns=col_names_list)



#check types

encoded_categorical_df.info()
# split df to X and Y

from sklearn.model_selection import train_test_split



y = encoded_categorical_df.loc[:, 'is_promoted'].values

X = encoded_categorical_df.drop('is_promoted', axis=1)



# split data into 80-20 for training set / test set

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=100)



X_train = X

y_train = y





#now, prepare the real test dataset to find preds and (to submit)

test_df = test_features_data.drop(['employee_id'], axis=1)



#fillna test_df

for column in test_df.columns:

    test_df[column].fillna(test_df[column].mode()[0], inplace=True)



#fix outliers in test_df same as before(baskilama)

test_df['age'][test_df['age'] > upper_bound[0]] = upper_bound[0]

test_df['length_of_service'][test_df['length_of_service'] > upper_bound[1]]=upper_bound[1]



#create dummy is_promoted in test_df to match column numbers (it will be removed again after encoding)

test_df["is_promoted"] = 1



#encode test_df

test_df_arr = enc.transform(test_df)

test_df_encoded = pd.DataFrame(test_df_arr, columns=col_names_list)



# drop "is_promoted"

X_test = test_df_encoded.drop('is_promoted', axis=1)

binary_cols = [col for col in list(encoded_categorical_df.columns) if encoded_categorical_df[col].nunique() <= 2] 

binary_cols.remove('is_promoted')



non_binary_cols = [col for col in list(encoded_categorical_df.columns) if encoded_categorical_df[col].nunique() > 2]
#normalization(make all values bet. 0-1)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train[non_binary_cols])



X_train_normalized_arr=scaler.transform(X_train[non_binary_cols])

X_train_normalized_df=pd.DataFrame(X_train_normalized_arr, columns=non_binary_cols)



X_test_normalized_arr=scaler.transform(X_test[non_binary_cols])

X_test_normalized_df=pd.DataFrame(X_test_normalized_arr, columns=non_binary_cols)
print(X_train_normalized_df.head())

print(X_test_normalized_df.head())
X_train_binary_cols_df = X_train[binary_cols]

X_train_binary_cols_df.reset_index(inplace=True, drop=True)



X_train_final_df = pd.concat([X_train_binary_cols_df,X_train_normalized_df], axis=1)



X_train_final_df.head()
X_test_binary_cols_df = X_test[binary_cols]

X_test_binary_cols_df.reset_index(inplace=True, drop=True)



X_test_final_df = pd.concat([X_test_binary_cols_df,X_test_normalized_df], axis=1)



X_test_final_df.head()
#here is size of our train and test datasets

print(len(X_train_final_df))

print(len(X_test_final_df))
print(len(y_train))
#import necessary libraries



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score



# cross-validation with 10 splits

cv = StratifiedShuffleSplit(n_splits=10, random_state = 42, test_size=0.2)
# display test scores and return result string and indexes of false samples

def display_test_scores(test, pred):

    str_out = ""

    str_out += ("TEST SCORES\n")

    str_out += ("=====================\n")

    

    #print accuracy

    accuracy = accuracy_score(test, pred)

    str_out += ("ACCURACY: {:.4f}\n".format(accuracy))

    str_out += ("\n")

    str_out += ("--------------------\n")

    



    #print AUC score

    auc = roc_auc_score(test, pred)

    str_out += ("AUC: {:.4f}\n".format(auc))

    str_out += ("\n")

    str_out += ("--------------------\n")



    #print confusion matrix

    str_out += ("CONFUSION MATRIX:\n")

    conf_mat = confusion_matrix(test, pred)

    str_out += ("{}".format(conf_mat))

    str_out += ("\n")

    str_out += ("\n")

    str_out += ("--------------------\n")

    

    #print FP, FN

    str_out += ("FALSE POSITIVES:\n")

    fp = conf_mat[1][0]

    pos_labels = conf_mat[1][0]+conf_mat[1][1]

    str_out += ("{} out of {} positive labels ({:.4f}%)\n".format(fp, pos_labels,fp/pos_labels))

    str_out += ("\n")

    str_out += ("------------------------------------\n")



    str_out += ("FALSE NEGATIVES:\n")

    fn = conf_mat[0][1]

    neg_labels = conf_mat[0][1]+conf_mat[0][0]

    str_out += ("{} out of {} negative labels ({:.4f}%)\n".format(fn, neg_labels, fn/neg_labels))

    str_out += ("\n")

    str_out += ("------------------------------------\n")



    #print classification report

    str_out += ("PRECISION, RECALL, F1 scores:\n\n")

    str_out += ("{}".format(classification_report(test, pred)))

    

    false_indexes = np.where(test != pred)

    return str_out, false_indexes
#train all dataset



from sklearn.ensemble import ExtraTreesClassifier



extra = ExtraTreesClassifier(random_state=0, bootstrap=True, class_weight=None, max_features=None, max_samples=0.3, n_estimators=200)





extra.fit(X_train_final_df, y_train)



# prediction results

y_pred = extra.predict(X_test_final_df)
y_pred=y_pred.astype(int)

type(y_pred[0])
# create output df

employee_ids = test_features_data["employee_id"].values

output_df=pd.DataFrame(zip(employee_ids, y_pred), columns=["employee_id", "is_promoted"])



#create output csv file

output_df.to_csv('extra_trees_file.csv', index=False, sep=",")