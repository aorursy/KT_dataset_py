import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # To plot graphs
from sklearn.preprocessing import RobustScaler ## For data scaling
from sklearn.model_selection import train_test_split
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import h2o
student_drop_out=pd.read_csv('../input/student-drop-out/studentDropIndia_20161215.csv')
student_drop_out.head()
student_drop_out_orig=student_drop_out
student_drop_out.dtypes
student_drop_out.shape
def find_column_na(dataframe,na_limit):
    dataframe_null_check = dataframe.isnull()
    dataframe_null_check_sum = dataframe_null_check.sum()
    columnlist=[]
    for i in range(0,len(dataframe_null_check_sum)):
        if(dataframe_null_check_sum[i]>na_limit):
            print(dataframe_null_check_sum.index[i],dataframe_null_check_sum[i])
            columnlist.append(dataframe.columns[i])
    return columnlist
na_columns = find_column_na(student_drop_out,0)
student_drop_out.describe()
#Function to replace blank numerical values with zero
def replaceNumBlankwithZero(dataframe,columns):
    data_no_blank = dataframe
    for col in columns:
            data_no_blank[col].fillna(0, inplace=True)
    
    return data_no_blank
student_drop_out_no_na=replaceNumBlankwithZero(student_drop_out,na_columns)
student_drop_out_no_na.isnull().any()
student_drop_out_no_na.sample(10)
#Function to find Numerical columns from our data frame
def find_numerical_col(dataframe):
    columnlist_num=[]
    #fig =  plt.figure(figsize=(10,10))
    for i in range(1,len(dataframe.columns)):
           if dataframe[dataframe.columns[i]].dtypes=='int64' or dataframe[dataframe.columns[i]].dtypes=='float64':
                    columnlist_num.append(dataframe.columns[i])
    return columnlist_num
# Robust  scaler
def do_robustScaler(X):
    robust=RobustScaler()
    X_robust = robust.fit_transform(X)
    return X_robust
num_columns = find_numerical_col(student_drop_out_no_na)
student_drop_out_no_na_num_col=student_drop_out_no_na[num_columns]
# Let run robus scale function now to scale our numneric data
student_drop_out_no_na_num_col_scaled = do_robustScaler(student_drop_out_no_na_num_col)
# Return data is numpy array
type(student_drop_out_no_na_num_col_scaled)
# We will first drop the student id column
student_drop_out_no_na = student_drop_out_no_na.drop('student_id', 1)
#Find categorical columns
def find_categ_column(dataframe):
    return dataframe.select_dtypes(exclude=["number","bool_"]).columns
# we have five columns with string data types
cat_columns = find_categ_column(student_drop_out_no_na)
# Let run getDummies function on it to encode categorical data
student_drop_out_no_na_dummy = pd.get_dummies(student_drop_out_no_na[cat_columns],columns = cat_columns)
student_drop_out_no_na_dummy.head()
scaled_data_df=pd.DataFrame(student_drop_out_no_na_num_col_scaled, columns=num_columns) 
scaled_data_df.head()
# Lets join the 2 data frame horizentally
X = pd.concat([scaled_data_df,student_drop_out_no_na_dummy],axis=1)
X.head()
# Lets split the dataset into training and test.
print("Total number of Rows in data set")
print(X.shape[0])

X_nonanomly=X.loc[X['continue_drop_continue'] == 1]
print("Total number of Rows for non anomly data set")
print(X_nonanomly.shape[0])

Xtrain, Xtest = train_test_split(X_nonanomly,test_size=0.2)
print("Total number of Rows for non anomly train data set")
print(Xtrain.shape[0])
print("Total number of Rows for non anomly test data set")
print(Xtest.shape[0])

print("Total number of Rows for anomly data set")
test_anomly=X.loc[X['continue_drop_drop'] == 1]
print(test_anomly.shape[0])
## So this will give us Test data set and we will detect anomly in our test data set using H2O encoder
print("Total number of Rows for non anomly test data set and anomly test datset")
Xtest_anom=pd.concat([Xtest,test_anomly],axis=0)
print(Xtest_anom.shape[0])
labels = ['continue', 'dropout']
sizes = [student_drop_out_no_na['continue_drop'].value_counts()['continue'],
         student_drop_out_no_na['continue_drop'].value_counts()['drop']
        ]
colors = ['gold','lightcoral']
explode = (0, 0.1)  # explode 1st slice
plt.figure(figsize=[7, 7])
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=70)
plt.title("Continue vs Dropout %",fontsize=20)
plt.axis('equal')
plt.show()
h2o.init(nthreads=-1, enable_assertions = False)
h2odf_train=h2o.H2OFrame(Xtrain)
h2odf_test=h2o.H2OFrame(Xtest_anom)
anomly_model = H2OAutoEncoderEstimator(activation="Tanh", 
                                hidden=[5,3,5], 
                                model_id="anomly_model",
                                ignore_const_cols=False, 
                                autoencoder=True,
                                epochs=100)
#List of predictors
predictors=list(range(0,14))
#Lets train our anomly model
anomly_model.train(x=predictors,training_frame=h2odf_train)
detect_anomly = anomly_model.anomaly(h2odf_test)
detect_anomly = detect_anomly.as_data_frame()

detect_anomly['id'] = detect_anomly.index

#Lets plot the result of our test dataset.

plt.scatter(x=detect_anomly['id'],y=detect_anomly['Reconstruction.MSE'], c='r', alpha=0.5)
plt.show()
# We will shutdown H2O server here
h2o.cluster().shutdown()