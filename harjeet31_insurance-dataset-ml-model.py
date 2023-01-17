# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter)ill list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Loading dataset



insurance_data = pd.read_csv("../input/insurance-dataset/insurance_data.csv")



insurance_data.head()
# Check for columns



insurance_data.columns
# Check the data info

insurance_data.info()
# describe the data

insurance_data.describe()
# Select the categorical columns



insurance_data.select_dtypes(include=['object']).columns
#select the int columns

insurance_data.select_dtypes(include=['int64']).columns
#select the float columns

insurance_data.select_dtypes(include=['float']).columns
# Select the target variable



insurance_data.Response.unique()
# Visualize the distribution of target variable



sns.countplot(x = "Response", data = insurance_data)

plt.title("Distribution of Target variable")



## histogram to show different features

insurance_data.hist(figsize=(18,22))

plt.show()
#function to define the category to feature based on their percentile



def category(feature,choices_values=[]):

    conditions = [

    (insurance_data[feature] <= insurance_data[feature].quantile(0.25)),

    (insurance_data[feature] > insurance_data[feature].quantile(0.25)) & (insurance_data[feature] <= insurance_data[feature].quantile(0.75)),

    (insurance_data[feature] > insurance_data[feature].quantile(0.75))]

    insurance_data[feature+str("_cat")] = np.select(conditions, choices_values)

    insurance_data[feature+"_cat"]

    return











    
# Let us analyze the average age



insurance_data.Ins_Age.describe()
# histogram of insurance age



plt.subplot(1,2,2)

insurance_data["Ins_Age"].hist()

plt.xlabel("Applicant Age")

plt.ylabel("Number of Applicants")

plt.title("Histogram - Distribution of Applicant Age")

#define category of 'Ht' height of the applicant



ch = ['young', 'average', 'old']

feature = 'Ins_Age'



print(ch)

category(feature,ch)

insurance_data[feature+str("_cat")]
# histogram of Heigh of applicant



plt.subplot(1,2,2)

insurance_data["Ht"].hist()

plt.xlabel("Applicant Height")

plt.ylabel("Number of Applicants")

plt.title("Histogram - Distribution of Applicant Height")
#define category of 'Ht' height of the applicant



ch = ['short', 'average', 'tall']

feature = 'Ht'



print(ch)

category(feature,ch)

insurance_data[feature+str("_cat")]
#Define the histogram for Wt for applicant



plt.subplot(1,2,2)

insurance_data["Wt"].hist()

plt.xlabel("Applicant Weight")

plt.ylabel("Number of Applicants")

plt.title("Histogram - Distribution of Applicant Weight")
#define category of 'Wt' height of the applicant



ch = ['thin', 'average', 'fat']

feature = 'Wt'



print(ch)

category(feature,ch)

insurance_data[feature+str("_cat")]
# Histogram for  BMI



plt.subplot(1,2,2)

insurance_data["BMI"].hist()

plt.xlabel("Applicant BMI")

plt.ylabel("Number of Applicants")

plt.title("Histogram - Distribution of Applicant BMI")
#define category of 'BMI' Body Mass Index of the applicant



ch = ['under_weight', 'average', 'over_weight']

feature = 'BMI'



print(ch)

category(feature,ch)

insurance_data[feature+str("_cat")]
#Find out null values columns

null_columns = insurance_data.columns[insurance_data.isnull().any()]

null_columns_values = insurance_data[null_columns].isnull().sum().sort_values(ascending=False)

null_columns_values
# Drop the 12 columns with more missing values



drop_columns = ['Medical_History_10','Medical_History_32','Medical_History_24','Medical_History_15',

               'Family_Hist_5','Family_Hist_3','Family_Hist_2','Insurance_History_5','Family_Hist_4',

               'Employment_Info_6','Medical_History_1','Employment_Info_4','Employment_Info_1']



insu_data_filter = insurance_data.drop(drop_columns,axis=1)

insu_data_filter
#shape of filter



insu_data_filter.shape
# Drop categorical columns



cat_columns = ['Ins_Age','Wt','Ht','BMI']

insu_data_filter.drop(cat_columns,axis=1)
# List all the medical keyword columns



medical_keyword_columns = insu_data_filter.columns[insu_data_filter.columns.str.startswith('Medical_Keyword')]

medical_keyword_columns
insu_data_filter["Medical_Keyword"] = insu_data_filter[medical_keyword_columns].sum(axis = 1)

insu_data_filter['Medical_Keyword']
insu_data_filter.drop(medical_keyword_columns, axis = 1, inplace = True)
insu_data_filter.shape
medical_history_columns = insu_data_filter.columns[insu_data_filter.columns.str.startswith("Medical_History_")]

medical_history_columns
for col in medical_history_columns:

    print("Checking the column {}, value counts in %:".format(col))

    print(insu_data_filter[col].value_counts(normalize = True)* 100)
drop_medical_columns = ['Medical_History_5','Medical_History_6','Medical_History_7','Medical_History_8',

                'Medical_History_11','Medical_History_12','Medical_History_14','Medical_History_17',

                'Medical_History_18','Medical_History_19','Medical_History_20','Medical_History_22',

                'Medical_History_27','Medical_History_28','Medical_History_30','Medical_History_31',

                'Medical_History_33','Medical_History_35','Medical_History_37','Medical_History_38',

                'Medical_History_39','Medical_History_40']
# drop the column

insu_data_filter.drop(drop_medical_columns,axis=1,inplace=True)
# check the shape of the dataset



insu_data_filter.shape
# drop the ID column

insu_data_filter.drop('Id',axis=1,inplace=True)
insu_data_filter.shape
insu_data_filter
insurance_history_columns = insu_data_filter.columns[insu_data_filter.columns.str.startswith("Insurance_History_")]

insurance_history_columns
for col in insurance_history_columns:

    print("Checking the column {}, value counts in %:".format(col))

    print(insu_data_filter[col].value_counts(normalize = True)* 100)
col = ['Ht','Wt','BMI']

insu_data_filter.drop(col,axis=1,inplace=True)

insu_data_filter.shape
categorical_columns = insu_data_filter.select_dtypes(include=['object']).columns

categorical_columns
insu_data_filter[categorical_columns].head()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

insu_data_filter['Product_Info_2']= label_encoder.fit_transform(insu_data_filter['Product_Info_2'])

insu_data_filter['Product_Info_2']
insu_data_filter = pd.get_dummies(insu_data_filter, columns = ["Ins_Age_cat","Ht_cat","Wt_cat","BMI_cat"], drop_first = True)

insu_data_filter
insu_data_filter.shape
X = insu_data_filter.loc[:, insu_data_filter.columns != "Response"]

X.head()
Y = insu_data_filter.Response

Y.head()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=1)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
from sklearn import tree

model = tree.DecisionTreeClassifier(random_state = 0)

model.fit(X_train, Y_train)
y_pred_train = model.predict(X_train)

y_pred_train

y_pred_test = model.predict(X_test)

y_pred_test
from sklearn.model_selection import GridSearchCV

decision_tree_classifier = tree.DecisionTreeClassifier(random_state = 0)



tree_para = [{'criterion':['gini','entropy'],'max_depth': range(2,60),

                             'max_features': ['sqrt', 'log2', None] }]

                            

grid_search = GridSearchCV(decision_tree_classifier,tree_para, cv=10, refit='AUC')

grid_search.fit(X_train, Y_train)
y_pred_test1 = grid_search.predict(X_test)
rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=12,

                             min_samples_leaf=30, 

                             min_samples_split=80,

                             max_features=30,

                             n_estimators=600)
y_pred_test2 = rfc.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy score for test data Model 1 is:', accuracy_score(Y_test,y_pred_test))

print('Accuracy score for test data Model 2 is:', accuracy_score(Y_test,y_pred_test1))

print('Accuracy score for test data Model 3 is:', accuracy_score(Y_test,y_pred_test2))