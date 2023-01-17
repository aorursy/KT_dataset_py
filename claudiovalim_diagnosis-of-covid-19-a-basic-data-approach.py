import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #ploting

import seaborn as sns



pd.set_option('display.max_columns', 500)



pd.set_option('display.max_rows', 500)

# Path of the file to read

covid_data_path = "/kaggle/input/covid19/dataset.xlsx"



# Read the file into a variable covid_data

covid_data = pd.read_excel(covid_data_path)



# Print the first 5 rows of the data

covid_data.head()



# Print the percentage of missing values



percent_of_missing_values = sum(covid_data.isnull().sum())/sum(covid_data.isnull().count())



print('There are {:.1%} of missing values in the dataset.'.format(percent_of_missing_values))

# Drop Id Column



ids = covid_data[["Patient ID"]]

covid_data = covid_data.drop("Patient ID",axis=1)





for col in covid_data.columns :

    

    if covid_data[col].dtypes == "object" and col != "Patient ID":

        

        covid_data.loc[covid_data[col] == "positive", col] = 1

        covid_data.loc[covid_data[col] == "negative", col] = 0

        covid_data.loc[covid_data[col] == "detected", col] = 1

        covid_data.loc[covid_data[col] == "not_detected", col] = 0

        covid_data.loc[covid_data[col] == "present", col] = 1

        covid_data.loc[covid_data[col] == "absent", col] = 0       

        covid_data.loc[covid_data[col] == "Não Realizado", col] = float("Nan")

        covid_data.loc[covid_data[col] == "not_done", col] = float("Nan")

        covid_data.loc[covid_data[col] == "<1000", col] = 999

        

        

        # print(covid_data.groupby(covid_data[col].astype(str))[['Patient ID']].count())

        

covid_data[["Urine - Leukocytes"]] = covid_data[["Urine - Leukocytes"]].astype(float)

covid_data[["Urine - pH"]] = covid_data[["Urine - pH"]].astype(float)

covid_data[["SARS-Cov-2 exam result"]] = covid_data[["SARS-Cov-2 exam result"]].astype(int)
col_num = [col for col in covid_data.columns if covid_data[col].dtype in ('float64', 'int64') 

           and len(covid_data[col].unique())>3

           and (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) <= 0.9 # at least 10% os values are filled

          ]



colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(covid_data[col_num].astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
df_1 = pd.DataFrame(columns = ['variable', 'value'])



for col in covid_data.columns:

    if len(covid_data[col].unique()) <= 5 :

        x1 = pd.DataFrame({'variable' : col,

                           'value' :covid_data[col].unique()

                          })

        df_1 = df_1.append(x1)



df_1["count_of_positive"] = 0

df_1["count_of_negative"] = 0



for row in range(len(df_1)):

    variable = df_1.iloc[row]["variable"]

    value = df_1.iloc[row]["value"]

    

    if pd.isna(value):

        pos = len(covid_data.loc[ (pd.isna(covid_data[variable])) & (covid_data['SARS-Cov-2 exam result'] == 1) ])

        neg = len(covid_data.loc[ (pd.isna(covid_data[variable])) & (covid_data['SARS-Cov-2 exam result'] == 0) ])

        

        df_1["count_of_positive"].loc[ (df_1['variable'] == variable) & (pd.isna(df_1['value'])) ] = pos

        df_1["count_of_negative"].loc[ (df_1['variable'] == variable) & (pd.isna(df_1['value'])) ] = neg

        

        # print(neg)

    else:

        pos = len(covid_data.loc[ (covid_data[variable] == value) & (covid_data['SARS-Cov-2 exam result'] == 1) ])

        neg = len(covid_data.loc[ (covid_data[variable] == value) & (covid_data['SARS-Cov-2 exam result'] == 0) ])



        df_1["count_of_positive"].loc[ (df_1['variable'] == variable) & (df_1['value'] == value) ] = pos

        df_1["count_of_negative"].loc[ (df_1['variable'] == variable) & (df_1['value'] == value) ] = neg
df_1["percentage_of_positive"] = df_1["count_of_positive"] /  (df_1["count_of_positive"] + df_1["count_of_negative"] )



df_1.sort_values(by = "percentage_of_positive", ascending = False)
from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



features = [col for col in covid_data.columns if covid_data[col].dtype in ('float64', 'int64') and

                                                 (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) == 0]



covid_data2 = covid_data[features]



# Splitting data

target = covid_data2['SARS-Cov-2 exam result']

X_train, X_test, y_train, y_test = train_test_split(covid_data2.drop('SARS-Cov-2 exam result',axis=1), 

                                                    target, test_size=0.20, 

                                                    random_state=0)





# Define model. Specify a number for random_state to ensure same results each run

covid_model = ensemble.RandomForestClassifier(random_state=1)



# Fit model

covid_model.fit(X_train, y_train)



# Make predictions

covid_predictions = covid_model.predict(X_test)



# Accuracy



print('Accuracy of: %.1f%%' % (accuracy_score(y_test, covid_predictions)*100))





# Confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, covid_predictions).ravel()



(tn, fp, fn, tp)
from imblearn.over_sampling import SMOTE





features = [col for col in covid_data.columns if covid_data[col].dtype in ('float64', 'int64') and

                                                 (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) == 0]



target = covid_data2['SARS-Cov-2 exam result']



# Splitting data



X_train, X_test, y_train, y_test = train_test_split(covid_data2.drop('SARS-Cov-2 exam result',axis=1), 

                                                    target, test_size=0.20, 

                                                    random_state=0)



# OverSample

oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)





# Define model. Specify a number for random_state to ensure same results each run

covid_model = ensemble.RandomForestClassifier(random_state=1)



# Fit model

covid_model.fit(X_train, y_train)



# Make predictions

covid_predictions = covid_model.predict(X_test)



# Accuracy



print('Accuracy of: %.1f%%' % (accuracy_score(y_test, covid_predictions)*100))





# Confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, covid_predictions).ravel()



(tn, fp, fn, tp)

features_cat = [col for col in covid_data.columns if  ( (covid_data[col].dtype == 'object') and

                                                    (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) >= 0 and

                                                    (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) <= 0.9

                                                  )]



features_num = [col for col in covid_data.columns if  ( (covid_data[col].dtype != 'object') and

                                                    (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) >= 0 and

                                                    (covid_data[col].isnull().sum() / covid_data[col].isnull().count()) <= 0.9

                                                  )]



features  = features_cat + features_num

covid_data2 = covid_data[features].fillna(-1)



# label encoding the data 

from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

           

for ele in features_cat:



    covid_data2[ele]= le.fit_transform(covid_data2[ele]) 

from imblearn.over_sampling import SMOTE



target = covid_data2['SARS-Cov-2 exam result']



# Splitting data



X_train, X_test, y_train, y_test = train_test_split(covid_data2.drop('SARS-Cov-2 exam result',axis=1), 

                                                    target, test_size=0.20, 

                                                    random_state=0)



# OverSample

oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)





# Define model. Specify a number for random_state to ensure same results each run

covid_model = ensemble.RandomForestClassifier(random_state=1)



# Fit model

covid_model.fit(X_train, y_train)



# Make predictions

covid_predictions = covid_model.predict(X_test)



# Accuracy



print('Accuracy of: %.1f%%' % (accuracy_score(y_test, covid_predictions)*100))





# Confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, covid_predictions).ravel()



(tn, fp, fn, tp)



features = [col for col in covid_data.columns if covid_data[col].dtype in ('float64', 'int64') 

           and len(covid_data[col].unique())<30

            and len(covid_data[col].unique())>=2

          ]



for ele in features:

    

    graph = covid_data.groupby([ele,'SARS-Cov-2 exam result'])['SARS-Cov-2 exam result'].count().unstack()

    graph.plot(kind='bar', stacked=True)


