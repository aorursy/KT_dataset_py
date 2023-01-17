import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from itertools import compress
df = pd.read_csv("../input/2015Ed.csv")



#First remove all rows that do not contain the target values of 'TUITIONFEE_OUT'

df = df[df['TUITIONFEE_OUT'].notna()]



#identify all columns that have above 10% null values

not_empty = []

for column in df.columns:

    if (df[column].isna().sum() / len(df[column])) <= 0.1:

        not_empty.append(column)

full = df[not_empty].copy()



#identify all columns that have 10% or below 'PrivacySuppressed' values

not_suppressed = []

for column in full.columns:

    if (full[column].apply(str).str.count('PrivacySuppressed').sum() / len(full[column])) <= 0.9:

        not_suppressed.append(column)

#create a new dataframe with those columns

cleaned = full[not_suppressed].copy()

#replace all values of 'PrivacySuppressed' to the number 0

cleaned = cleaned.replace(to_replace = 'PrivacySuppressed', value = 0)



#drop all rows that contain null values to create the final dataset

df = cleaned.dropna()
df.shape
df.head()
def linear_predict(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    reg = linear_model.Ridge(alpha=.6)

    reg.fit(X_train,y_train)

    y_predict = reg.predict(X_test)

    return y_predict, y_test
def epoch_test(X, y, epochs = 100, print_output=False, graph_output = False):

    accuracy = []

    average = {}

    for i in range(epochs):

        y_predict, y_test = linear_predict(X,y)

        accuracy.append(np.sqrt(mean_squared_error(y_test,y_predict)))

        average[i] = np.mean(accuracy)

    if graph_output == True:

        plt.scatter(average.keys(),average.values())

        plt.title('Average RMSE over ' + str(epochs) + ' epochs')

        plt.xlabel("Epoch")

        plt.ylabel("RMSE")

        plt.show()

        plt.scatter(y_predict,y_test)

        plt.title("Predicted vs Actual " + y.name)

        plt.xlabel("Actual " + y.name)

        plt.ylabel("Predicted " + y.name)

        plt.show()

    if print_output == True:

        print('number of features: ' + str(len(X.columns)))

        print('number of examples: ' + str(len(X)))

        print('mean target value: ' + str(np.mean(y)))

        print('median target value: ' + str(np.median(y)))

        print('std. dev. of target value: ' + str(np.std(y)))

        print('final RMSE over ' + str(epochs) + ' epochs: ' + str(average[epochs-1]))

    return average[epochs-1]
def feature_compare_df(df, feature_list, target_list):

    output = pd.DataFrame()

    for target in target_list:

        comparison = {}

        for feature in feature_list.keys():

            comparison[str(feature)] = epoch_test(df[feature_list[feature]], df[target])

        comparison['Mean'] = np.mean(df[target])

        comparison['Median'] = np.median(df[target])

        comparison['StdDev'] = np.std(df[target])

        output = pd.concat([output, pd.DataFrame({str(target):pd.Series(comparison)})], axis = 1)

    return output
target_features = ['COSTT4_A', 'TUITIONFEE_OUT', 'TUITFTE', 'INEXPFTE', 'AVGFACSAL']
figs, axs = plt.subplots(ncols=5, figsize=(20,5))

for ax in axs:

    ax.set_yticks([])

sns.set()

sns.distplot(df['TUITIONFEE_OUT'], ax=axs[0], color='r')

sns.distplot(df['COSTT4_A'], ax=axs[1], color='c')

sns.distplot(df['TUITFTE'], ax=axs[2], color='y')

sns.distplot(df['INEXPFTE'], ax=axs[3], color='g')

sns.distplot(df['AVGFACSAL'], ax=axs[4], color='m')
social_factors = ['UGDS', 'UGDS_WHITE', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN', 

            'UGDS_AIAN', 'UGDS_NHPI','UGDS_2MOR', 'UGDS_NRA', 'UGDS_UNKN', 'AGE_ENTRY', 'UGDS_MEN', 'UGDS_WOMEN'

           , 'FEMALE', 'MARRIED', 'DEPENDENT', 'VETERAN', 'FIRST_GEN', 'UG25ABV', 'PAR_ED_PCT_MS', 

            'PAR_ED_PCT_PS', 'PAR_ED_PCT_HS', 'PPTUG_EF', 'PFTFTUG1_EF', 'UGNONDS']
family_income = ['FAMINC', 'MD_FAMINC', 'FAMINC_IND', 'DEP_INC_AVG', 'IND_INC_AVG', 

                 'PCTPELL', 'PCTFLOAN', 'INC_PCT_LO','DEP_STAT_PCT_IND','DEP_INC_PCT_LO','IND_INC_PCT_LO',

                'INC_PCT_M1','INC_PCT_M2','INC_PCT_H1','INC_PCT_H2','DEP_INC_PCT_M1',

                 'DEP_INC_PCT_M2','DEP_INC_PCT_H1','DEP_INC_PCT_H2','IND_INC_PCT_M1','IND_INC_PCT_M2','IND_INC_PCT_H1',

                 'IND_INC_PCT_H2']
debt_statistics = []

keys = ['DEBT', 'RPY']

for name in df.columns:

    for key in keys:

        if key in name:

            debt_statistics.append(name)
attendance_statistics = []

keys = ['ENRL', 'COMP', 'WDRAW']

for name in df.columns:

    for key in keys:

        if key in name:

            attendance_statistics.append(name)
degree_types = list(df.columns[15:53])
total = social_factors + family_income + debt_statistics + attendance_statistics + degree_types
features = {'Social Factors':social_factors, 'Family Income':family_income, 

            'Debt Statistics':debt_statistics, 'Attendance Statistics':attendance_statistics, 

            'Degree Types':degree_types}
for name, items in features.items():

    print('Length of ' + name + ' = ' + str(len(items)))
analysis = feature_compare_df(df, features, target_features)

analysis
analysis
percent_analysis = analysis.copy()

for column in analysis.columns:

    percent_analysis[column] = percent_analysis[column] / percent_analysis.loc['StdDev'][column] * 100
percent_analysis[:5]
num_features = 25

rfe = RFE(linear_model.Ridge(alpha=.6), num_features)

rfe = rfe.fit(df[attendance_statistics], df['TUITIONFEE_OUT'])

list_a = attendance_statistics

fil = list(rfe.support_)

attendance_compressed = list(compress(list_a, fil))
num_features = 25

rfe = RFE(linear_model.Ridge(alpha=.6), num_features)

rfe = rfe.fit(df[debt_statistics], df['TUITIONFEE_OUT'])

list_a = debt_statistics

fil = list(rfe.support_)

debt_compressed = list(compress(list_a, fil))
compressed_features = {'Social Factors':social_factors, 'Family Income':family_income, 

            'Debt Compressed':debt_compressed, 'Attendance Compressed':attendance_compressed, 

            'Degree Types':degree_types}
compressed_analysis = feature_compare_df(df, compressed_features, target_features)
compressed_analysis
compressed_percent_analysis = compressed_analysis.copy()

for column in compressed_analysis.columns:

    compressed_percent_analysis[column] = compressed_percent_analysis[column] / compressed_percent_analysis.loc['StdDev'][column] * 100
compressed_percent_analysis[:5]
epoch_test(df[attendance_compressed], df['TUITIONFEE_OUT'], graph_output = True)