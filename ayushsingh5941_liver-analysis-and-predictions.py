# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

import category_encoders as ce

import xgboost as xgb
df_train = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')

pd.set_option('display.max_columns', 90)

pd.set_option('display.max_rows', 90)

# ===================================________Data Exploration________==================================================





def data_exploration(data):

    """

    Understanding data to make better feature engineering

    :param data: Data to be explored

    :return: None

    """

    # ============______Basic FAMILIARIZATION________==================

    print('______________DATA HEAD__________ \n', data.head())

    print('______________DATA DESCRIBE______ \n', data.describe())

    print('______________DATA INFO__________ \n', data.info())



    # ===========_______DATA FREQUENT TERM___________===================

    print('_____________Total unique values in data_______ \n', data.nunique())

    print('___________________ DATA UNIQUE VALUES_____________ \n')

    print('\n', [pd.value_counts(data[cols]) for cols in data.columns], '\n')



    # ===========_______DATA CORRELATION_____________====================

    corr_mat_graph(data, 'EDA MATRIX')



    # =================____________DISTRIBUTION VISUALIZATION_________=================

    dist_plot(data)



    # ======================___________ Outliers__________________======================

    box_plot(data)



# ================================___________GRAPHS FUNCTIONS____________==============================================





def corr_mat_graph(data, title):

    """

    function to plot correlation matrix for better understanding of data

    :param data: correlation matrix

    :param title: Title of the graph

    :return: None

    """

    print('\n \n ____________________CORRELATION MATRIX_______________ \n \n')

    corr_matrix = data.corr()

    corr_matrix_salePrice = corr_matrix['Dataset'].sort_values(ascending=False)

    print('________CORRELATION MATRIX BY DATA SET________ \n', corr_matrix_salePrice)

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(corr_matrix, square=False, linewidths=0.5, ax=ax, vmax=0.8, vmin=0.42, annot=True)

    ax.title.set_text(title)





def dist_plot(data):

    """

    Function to plot subplots of distribution for numerical data

    :param data: data which needs to be plotted

    :return: None

    """

    print('\n \n ________________________DISTRIBUTION PLOT___________________ \n \n')

    # Plotting numerical graph

    data = data.select_dtypes(exclude='object')

    data_filed = data.dropna(axis=0)



    for cols in data.columns:

        fig, ax = plt.subplots()

        sns.distplot(data_filed[cols])

        ax.title.set_text(cols)





def box_plot(data):

    """

    To find oultliers in the data

    :param data: data to be plot

    :return:

    """

    print('\n \n ________________________BOX PLOT___________________ \n \n')

    data = data.select_dtypes(exclude='object')

    for cols in data.columns:

        fig, ax = plt.subplots()

        sns.boxplot(data[cols])

        ax.title.set_text(cols)



# =============================_____________________FEATURE ENGINEERING_________________===============================





def feature_engineering(data):

    """

    To clean and add features in dataset

    :param data: Dataset to be cleaned

    :return: cleaned dataset

    """

    print('\n \n ________________FEATURE ENGINEERING_________________ \n \n')

    # =====================__________________OUTLIERS________________==========================

    # We need to deal with outliers

    # To many outliers in Total_Bilirubin should drop whole column after calculating indirect Bilirubin

    # Direct_Bilirubin have many outliers after outliers 85, eliminating such outliers

    data = data.drop(data[data.Direct_Bilirubin > data.Direct_Bilirubin.quantile(0.85)].index)



    # Alkaline Phosphate have many outliers after outliers 85, eliminating such outliers

    data = data.drop(data[data.Alkaline_Phosphotase > data.Alkaline_Phosphotase.quantile(0.82)].index)



    # Alamine Aminotransferase has heavy outliers after 93% quantile, eliminating such outliers

    data = data.drop(data[data.Alamine_Aminotransferase > data.Alamine_Aminotransferase.quantile(0.93)].index)



    # Alamine Aminotransferase has heavy outliers after 93% quantile, eliminating such outliers

    data = data.drop(data[data.Aspartate_Aminotransferase > data.Aspartate_Aminotransferase.quantile(0.92)].index)



    # All the major outliers are taken care of but Total and Direct Bilirubin is still heavily right skewed.

    # Further removal of data will decrease data size



    # =============================____________________IMPUTING MISSING VALUES_________________=================

    # Since all features are numerical except Gender we need to drop rows where Gender.

    # Fill NA of numerical data with median as dataset have way too much outliers

    data['Gender'].dropna(axis=0, inplace=True)

    data.fillna(data.median(), inplace=True)



    # ===========================_____________________ADDING NEW FEATURES_______________________================

    # Indirect Bilirubin is calculated not tested

    data['Indirect_Bilirubin'] = data['Total_Bilirubin'] - data['Direct_Bilirubin']



    # Normal and high Bilirubin level in Total Bilirubin can be grouped together

    data['TotalBilirubinGroup'] = data['Total_Bilirubin'].apply(lambda x: 'Normal' if x <= 1.2 else 'High')



    # Normal and high Bilirubin level in Direct Bilirubin can be grouped together

    data['DirectBilirubinGroup'] = data['Direct_Bilirubin'].apply(lambda x: 'Normal' if x <= 0.3 else 'High')



    # Low, normal and high Bilirubin level in Indirect Bilirubin can be grouped together

    data['IndirectBilirubinGroup'] = data['Indirect_Bilirubin'].apply(lambda x: 'Low' if x < 0.3

    else ('Normal' if 0.3 <= x <= 1.0 else 'High'))



    # Alkaline phosphotase levels in high and low bins

    data['Alkaline_PhosphotaseGroup'] = data['Alkaline_Phosphotase'].apply(

        lambda x: 'Low' if x < 20.0 else ('Normal' if 20.0 <= x <= 140.0 else 'High'))



    # Alamine Aminotransferase levels in high and low bins

    data['Alamine_AminotransferaseGroup'] = data['Alamine_Aminotransferase'].apply(lambda x: 'Low' if x < 20.0

    else ('Normal' if 20.0 <= x <= 60.0 else 'High'))



    # Aspartate Aminotransferase (Male) levels

    data.loc[(data['Gender'] == 'Male'), 'AspartateLevel'] = data['Aspartate_Aminotransferase'].apply(

        lambda x: 'Low' if x < 6

        else ('Normal' if 6 <= x <= 34 else 'High'))

    # Aspartate Aminotransferase (FEMALE)

    data.loc[(data['Gender'] == 'Female'), 'AspartateLevel'] = data['Aspartate_Aminotransferase'].apply(

        lambda x: 'Low' if x < 8 else ('Normal' if 8 <= x <= 40 else 'High'))



    # Total protein levels

    data['Total_Protiens_Level'] = data['Total_Protiens'].apply(lambda x: 'Low' if x < 6.0

    else ('Normal' if 6.0 <= x <= 8.3 else 'High'))



    # Albumin levels

    data['Albumin_Level'] = data['Albumin'].apply(lambda x: 'Low' if x < 3.4

    else ('Normal' if 3.4 <= x <= 5.4 else 'High'))



    # ===================___________________REDUCING SKEWNESS BY LOG____________====================

    numeric_cols = data.select_dtypes(exclude='object').columns

    for cols in numeric_cols:

        if cols not in ['Dataset']:

            data[cols] = np.log1p(data[cols])



    # ==================___________________VISUALIZING TRANSFORMED DATA____________==================

    dist_plot(data)

    corr_mat_graph(data, 'Feature Engineering')

    return data





# Calling data exploration and feature

data_exploration(df_train)





# ===================================_________________SPLITTING DATA______________=======================

print('___________________SPLITTING DATA________________')



x_train, x_test = train_test_split(df_train, random_state=42, test_size=0.25)



x_train = feature_engineering(x_train)

x_test = feature_engineering(x_test)

y_train = x_train['Dataset']

x_train.drop('Dataset', axis=1, inplace=True)

y_test = x_test['Dataset']

x_test.drop('Dataset', axis=1, inplace=True)



print('train data size', x_train.shape, y_train.shape)

print('Test data size', x_test.shape, y_test.shape)

# =========================__________________SCALING DATA____________====================

sc = StandardScaler()

enc = ce.OrdinalEncoder()

pipe = Pipeline(steps=[('enc', enc), ('sc', sc)])

X_train = pipe.fit_transform(x_train)

X_test = pipe.transform(x_test)



# ===========================________________Model____________________==============================

xgboost = xgb.XGBClassifier(n_jobs=-1)



grid_param = {'n_estimators': [500, 1000, 1500, 2000],

              'max_depth': [9, 10, 11],

              'learning_rate': [0.1, 0.07, 0.03, 0.01],

             'subsample': [0.5, 1.0],

             'booster' : ['dart', 'gbtree']}



# GridSearchCv and Cross Validation

grid = GridSearchCV(xgboost, grid_param, cv=2, scoring='roc_auc')

grid.fit(X_train, y_train)

print('Best Params', grid.best_params_)

model = grid.best_estimator_



# Predicting

predict = model.predict(X_test)

predictions = cross_val_predict(model, X_test, y_test, cv=2)

print(confusion_matrix(y_test, predictions))

score = np.mean(cross_val_score(model, X_test, y_test, cv=2, scoring='roc_auc'))

print(np.around(score, decimals=4))



plt.show()
