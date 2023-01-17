# Bare minimum required imports

import numpy as np

import pandas as pd
!pip install kesh-utils
# Import the chartil from Kutils 

from KUtils.eda import chartil
# Load the dataset

heart_disease_df = pd.read_csv('../input/heart.csv')
heart_disease_df.head(10)

heart_disease_df.info()

heart_disease_df.describe()

heart_disease_df.shape
# Null checks

heart_disease_df.isnull().sum() # No null found
# Number of unique values in each column 

{x: len(heart_disease_df[x].unique()) for x in heart_disease_df.columns}
# Quick data preparation, convert few to categorical column and add new age_bin

heart_disease_df['target'].describe()

heart_disease_df['target'] = heart_disease_df['target'].astype('category')

heart_disease_df['age'].describe()

heart_disease_df['age_bin'] = pd.cut(heart_disease_df['age'], [0, 32, 40, 50, 60, 70, 100], 

                labels=['<32', '33-40','41-50','51-60','61-70', '71+'])



heart_disease_df['sex'].describe()

heart_disease_df['sex'] = heart_disease_df['sex'].map({1:'Male', 0:'Female'})



heart_disease_df['cp'].describe()

heart_disease_df['cp'] = heart_disease_df['cp'].astype('category')



heart_disease_df['trestbps'].describe()

heart_disease_df['chol'].describe()



heart_disease_df['fbs'] = heart_disease_df['fbs'].astype('category')

heart_disease_df['restecg'] = heart_disease_df['restecg'].astype('category')



heart_disease_df['thalach'].describe()



heart_disease_df['exang'] = heart_disease_df['exang'].astype('category')

heart_disease_df['oldpeak'].describe()



heart_disease_df['slope'] = heart_disease_df['slope'].astype('category')

heart_disease_df['ca'] = heart_disease_df['ca'].astype('category')

heart_disease_df['thal'] = heart_disease_df['thal'].astype('category')



heart_disease_df.info()
import warnings  

warnings.filterwarnings('ignore')
# Univariate Categorical variable

chartil.plot(heart_disease_df, ['target'])
# Univariate Numeric/Continuous variable

chartil.plot(heart_disease_df, ['trestbps'])
# Smae as above, but force to use barchart on numeric/Continuous (Automatically creates 10 equal bins)

chartil.plot(heart_disease_df, ['age'], chart_type='barchart')
# Age doesn't look normal with auto bin barchart, instead use age_bin column to plot the same

chartil.plot(heart_disease_df, ['age_bin'])
chartil.plot(heart_disease_df, ['age_bin'], 

             optional_settings={'sort_by_value':True})
chartil.plot(heart_disease_df, ['age_bin'], 

             optional_settings={'sort_by_value':True, 'limit_bars_count_to':5})
chartil.plot(heart_disease_df, ['trestbps'], chart_type='distplot')
# Bi Category vs Category (+ Univariate Segmented)

chartil.plot(heart_disease_df, ['sex', 'target'])
chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='crosstab')
chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='stacked_barchart')
# Bi Continuous vs Continuous (Scatter plot)

chartil.plot(heart_disease_df, ['chol', 'thalach'])
# Bi Continuous vs Category

chartil.plot(heart_disease_df, ['thalach', 'sex'])

# Same as above, but use distplot

chartil.plot(heart_disease_df, ['thalach', 'sex'], chart_type='distplot')
# Multi variavte - 3D view of 3 Continuous variables coloured by the same contious varibale amplitude in RGB form

chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps'])
# Multi 2 Continuous, 1 Category

chartil.plot(heart_disease_df, ['chol', 'thalach', 'target'])
# Multi 1 Continuous, 2 Category

chartil.plot(heart_disease_df, ['thalach', 'cp', 'target'])
# Same as above, but use violin plot

chartil.plot(heart_disease_df, ['thalach', 'sex', 'target'], chart_type='violinplot')
# Multi 3D view of 3 Continuous variable and color it by target/categorical feature

chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps', 'target'])
# Heatmap (Send list of all columns, it will plot the co-relation matrix of all numerical/continuous variables)

chartil.plot(heart_disease_df, heart_disease_df.columns)
# If you want sort the corelation based on one specific columns.

chartil.plot(heart_disease_df, heart_disease_df.columns, optional_settings={'sort_by_column':'thalach'})
# Include categorical variables - Internally creates dummies

chartil.plot(heart_disease_df, heart_disease_df.columns, optional_settings={'include_categorical':True} )
# If you want sort the corelation based on one specific columns. 

# Below example will sort the feature co-relation by feature 'trestbps'

chartil.plot(heart_disease_df, heart_disease_df.columns, 

             optional_settings={'include_categorical':True, 'sort_by_column':'trestbps'} )