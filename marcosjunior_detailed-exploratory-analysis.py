import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf

import seaborn as sns

import os



df = pd.read_csv("../input/insurance.csv")

pd.options.display.max_columns = None
df.info()
df.describe()
df['bmi_desc'] = 'default value'

df.loc[df.bmi >= 30, 'bmi_desc'] = 'Obesity'

df.loc[df.bmi <18.5, 'bmi_desc'] = 'Underweight'

df.loc[ (df.bmi >= 18.5) & (df.bmi < 24.9), 'bmi_desc'] = 'Normal'

df.loc[ (df.bmi > 24.9) & (df.bmi <30), 'bmi_desc'] = 'Overweight'

df.head()
df['bmi_smoker'] = 'default value'

df.loc[(df.bmi_desc == 'Obesity') & (df.smoker == 'yes'), 'bmi_smoker'] = 'obese_smoker'

df.loc[(df.bmi_desc == 'Obesity') & (df.smoker == 'no'), 'bmi_smoker'] = 'obese_no_smoker'

df.loc[(df.bmi_desc != 'Obesity') & (df.smoker == 'yes'), 'bmi_smoker'] = 'other_smoker'

df.loc[(df.bmi_desc != 'Obesity') & (df.smoker == 'no'), 'bmi_smoker'] = 'other_no_smoker'

df.head()
sns.distplot(df[['charges']])
sns.distplot(df.loc[df.smoker == 'yes']['charges'].values.tolist())
sns.distplot(df.loc[df.smoker == 'no']['charges'].values.tolist())
sns.distplot(df[['age']])
sns.scatterplot(x= df['age'] , y= df['charges']).set_title('3.1 - Charges vs Age without filter')

sns.despine()
sns.scatterplot(x= df['age'] , y= df['charges'], hue='sex', data=df).set_title('3.1 - Charges vs Age filter by sex')

sns.despine()
sns.scatterplot(x= df['age'] , y= df['charges'], hue='children', data=df).set_title('3.2 - Charges vs Age filtered by the number of children')

sns.despine()
sns.scatterplot(x= df['age'] , y= df['charges'], hue='region', data=df).set_title('3.3 - Charges vs Age filtered by region')

sns.despine()
sns.scatterplot(x= df['age'] , y= df['charges'], hue='bmi_desc', data=df).set_title('3.4 - Charges vs Age filtered by bmi description')

sns.despine()
sns.scatterplot(x= df['age'] , y= df['charges'], hue='smoker', data=df).set_title('3.5 - Charges vs Age filtered by smoker')

sns.despine()
sns.scatterplot(x= df['age'] , y= df['charges'], hue='bmi_smoker', data=df).set_title('3.6 - Charges vs Age filtered by bmi_smoker')

sns.despine()
sns.distplot(df[['bmi']])
sns.scatterplot(x= df['bmi'] , y= df['charges']).set_title('4.1 - Charges vs Bmi')

sns.despine()
sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='sex', data=df).set_title('4.2 - Charges vs Bmi filtered by sex')

sns.despine()
sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='region', data=df).set_title('4.3 - Charges vs Bmi filtered by region')

sns.despine()
sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='children', data=df).set_title('4.4 - Charges vs Bmi filtered by number of children')

sns.despine()
sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='smoker', data=df).set_title('4.5 - Charges vs Bmi filtered by smoker_bmi')

sns.despine()