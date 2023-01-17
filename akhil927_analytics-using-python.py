# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
insurance_data = pd.read_csv('../input/insurance/insurance.csv', dtype={'age': 'float', 'children': 'float', 'bmi': 'float', 'charges': 'float'})
insurance_data.head(), insurance_data.info()
insurance_data = insurance_data.drop(columns=['region'], axis=1)
insurance_data.shape
print(insurance_data['age'].describe(percentiles=[0.1, 0.5, 1.0]))
print(insurance_data['age'].value_counts())
print(insurance_data['charges'].max(), insurance_data['charges'].min())
insurance_data['is_fit'] = np.where((insurance_data['bmi'] < 24.99) & (insurance_data['bmi'] > 18.5), 'fit', 'unfit')
insurance_data['is_fit'].describe()
insurance_data[0:10]
insurance_data = insurance_data.sort_values(['charges'])
insurance_data.head(10)
insurance_data = pd.concat([insurance_data, insurance_data[500:600]])
insurance_data = insurance_data.drop_duplicates()
insurance_data.info()
male_insurance_charges = insurance_data[insurance_data['sex'] == 'male']
female_insurance_charges = insurance_data[insurance_data['sex'] == 'female']
male_insurance_charges.head(),female_insurance_charges.head()
output_data = pd.merge(male_insurance_charges, female_insurance_charges, how='outer')
output_data.info()
output_data.to_csv('output.csv')