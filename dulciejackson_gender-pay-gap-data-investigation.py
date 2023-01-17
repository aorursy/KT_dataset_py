# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime as dt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import the dataset

pay_gap_data_2017_18 = pd.read_csv('../input/uk-gender-pay-gap-data-2019-to-2020/UK Gender Pay Gap Data - 2017 to 2018.csv', parse_dates=True)

pay_gap_data_2018_19 = pd.read_csv('../input/uk-gender-pay-gap-data-2019-to-2020/UK Gender Pay Gap Data - 2018 to 2019.csv', parse_dates=True)

pay_gap_data_2019_20 = pd.read_csv('../input/uk-gender-pay-gap-data-2019-to-2020/UK Gender Pay Gap Data - 2019 to 2020.csv', parse_dates=True)

pay_gap_data_2020_21 = pd.read_csv('../input/uk-gender-pay-gap-data-2019-to-2020/UK Gender Pay Gap Data - 2020 to 2021.csv', parse_dates=True)



pay_gap_data = pd.concat([pay_gap_data_2017_18, pay_gap_data_2018_19, pay_gap_data_2019_20, pay_gap_data_2020_21])



pay_gap_data.head()
# Remove rows with no SIC codes

cleaned_sic_codes_data = pay_gap_data.dropna(subset=['SicCodes'])



# Create dataset of all companies with SIC code 62020 (Information technology consultancy activities)

it_consultancy_data = cleaned_sic_codes_data[cleaned_sic_codes_data['SicCodes'].str.contains('62020')]



it_consultancy_data.head()
it_consultancy_data.loc[:,'DueDate'] = pd.to_datetime(it_consultancy_data['DueDate'])

it_consultancy_data.loc[:,'ReportingYear'] = it_consultancy_data['DueDate'].dt.year
# Set the width and height of the figure

plt.figure(figsize=(16,6))



sns.lineplot(data=it_consultancy_data, x="ReportingYear", y="DiffMeanHourlyPercent", hue="EmployerSize", ci=None)