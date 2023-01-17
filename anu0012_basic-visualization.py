# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loans = pd.read_csv('../input/kiva_loans.csv')
loan_themes_by_region = pd.read_csv('../input/loan_themes_by_region.csv')
kiva_mpi_region_locations = pd.read_csv('../input/kiva_mpi_region_locations.csv')
loan_theme_ids = pd.read_csv('../input/loan_theme_ids.csv')
kiva_loans.head()
plt.figure(figsize=(12,8))
plt.scatter(range(len(kiva_loans['loan_amount'])),np.sort(kiva_loans['loan_amount'].values))
plt.xlabel("index")
plt.ylabel("Loan in USD")
plt.show()
plt.figure(figsize=(12,8))
plt.scatter(range(len(kiva_loans['funded_amount'])),np.sort(kiva_loans['funded_amount'].values))
plt.xlabel("index")
plt.ylabel("Fund amount in USD")
plt.show()
loans_without_outlier = kiva_loans[kiva_loans['loan_amount'] < 20000.0]
funds_without_outlier = kiva_loans[kiva_loans['funded_amount'] < 20000.0]
plt.figure(figsize=(12,8))
sns.distplot(loans_without_outlier.loan_amount.values,bins=50)
plt.xlabel("Loan amount")
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(funds_without_outlier.loan_amount.values,bins=50)
plt.xlabel("Fund amount")
plt.show()
kiva_loans['loan_amount'].sum()
kiva_loans['funded_amount'].sum()
plt.figure(figsize=(12,8))
plt.scatter(range(len(kiva_loans['lender_count'])),np.sort(kiva_loans['lender_count'].values))
plt.xlabel("index")
plt.ylabel("Lender Count")
plt.show()
top_lender = kiva_loans['lender_count'].value_counts().head(30)
plt.figure(figsize=(12,8))
sns.barplot(top_lender.index, top_lender.values)
plt.xlabel("Lender Count(Number of lenders)")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(12,6))
kiva_loans['sector'].value_counts().head(10).plot.bar()
plt.figure(figsize=(12,6))
kiva_loans['activity'].value_counts().head(10).plot.barh()
plt.figure(figsize=(12,6))
kiva_loans['country'].value_counts().head(10).plot.barh()
plt.figure(figsize=(12,6))
kiva_loans['region'].value_counts().head(10).plot.barh()
male_count = 0
female_count = 0
for i in kiva_loans['borrower_genders']:
    li = str(i).split(',')
    for j in li:
        if j.strip() == 'female':
            female_count += 1
        else:
            male_count += 1
[male_count, female_count]
sns.barplot(x=['male', 'female'], y=[male_count, female_count],
            label="Total", color="b")
plt.figure(figsize=(12,6))
kiva_loans['repayment_interval'].value_counts().head().plot.bar()
plt.figure(figsize=(12,6))
kiva_loans['date'].value_counts().head().plot.barh()
use = kiva_loans["use"][~pd.isnull(kiva_loans["use"])]
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(use))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#Correlation Matrix
corr = kiva_loans.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='cubehelix', square=True)
plt.title('Correlation between different features')
corr
loan_themes_by_region.head()
loan_theme_ids.head()
kiva_mpi_region_locations.head()
kiva_loans.isnull().sum(axis=0)
loan_theme_ids.isnull().sum(axis=0)
loan_themes_by_region.isnull().sum(axis=0)
kiva_mpi_region_locations.isnull().sum(axis=0)
top_mpi = kiva_mpi_region_locations['MPI'].value_counts().head()
plt.figure(figsize=(12,8))
top_mpi.plot.bar()
#sns.barplot(kiva_mpi_region_locations.index, kiva_mpi_region_locations.values)
plt.xlabel("Top MPI")
plt.ylabel("Count")
plt.show()