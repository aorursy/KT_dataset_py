# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
%matplotlib inline
#load the data
loans = pd.read_csv('../input/prosper-loan/prosperLoanData.csv')
#view the data shape and features
print(loans.shape)
print(loans.dtypes)
print(loans.head(10))
loans = loans[['Term', 'LoanStatus','BorrowerAPR','BorrowerRate','LenderYield','EstimatedEffectiveYield'
,'EstimatedLoss','EstimatedReturn','IncomeRange'
,'Recommendations','Investors']]
#view the data shape and features
print(loans.shape)
print(loans.dtypes)
print(loans.head(10))
loans.info()
loans['EstimatedEffectiveYield'].fillna(loans['EstimatedEffectiveYield'].mean(), inplace = True)
loans['EstimatedLoss'].fillna(loans['EstimatedLoss'].mean(), inplace = True)
loans['EstimatedReturn'].fillna(loans['EstimatedReturn'].mean(), inplace = True)
loans.drop_duplicates(inplace=True)
loans.info()
base_color = sb.color_palette()[0];
sb.distplot(loans['BorrowerRate'], kde = True);

sb.distplot(loans['BorrowerAPR'], kde = True)
pltData = loans['LoanStatus'].value_counts()
plt.bar(pltData.index, pltData)
plt.xticks(rotation = 90);
sb.distplot(loans['LenderYield'], kde = False)
sb.distplot(loans['EstimatedEffectiveYield'], kde = False)
sb.distplot(loans['EstimatedLoss'], kde = False)
sb.distplot(loans['EstimatedReturn'], kde = False)
pltData = loans['IncomeRange'].value_counts()
plt.bar(pltData.index, pltData)
plt.xticks(rotation = 90);
sb.countplot(loans['Recommendations'], color = base_color);
bins = 10**np.arange(0, np.log10(loans['Investors'].max())+0.2, 0.2);
plt.hist(loans['Investors'], bins = bins);
ticks = np.arange(0, np.log10(loans['Investors'].max())+0.5, 0.5);
plt.xscale('log');
labels = [1, 3, 10, 30, 100, 300, 1000, 3000]
plt.xticks(10**ticks, labels);
sb.countplot(loans['Term'], color = base_color);
pd.plotting.scatter_matrix(loans,figsize = [21,21]);
sb.boxplot(data = loans, x = 'Term', y = 'BorrowerAPR', color = base_color);
sb.boxplot(data = loans, x = 'Term', y = 'BorrowerRate', color = base_color);
sb.boxplot(data = loans, x = 'LoanStatus', y = 'BorrowerRate', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'LoanStatus', y = 'BorrowerAPR', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'IncomeRange', y = 'BorrowerRate', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'IncomeRange', y = 'BorrowerAPR', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'Term', y = 'EstimatedLoss', color = base_color);
sb.boxplot(data = loans, x = 'Term', y = 'EstimatedReturn', color = base_color);
sb.boxplot(data = loans, x = 'IncomeRange', y = 'EstimatedReturn', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'IncomeRange', y = 'EstimatedLoss', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'LoanStatus', y = 'EstimatedLoss', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'LoanStatus', y = 'EstimatedReturn', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'LoanStatus', y = 'Investors', color = base_color);
plt.xticks(rotation = 90);
sb.boxplot(data = loans, x = 'LoanStatus', y = 'LenderYield', color = base_color);
plt.xticks(rotation = 90);
g = sb.FacetGrid(data = loans, col = 'Term', height = 3,
                margin_titles = True)
g.map(plt.scatter, 'EstimatedReturn', 'BorrowerRate');
g = sb.FacetGrid(data = loans, col = 'Term', height = 3,
                margin_titles = True)
g.map(plt.scatter, 'EstimatedLoss', 'BorrowerRate');
g = sb.FacetGrid(data = loans, col = 'Term', height = 3,
                margin_titles = True)
g.map(plt.scatter, 'LenderYield', 'BorrowerRate');
g = sb.FacetGrid(data = loans, col = 'Term', height = 3,
                margin_titles = True)
g.map(plt.scatter, 'EstimatedEffectiveYield', 'BorrowerRate');
plt.scatter(data = loans, x = 'LenderYield', y = 'BorrowerRate', c = 'EstimatedReturn')
plt.colorbar()
plt.scatter(data = loans, x = 'LenderYield', y = 'BorrowerRate', c = 'EstimatedLoss')
plt.colorbar()
plt.scatter(data = loans, x = 'LenderYield', y = 'BorrowerRate', c = 'EstimatedEffectiveYield')
plt.colorbar()
