import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
HousePrices = pd.read_csv('../input/train.csv')
#Code inspired by: https://www.kaggle.com/nhirons/simple-visualizations-of-the-impact-of-null-values
Null = HousePrices.isnull()
Null = Null[[col for col in Null.columns if Null[col].sum() > 0]]
plt.figure(figsize = (30,10))
plt.suptitle('Amount of Null Values by Column', fontsize = 24)
plt.xlabel('Housing Feature', fontsize = 20)
plt.ylabel("Number of Null Values", fontsize = 20)
plt.bar(x = Null.columns, height = Null.sum())
plt.show()
HousePrices.fillna('None', inplace = True)
Correlation = HousePrices.corr()
#https://stackoverflow.com/questions/39409866/correlation-heatmap
plt.figure(figsize=(16,10))
sns.heatmap(Correlation, 
        xticklabels=Correlation.columns,
        yticklabels=Correlation.columns)
#https://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe/32011969
Categorical = HousePrices.select_dtypes(['object']).astype('category')
Categorical = pd.get_dummies(Categorical)
Categorical['SalePrice'] = HousePrices['SalePrice']
CategoricalCorrelation = Categorical.corr()
plt.figure(figsize=(100,100))
sns.heatmap(CategoricalCorrelation, 
        xticklabels=CategoricalCorrelation.columns,
        yticklabels=CategoricalCorrelation.columns)