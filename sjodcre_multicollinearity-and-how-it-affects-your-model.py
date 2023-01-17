# 1.Correlation Matrix

# import pandas as pd

# import seaborn as sns



# df = pd.read_csv('data.csv')



# sns.heatmap(df.drop('target', axis=1), cmap='cool', annot=True)
# 2. Variance Inflation Factor (VIF)

# import pandas as pd

# import matplotlib.pyplot as plt

# import statsmodels.formula.api as smf

# from statsmodels.stats.outliers_influence import variance_inflation_factor



# data = pd.read_csv('dara.csv', index_col=0)

# y = df['target'] # dependent variable

# X = df.drop('target', axis=1)



# vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



# print(vif)