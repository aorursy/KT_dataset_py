import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
%matplotlib inline
from sklearn.datasets import load_boston
boston_data = load_boston()
df =pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df.head()
X = df
y = boston_data.target
X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant)
lin_reg = model.fit()
lin_reg.summary()
f_model = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
f_lin_reg = f_model.fit()
f_lin_reg.summary()
print(lin_reg.predict(X_constant[:10]))
print(f_lin_reg.predict(X_constant[:10]))
pd.options.display.float_format = '{:,.4f}'.format
corr = df.corr()
corr[np.abs(corr) < 0.65] = 0
plt.figure(figsize=(16,10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()
from sklearn.metrics import r2_score
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
base = linear_reg.fit()
print(r2_score(y, base.predict(df)))
# WITHOUT LSTAT
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B', 
              data=df)
base = linear_reg.fit()
print(r2_score(y, base.predict(df)))
# WITHOUT AGE
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM +DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
base = linear_reg.fit()
print(r2_score(y, base.predict(df)))
