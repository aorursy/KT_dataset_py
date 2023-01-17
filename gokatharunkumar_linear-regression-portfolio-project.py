import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
df.head().transpose()
df.columns
cols=df.iloc[:,3:]

cols

rows=df.iloc[:,3:]

rows

from scipy.stats import pearsonr

for x in rows.columns:

    for y in cols.columns:

        if x!=y:

            print("R("+x+","+y+")is:"+"  "+(str(pearsonr(df[x],df[y])[0])))
df.dtypes
df.info()
df.describe()
sns.pairplot(df,diag_kind='kde')
sns.heatmap(df.corr(),cmap = 'Blues', annot=True)
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

y = df['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
lm.coef_
pred = lm.predict(X_test)
plt.scatter(y_test,pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print("Performance of the model: "+str(metrics.r2_score(y_test,pred)*100))
import statsmodels.formula.api as smf

ecommerce=pd.concat([X_train,y_train] , axis=1)

ecommerce=ecommerce.rename(columns={'Avg. Session Length':'Avg_Session_Length','Time on App':'Time_on_App','Time on Website':'Time_on_Website','Length of Membership':'Length_of_Membership','Yearly Amount Spent':'Yearly_Amount_Spent'})

ecommerce
linearmodel=smf.ols(formula='Yearly_Amount_Spent ~ Avg_Session_Length + Time_on_App + Time_on_Website + Length_of_Membership',data=ecommerce)

linearmodelfit=linearmodel.fit()
linearmodelfit.params
print(linearmodelfit.summary())
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
intercept= lm.intercept_

intercept