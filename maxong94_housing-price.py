import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.stats as st
housing = pd.read_csv("/kaggle/input/house/USA_Housing.csv")
housing.head(5)
housing.dtypes
quantitative = [f for f in housing.columns if housing.dtypes[f] != "object"]
quantitative.remove("Price")
quantitative
sns.set()
sns.pairplot(housing)
# shows the pairwise relationship between features and target variable, kinda useful.
# if need more customization, go to pairgrid
sns.heatmap(housing.corr(), annot = True)
sns.distplot(housing["Price"])
# visual test does pass normality, but let us do the normal test to be sure
stats, p = st.normaltest(housing["Price"])
alpha = 0.05 
# H0 is that distribution is Gaussian, so if p < 0.05,
# reject null hypothesis 
if p < alpha: 
    print("Reject H0, not Gaussian")
else: 
    print("Do not reject H0, is Gaussian")
X = housing[quantitative]
y = housing["Price"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size =  0.4, 
                                                 random_state = 42)
# random_state parameter --> provides a seed value 
# to function's internal random number generator
# Will result in different train test splits if we give it different number. 
# So just use a value and stick to the value to get the same train test split

from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
model = LinearRegression()
cv_scores = cross_val_score(model, X_train, y_train, cv = 5)
print("CV scores (5-fold) :", "\n", cv_scores)
print("Mean-CV scores :" ,np.mean(cv_scores),"\n A measure of how accurate the model is on average")

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
model.fit(X_train,y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
print("R-square Train ", r2_score(y_train,y_train_predict))
print("R-square Test ", r2_score(y_test,y_test_predict))

##