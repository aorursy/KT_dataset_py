import numpy as np
import pandas as pd
from patsy import dmatrices #generate data for model training
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
# from Jiuzhang ML Project
data = pd.read_csv('../input/HR_comma_sep.csv')
print(data.dtypes)
print(data.shape)# look each column and data type of each column; object means string here
# observe the relationship between left and salary, as example of discrete variable
pd.crosstab(data.salary, data.left).plot(kind = 'bar')
plt.show()
# we see left = 0 is much more than left=1, we can also see percentages in the next line
# we actually want to see the percentage difference between left and not left, in each salary category
q = pd.crosstab(data.salary, data.left)
print(q.sum(1))
q.div(q.sum(1), axis = 0).plot(kind = 'bar', stacked=True)
plt.show()
# in high and medium, employees tend to stay, but in low salary, employees tend to leave
# for this dataset, we only have salary in low, med, and high; it would be more precise if we have salary numbers
q
# now look at distribution of satisfaction level, eample of continuous variable
data[data.left == 0].satisfaction_level.hist()
plt.show()
# Foe people who stay, their satisfaction is high. Cutoff is clearly at 0.5.
data[data.left == 1].satisfaction_level.hist()
plt.show()
# For people who left, many are not satisfied. Not clear cutoff.
# look work injuries
pd.crosstab(data.Work_accident, data.left).plot(kind = 'bar', stacked=True)
plt.show()
# look average monthly hours
pd.crosstab(data.average_montly_hours, data.left).plot(kind = 'hist',stacked=True)
plt.show()
# how to explain this? what does 0 hours mean??? Is this feature sensible?
# look at time spent company
pd.crosstab(data.time_spend_company, data.left).plot(kind = 'hist', stacked = True)
plt.show()
# look at promotion in last 5 years
pd.crosstab(data.promotion_last_5years, data.left).plot(kind = 'bar', stacked = False)
plt.show()
# look at sales
pd.crosstab(data.sales, data.left).plot(kind = 'bar', stacked = False)
plt.show()
# my guess is that sales means deparment here


# now start to use model
model = LogisticRegression()
data.dtypes
# y is label, X is dimension
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = X.rename(columns = {
    'C(sales)[T.RandD]': 'Department: Random',
    'C(sales)[T.accounting]': 'Department: Accounting',
    'C(sales)[T.hr]': 'Department: HR',
    'C(sales)[T.management]': 'Department: Management',
    'C(sales)[T.marketing]': 'Department: Marketing',
    'C(sales)[T.product_mng]': 'Department: Product_Management',
    'C(sales)[T.sales]': 'Department: Sales',
    'C(sales)[T.support]': 'Department: Support',
    'C(sales)[T.technical]': 'Department: Technical',
    'C(salary)[T.low]': 'Salary: Low',
    'C(salary)[T.medium]': 'Salary: Medium'}) 

# y must be number vector
y = np.ravel(y)
y
sum(y)
model.fit(X, y)
print(model.score(X, y))

1 - sum(y) / len(y) # what if we predict 1 for all cases
# is this unbalanced data?
# yes it is, we can add weight to certain features
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)
model2 = LogisticRegression()
model2.fit(Xtrain, ytrain)
pred = model2.predict(Xtest)
metrics.accuracy_score(ytest, pred)
print(metrics.classification_report(ytest, pred))
print(cross_val_score(LogisticRegression(C=1e5), X, y, scoring = 'accuracy', cv=10).mean())
# note this accuracy will be worse than mnodel1 because here we use cross validation to prevent overfitting
