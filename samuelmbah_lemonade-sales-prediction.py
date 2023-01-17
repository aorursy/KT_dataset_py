#Loading packages and libraries required for data analysis
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

#For data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Reading dataset
df = pd.read_csv('../input/lemona/Lemonade.csv')
df.head()
#Creating new column (Revenue)
df['Revenue'] = df['Price'] * df['Sales']
df.head()
# Print desciptive statistics
df.describe()
df.describe(include='O')
df.info()
df.count()
# Data Visualization
sns.pairplot(df)
#Comparing weekdays Sales
sns.barplot(x='Day',y='Sales',data=df)
df.groupby('Day',as_index=False).Sales.mean()
#Grasping important columns
df1 = df[['Sales', 'Temperature', 'Rainfall', 'Flyers', 'Price', 'Revenue']]
df1.head()
#Find the correlation between the variables in the dataset.Export to excel to check for muticollinearity
df1.corr()
# Checking for Outlier in Flyers
sns.boxplot(data=df1, x=df1['Flyers'])
# Treating Flyers Outlier
Q1 = df1['Flyers'].quantile(0.25)
Q3 = df1['Flyers'].quantile(0.75)
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)

Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)

df1 = df1[df1['Flyers'] < Upper_Whisker]

df1.shape

sns.boxplot(data = df1, x = df1['Flyers'])
df1.head()
from sklearn import model_selection
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
X = np.array(df1.drop(['Sales'], 1))
y = np.array(df1['Sales'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
#y_pred = lr.predit(X_test)

lr.score(X_test, y_test)

acc_lr = round(lr.score(X_test, y_test)*100,2)
#Stochastic Gradient Descent (SGD):
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
#y_pred = sgd.predict(X_test)

sgd.score(X_test, y_test)

acc_sgd = round(sgd.score(X_test, y_test) * 100, 2)
#Random Forest:
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#y_pred = random_forest.predict(X_test)

random_forest.score(X_test, y_test)

acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
#Gaussian Naive Bayes:
gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)  
#y_pred = gaussian.predict(X_test) 

gaussian.score(X_test, y_test)

acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
#Perceptron:
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, y_train)
#y_pred = perceptron.predict(X_test)

perceptron.score(X_test, y_test)

acc_perceptron = round(perceptron.score(X_test, y_test) * 100, 2)
#Linear Support Vector Machine:
svm = LinearSVC()
svm.fit(X_train, y_train)
#y_pred = svm.predict(X_test)

svm.score(X_test, y_test)

acc_svm = round(svm.score(X_test, y_test) * 100, 2)
#Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
#y_pred = decision_tree.predict(X_test)  

decision_tree.score(X_test, y_test)

acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
#Gradient Boost Classifier
gbk = GradientBoostingClassifier()
ne = np.arange(1,20)
dep = np.arange(1,10)
param_grid = {'n_estimators' : ne,'max_depth' : dep}
gbk_cv = GridSearchCV(gbk, param_grid=param_grid, cv=5)
gbk_cv.fit(X, y)
#y_pred = gbk_cv.predict(X_test)

gbk_cv.score(X_test, y_test)

acc_gbk = round(gbk_cv.score(X_test, y_test)*100, 2)
#Which is the best Model ?
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Stochastic Gradient Decent', 'Random Forest', 'Gaussian Naive Bayes','Perceptron', 
              'Support Vector Machines', 'Decision Tree', 'Gradient Boost Classifier'],
    'Score': [acc_lr, acc_sgd, acc_random_forest, acc_gaussian, acc_perceptron, acc_svm, acc_decision_tree, acc_gbk]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(8)
y_pred = gbk_cv.predict(X_test)
plt.scatter(y_test, y_pred)
sales_pred = pd.DataFrame({'Sales': y_test, 'pred_sales': y_pred})
print(sales_pred)
# Converting rest to excel.csv file

sales_pred = pd.DataFrame({'Sales': y_test, 'pred_sales': y_pred}).to_csv('Lemonade_Sales_Pred.csv')
