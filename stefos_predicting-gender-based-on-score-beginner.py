import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
students=pd.read_csv("../input/StudentsPerformance.csv")
students.head(5)
students.info()
students.describe()
students.gender.value_counts()
sns.set_context("notebook", font_scale=1.5)
sns.pairplot(students, hue="gender", height=3.5, palette='husl', diag_kind="kde",
             plot_kws=dict(s=30, linewidth=0.2))
#darkgrid, whitegrid, dark, white, ticks
sns.set_style("darkgrid")
#paper, notebook, talk, poster
sns.set_context("notebook", font_scale=1.5)
plt.figure(figsize=(12,8))
p1=sns.kdeplot(students['reading score'], shade=True, color="teal", bw=.9)
p1=sns.kdeplot(students['math score'], shade=True, color="lightslategray", bw=.9)
plt.show()
#darkgrid, whitegrid, dark, white, ticks
sns.set_style("darkgrid")
#paper, notebook, talk, poster
sns.set_context("notebook", font_scale=1.5)
plt.figure(figsize=(12,8))
p1=sns.kdeplot(students['writing score'], shade=True, color="indianred", bw=.9)
p1=sns.kdeplot(students['math score'], shade=True, color="lightslategray", bw=.9)
plt.show()
#darkgrid, whitegrid, dark, white, ticks
sns.set_style("darkgrid")
#paper, notebook, talk, poster
sns.set_context("notebook", font_scale=1.5)
ax = sns.lmplot( x="reading score", y="math score", data=students, fit_reg=False, hue='gender', height=9,
           palette="Set2", aspect=1.3, legend=False,
           scatter_kws={"alpha":0.5,"s":50})
#ax.set(xlim=(0,800))
#ax.set(ylim=(0,300))
plt.title("Reading vs Math")
plt.legend(loc='lower right')
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.show()
#darkgrid, whitegrid, dark, white, ticks
sns.set_style("darkgrid")
#paper, notebook, talk, poster
sns.set_context("notebook", font_scale=1.5)
ax = sns.lmplot( x="writing score", y="math score", data=students, fit_reg=False, hue='gender', height=9,
           palette="Set2", aspect=1.3, legend=False,
           scatter_kws={"alpha":0.5,"s":50})
#ax.set(xlim=(0,800))
#ax.set(ylim=(0,300))
plt.title("Writing vs Math")
plt.legend(loc='lower right')
plt.xlabel("Writing Score")
plt.ylabel("Math Score")
plt.show()
students.head(3)
# udents.drop(['race/ethnicity','parental level of education','lunch','test preparation course'], axis=1, inplace=True)
students.drop(students.columns[1:5], axis=1, inplace=True)
students.head(3)
students['gender'] = students['gender'].map({'female':1, 'male': 0})
students.corr()
X = students.drop(columns=['gender'])
y = students['gender']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
print(rescaledX[0:5,:])
from sklearn.model_selection import train_test_split
test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=test_size,
random_state=seed)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0, max_depth=6)
tree.fit(X_train, y_train)
print('Training set is {}'.format(tree.score(X_train, y_train))) 
print('Testing set is {}'.format(tree.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=7, random_state=5)
forest.fit(X_train, y_train)
print('Training set is {}'.format(forest.score(X_train, y_train))) 
print('Testing set is {}'.format(forest.score(X_test, y_test)))
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, max_depth=5)
gbrt.fit(X_train, y_train)
print('Training set is {}'.format(gbrt.score(X_train, y_train))) 
print('Testing set is {}'.format(gbrt.score(X_test, y_test)))
threshold = students['math score'].mean()
threshold
students["math-strong"] = ["strong" if i > threshold else "weak" for i in students['math score']]
students.head(3)
students['math-strong'] = students['math-strong'].map({'strong':1, 'weak': 0})
students.head(3)
students.corr()
students["borw"] = students["math score"] - students["writing score"]
students["borw"] = ["better" if i > 0 else "worse" for i in students["borw"]]
students.head(3)
students['borw'] = students['borw'].map({'better':1, 'worse': 0})
students.corr()
X = students.drop(columns=['gender'])
y = students['gender']
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
print(rescaledX[0:5,:])
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=test_size,
random_state=seed)
new_tree = DecisionTreeClassifier(random_state=0, max_depth=7)
new_tree.fit(X_train, y_train)
print('Training set is {}'.format(new_tree.score(X_train, y_train))) 
print('Testing set is {}'.format(new_tree.score(X_test, y_test)))
