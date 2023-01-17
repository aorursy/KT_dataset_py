import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import data CSV into dataframe
data_raw = pd.read_csv("../input/insurance.csv")

#show sample of data
display(data_raw.head(n=7))
#smoker = 1; non-smoker = 0
data = data_raw.replace(['yes','no'], [1,0])
#female = 1; male = 0
data = data.replace(['female','male'], [1,0])

display(data.head())
#prepare subplots, with 2 columns and 2 rows
fig1, ((ax11,ax12),(ax13,ax14)) = plt.subplots(2,2)
#set full plot size
fig1.set_size_inches(15,12)

#Create a pie chart of the region distribution
ax11.pie(data.groupby("region").size().values, labels=data.groupby('region').size().keys(), autopct='%1.1f%%')
ax11.set_title("Region Distribution", fontsize=20)
ax11.axis('equal')
#Create a pie chart of the sex distribution
ax13.pie(data.groupby("sex").size().values, labels=data.groupby('sex').size().keys(), autopct='%1.1f%%', startangle=90)
ax13.set_title("Sex Distribution", fontsize=20)
ax13.axis('equal')
#Create a pie chart of the smoker distribution
ax12.pie(data.groupby("smoker").size().values, labels=data.groupby('smoker').size().keys(), autopct="%1.1f%%")
ax12.set_title("Smoker Distribution", fontsize=20)
ax12.axis('equal')
#Create a histogram of the children/dependnet distribution
ax14.hist('children', data=data,edgecolor =' 0.2', bins = 5)
ax14.set_title("Dependent Distribution", fontsize=20)
#Prepare subplots with 1 row, 2 columns
fig21, (ax21,ax22) = plt.subplots(1,2)
fig21.set_size_inches(15,6)

#Create a density curve of the BMI distribution
sns.kdeplot(data['bmi'], ax=ax21, shade=True, legend=False)
ax21.set_xlabel("BMI", fontsize=14)
ax21.set_ylabel("Frequency", fontsize=14)
ax21.set_title("BMI Distribution", fontsize=20)

#Create a histogram of the age distribution
ax22.hist('age', data=data, bins=10, edgecolor='0.2')
ax22.set_xlabel("Age", fontsize=14)
ax22.set_ylabel("Frequency", fontsize=14)
ax22.set_title("Age Distribution", fontsize=20)

#Create a separate subplot for the charges distribution
#This is because this is a more important graph, and is better to take up two columns
fig22, ax23 = plt.subplots()
fig22.set_size_inches(15,6)

#Create density plot of charges distribution
sns.kdeplot(data['charges'], ax=ax23, shade=True, legend=False)
ax23.set_xlabel("Charges", fontsize=14)
ax23.set_ylabel("Frequency", fontsize=14)
ax23.set_title("Charges Distribution", fontsize=20)
#Prepare subplots of 4 rows and 1 column
fig3, (ax31,ax32,ax33,ax34) = plt.subplots(4,1)
fig3.set_size_inches(14,28)

#Add 4 density curves to subplot ax31 for the charges distribution, each for one of the 4 regions
sns.kdeplot(data.loc[data["region"] == 'southeast']["charges"], ax=ax31, shade=True, label="southeast")
sns.kdeplot(data.loc[data["region"] == 'southwest']["charges"], ax=ax31, shade=True, label="southwest")
sns.kdeplot(data.loc[data["region"] == 'northwest']["charges"], ax=ax31, shade=True, label="northwest")
sns.kdeplot(data.loc[data["region"] == 'northeast']["charges"], ax=ax31, shade=True, label="northeast")
ax31.set_ylabel("Frequency", fontsize=15)
ax31.set_xlabel('Charges', fontsize=15)
ax31.set_title("Effect of Regions on Cost", fontsize=20)

#Add 2 density curves to subplot ax32 for the charges distribution, each for one of the 2 sexes
#Remember: female = 1, male = 0
sns.kdeplot(data.loc[data["sex"] == 0]["charges"], ax=ax32, shade=True, label="male")
sns.kdeplot(data.loc[data["sex"] == 1]["charges"], ax=ax32, shade=True, label="female")
ax32.set_ylabel("Frequency", fontsize=15)
ax32.set_xlabel('Charges', fontsize=15)
ax32.set_title("Effect of Sex on Cost", fontsize=20)

#Add 6 density curves to subplot ax33 for the charges distribution, each for one caregory of the number of dependents
sns.kdeplot(data.loc[data["children"] == 0]["charges"], ax=ax33, shade=True, label="0 children")
sns.kdeplot(data.loc[data["children"] == 1]["charges"], ax=ax33, shade=True, label="1 children")
sns.kdeplot(data.loc[data["children"] == 2]["charges"], ax=ax33, shade=True, label="2 children")
sns.kdeplot(data.loc[data["children"] == 3]["charges"], ax=ax33, shade=True, label="3 children")
sns.kdeplot(data.loc[data["children"] == 4]["charges"], ax=ax33, shade=True, label="4 children")
sns.kdeplot(data.loc[data["children"] == 5]["charges"], ax=ax33, shade=True, label="0 children")
ax33.set_ylabel("Frequency", fontsize=15)
ax33.set_xlabel('Charges', fontsize=15)
ax33.set_title("Effect of Number of Dependents on Cost", fontsize=20)

#Add 2 density curves to subplot ax34 for the charges distribution, one for smokers and one for non-smokers
sns.kdeplot(data.loc[data["smoker"] == 1]["charges"], ax=ax34, shade=True, label='smoker')
sns.kdeplot(data.loc[data["smoker"] == 0]["charges"], ax=ax34, shade=True, label='non-smoker')
ax34.set_ylabel("Frequency", fontsize=15)
ax34.set_xlabel('Charges', fontsize=15)
ax34.set_title("Effect of Smoking on Cost", fontsize=20)
#Prepare subplots with 1 row and 2 columns
fig4, (ax41,ax42) = plt.subplots(1,2)
fig4.set_size_inches(15,6)

#create a scatterplot with best linear fit for Age vs. Charges
sns.regplot("age", "charges", data, ax=ax41)
ax41.set_title("Effect of Age on Cost", fontsize=20)
ax41.set_xlabel("Age", fontsize=15)
ax41.set_ylabel("Charges", fontsize=15)

#create a scatterplot with best linear fit for BMI vs. Charges
sns.regplot("bmi", "charges", data, ax=ax42)
ax42.set_title("Effect of BMI on Cost", fontsize=20)
ax42.set_xlabel("BMI", fontsize=15)
ax42.set_ylabel("Charges", fontsize=15)
#Prepare subplots with 1 row and 2 columns
fig5, (ax51,ax52) = plt.subplots(1,2)
fig5.set_size_inches(15,6)

#create 2x Age vs Charges scatter plots on the same axes; one for smokers and one for non-smokers
sns.regplot("age", "charges", data.loc[data["smoker"] == 1], ax=ax51)
sns.regplot("age", "charges", data.loc[data["smoker"] == 0], ax=ax51)
ax51.set_title("Effect of Age on Cost", fontsize=20)
ax51.set_xlabel("Age", fontsize=15)
ax51.set_ylabel("Charges", fontsize=15)
ax51.legend(("smoker", "non-smoker"))

#create 2x BMI vs Charges scatter plots on the same axes; one for smokers and one for non-smokers
sns.regplot("bmi", "charges", data.loc[data["smoker"] == 1], ax=ax52)
sns.regplot("bmi", "charges", data.loc[data["smoker"] == 0], ax=ax52)
ax52.set_title("Effect of BMI on Cost", fontsize=20)
ax52.set_xlabel("BMI", fontsize=15)
ax52.set_ylabel("Charges", fontsize=15)
ax52.legend(("smoker", "non-smoker"))
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score

#Create a dataframe of the input data X
#Create a dataframe of the output data Y
X = data.drop(['charges','sex','region'], axis=1)
Y = data["charges"]

#split data into training and testing sets; 20% testing size.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Call training model
regress=DecisionTreeRegressor()

#Parameters for Grid Search
params_tree = {"max_depth":np.arange(3,6), "min_samples_split":np.arange(2,8), "max_leaf_nodes":np.arange(2,20)}

#Call and fit grid search
grid_tree=GridSearchCV(regress, params_tree)
grid_tree.fit(X_train,Y_train)

#Obtain optimized parameters
grid_tree.best_estimator_
predictions = grid_tree.predict(X_test)
r2_score(predictions, Y_test)
#Now, call training model with optimized parameters
clf_tree = DecisionTreeRegressor(max_depth=3,max_leaf_nodes=8,splitter="best", random_state=1)
clf_tree.fit(X_train,Y_train)
predictions_tree = clf_tree.predict(X_test)
print(r2_score(predictions_tree, Y_test))
#Show which features are most significant
feats = {}
for feature, importance in zip(X_train.columns, clf_tree.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance',ascending=False).plot(kind='bar', rot=45)
print(importances.sort_values(by='Gini-importance',ascending=False))
plt.show()
import pydotplus
from sklearn import tree
import collections
from PIL import Image

dot_data = tree.export_graphviz(clf_tree, feature_names=X_train.columns, out_file=None,filled=True, rounded=True)


graph = pydotplus.graph_from_dot_data(dot_data)
colors = ( 'lightblue','orange')
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        
        dest.set_fillcolor(colors[i])
graph.write_png('tree.png')        

from IPython.display import Image
Image("tree.png")
from sklearn.linear_model import LinearRegression

#Train and fit Linear Regression
linreg = LinearRegression()
linreg.fit(X_train,Y_train)
predictions_linear = linreg.predict(X_test)
r2_score(predictions_linear, Y_test)
from sklearn.ensemble import AdaBoostRegressor

#Train and fit AdaBoost Regressor
adaboost = AdaBoostRegressor(n_estimators=5, learning_rate=2)
adaboost.fit(X_train, Y_train)
predictions_boost = adaboost.predict(X_test)
r2_score(predictions_boost, Y_test)
#Show which features are most significant
feats = {}
for feature, importance in zip(X_train.columns, adaboost.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance',ascending=False).plot(kind='bar', rot=45)
print(importances.sort_values(by='Gini-importance',ascending=False))
plt.show()
