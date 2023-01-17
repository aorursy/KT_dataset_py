#Import libraries

from IPython.display import display

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 
#Import dataset

data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

data.columns = ["Id", "GRE", "TOEFL", "University_rating", "SOP", "LOR", "CGPA", "Research", "Chance"]

display(data.head())
data.describe()
plt.figure(figsize=(25,10))

sns.set_style("darkgrid")



plt.subplot(1,2,1)

plt.title("GRE Score Distribution (A)", fontsize = 20)

a = sns.distplot(data["GRE"], color = "blue")

a.set_xlabel("GRE Score",fontsize=15)

a.set_ylabel("Percentage (%)",fontsize=15)

plt.plot()



plt.subplot(1,2,2)

plt.title("GRE Score Distribution (B)", fontsize = 20)

a = sns.countplot(data["GRE"], color = "blue")

a.set_xlabel("GRE Score",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.xticks(rotation = 90)

plt.plot()



plt.show()
plt.figure(figsize=(15,7))

plt.title("Chance of admit vs GRE Score", fontsize = 20)

a = sns.scatterplot(data["GRE"], data["Chance"], color = "blue")

a.set_xlabel("GRE Score",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.show()
plt.figure(figsize=(25,10))



plt.subplot(1,2,1)

plt.title("TOEFL Score Distribution (A)", fontsize = 20)

a = sns.distplot(data["TOEFL"], color = "0.25")

a.set_xlabel("TOEFL Score",fontsize=15)

a.set_ylabel("Percentage (%)",fontsize=15)

plt.plot()



plt.subplot(1,2,2)

plt.title("TOEFL Score Distribution (B)", fontsize = 20)

a = sns.countplot(data["TOEFL"], color = "0.6")

a.set_xlabel("TOEFL Score",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.plot()



plt.show()
sns.set_style("darkgrid")

plt.figure(figsize=(15,7))

plt.title("Chance of admit vs TOEFL Score", fontsize = 20)

a = sns.scatterplot(data["TOEFL"], data["Chance"], color = "0.6")

a.set_xlabel("TOEFL Score",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.show()
data.University_rating.astype('category')



plt.figure(figsize=(25,10))



plt.subplot(1,2,1)

plt.title("University Rating Distribution", fontsize = 20)

a = sns.countplot(data["University_rating"], color = "green")

a.set_xlabel("University Rating",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.plot()



plt.subplot(1,2,2)

plt.title("Chance of admit vs University Rating", fontsize = 20)

a = sns.swarmplot(data["University_rating"],data["Chance"], hue = data["University_rating"])

a.set_xlabel("University Rating",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.plot()



plt.show()
plt.figure(figsize=(25,10))



plt.subplot(1,2,1)

plt.title("Statement of Purpose Distribution", fontsize = 20)

a = sns.countplot(data["SOP"], color = "purple")

a.set_xlabel("Statement of Purpose score",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.plot()



plt.subplot(1,2,2)

plt.title("Chance of admit vs Statement of Purpose Score", fontsize = 20)

a = sns.swarmplot(data["SOP"],data["Chance"], hue = data["SOP"])

a.set_xlabel("Statement of Purpose score",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.plot()



plt.show()
plt.figure(figsize=(25,10))



plt.subplot(1,2,1)

plt.title("Letter of Recommendation Strength Distribution", fontsize = 20)

a = sns.countplot(data["LOR"], color = "red")

a.set_xlabel("Letter of Recommendation Strength score",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.plot()



plt.subplot(1,2,2)

plt.title("Chance of admit vs Letter of Recommendation Strength Score", fontsize = 20)

a = sns.swarmplot(data["LOR"],data["Chance"], hue = data["LOR"])

a.set_xlabel("Recommendation Strength score",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.plot()



plt.show()
plt.figure(figsize=(25,10))



plt.subplot(1,2,1)

plt.title("Cumulative Grade Points Average (CGPA) Distribution", fontsize = 20)

a = sns.distplot(data["CGPA"], color = "magenta")

a.set_xlabel("Cumulative Grade Points Average (CGPA) score",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.plot()



plt.subplot(1,2,2)

plt.title("Chance of admit vs Cumulative Grade Points Average (CGPA) Score", fontsize = 20)

a = sns.scatterplot(data["CGPA"],data["Chance"], color = "magenta")

a.set_xlabel("Cumulative Grade Points Average (CGPA) score",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.plot()



plt.show()
research_yes = data.Research[data.Research == 1].count()

research_no = data.Research[data.Research == 0].count()



plt.figure(figsize=(25,10))

sns.set_style("whitegrid")



plt.subplot(1,2,1)

explode = (0.05,0.05)

sizes = [research_yes/len(data)*100, research_no/len(data)*100]

plt.pie(sizes, labels=["YES", "NO"],colors = ["lightgreen", "lightblue"], shadow=True, startangle=90,explode = explode, autopct='%1.1f%%', textprops={'fontsize': 20})

my_circle=plt.Circle( (0,0), 0.4, color='white')

plt.title("Research Experience Distribution", fontsize= 20)

plt.gca().add_artist(my_circle)

plt.plot()



plt.subplot(1,2,2)

plt.title("Chance of admit vs Research Experience", fontsize = 20)

a = sns.swarmplot(data["Research"],data["Chance"], hue = data["Research"])

a.set_xlabel("Research Experience score",fontsize=15)

a.set_ylabel("Chance of admit (%)",fontsize=15)

plt.plot()



plt.show()
plt.figure(figsize=(25,8))

sns.set_style("darkgrid")

plt.title("Chance of Admit Distribution", fontsize = 20)

a = sns.distplot(data["Chance"], color = "darkgreen")

a.set_xlabel("Chance of Admit score",fontsize=15)

a.set_ylabel("Count",fontsize=15)

plt.plot()

plt.show()
data.drop('Id', axis=1, inplace=True)
plt.figure(figsize=(25,10))

sns.heatmap(data.corr(), linewidths=.02, cmap="YlGnBu", annot=True)

plt.plot()

plt.show()
features = ["GRE", "TOEFL", "University_rating", "SOP", "LOR", "CGPA", "Research"]

y = data.Chance

X = data[features]
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



model = LinearRegression()

model.fit(train_X, train_y)

preds_lin = model.predict(val_X)



print("MAE (Decision Tree regression): ", mean_absolute_error(val_y, preds_lin))

print("R^2 score (Decision Tree regression): ", r2_score(val_y, preds_lin))
plt.figure(figsize=(25,10))

plt.title("Observed vs Predicted values", fontsize = 25)

a = sns.regplot(val_y, preds_lin)

plt.xlabel("Observed Value", fontsize = 15)

plt.ylabel("Predicted Value", fontsize = 15)

plt.plot()

plt.show()
import statistics

from scipy import stats



plt.figure(figsize=(25,10))

plt.title("Std Error vs Fitted Value", fontsize = 25)

a = sns.scatterplot(preds_lin, (val_y - preds_lin) / statistics.variance(val_y - preds_lin))

plt.xlabel("Predicted Value ", fontsize = 15)

plt.ylabel("Residual standardized", fontsize = 15)

plt.plot()

plt.show()



print("Shapiro test: ",stats.shapiro(val_y - preds_lin))

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



validation_nodes = list()

validation_mae = list()



print("VALIDATION SET" )

for max_leaf_nodes in [5, 25, 50, 100, 500, 1000, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("\t Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))

    validation_nodes.append(max_leaf_nodes)

    validation_mae.append(my_mae)



training_nodes = list()

training_mae = list()    



print("TRAINING SET" )

for max_leaf_nodes in [5, 25, 50, 100, 500, 1000, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, train_X, train_y, train_y)

    print("\t Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))

    training_nodes.append(max_leaf_nodes)

    training_mae.append(my_mae)

my_xticks = ["5", "25", "50", "100", "500", "1000", "5000"]



plt.figure(figsize=(25,10))

plt.title("MAE (Training and Validation set) vs Number of Nodes ", fontsize = 25)

plt.plot(my_xticks, training_mae, label='Training Set')

plt.plot(my_xticks, validation_mae, label='Validation Set')

plt.xlabel("Number of Max Leaf Nodes",fontsize=15)

plt.ylabel("MAE",fontsize=15)

plt.legend(fontsize = 10)

plt.plot()

plt.show()
model = DecisionTreeRegressor(max_leaf_nodes = 5, random_state=0)

model.fit(train_X, train_y)

preds_val = model.predict(val_X)

print("MAE (Decision Tree regression): ", mean_absolute_error(val_y, preds_val))

print("R^2 score (Decision Tree regression): ", r2_score(val_y, preds_val))
display(pd.DataFrame({"Observed": val_y, "Predicted": preds_val}))
from sklearn.ensemble import RandomForestRegressor



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

print("MAE (Random Forest): ", mean_absolute_error(val_y, melb_preds))

print("R^2 score (Random Forest): ", r2_score(val_y, melb_preds))
display(pd.DataFrame({"Observed": val_y, "Predicted": melb_preds}))
plt.figure(figsize=(25,10))



plt.subplot(1,2,1)

y = np.array([r2_score(val_y, preds_lin), r2_score(val_y, preds_val), r2_score(val_y, melb_preds)])

x = ["Linear Regression","Decision Tree Regression", "Random Forest Regression"]

sns.barplot(x,y)

plt.title("Comparison of Regression Algorithms (R^2)", fontsize=20)

plt.xlabel("Regression Type", fontsize=15)

plt.ylabel("R^2 score", fontsize=15)

plt.plot()



plt.subplot(1,2,2)

y = np.array([mean_absolute_error(val_y, preds_lin), mean_absolute_error(val_y, preds_val), mean_absolute_error(val_y, melb_preds)])

x = ["Linear Regression","Decision Tree Regression", "Random Forest Regression"]

sns.barplot(x,y)

plt.title("Comparison of Regression Algorithms (MAE)", fontsize=20)

plt.xlabel("Regression Type", fontsize=15)

plt.ylabel("MAE score", fontsize=15)

plt.plot()



plt.show()