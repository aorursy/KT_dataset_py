import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
data_2C = pd.read_csv("../input/column_2C_weka.csv")  # for clearence I will name both of my DataFrames corresponding to original csv files
data_2C.info()    # except our classes, whole dataset is float and there is no null value at all.
data_2C.head(10)   # as we also know from the overview provided by owner of data, there is only 2 classes, Normal and Abnormal
data_2C.describe()
data_2C.corr()  # there are some highly correlated columns. such as pelvic_incidence and sacral_slope
import seaborn as sns   # for making better plots easily
sns.pairplot(data_2C,hue="class",palette="Set2")
plt.show()
data_2C.boxplot(figsize=(20,16),by="class",grid=True)
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(data_2C.corr(),vmax=1,vmin=-1,linewidths=0.4,annot=True)
plt.show()
# first I will work on our x and y values for model

x_train = data_2C.pelvic_incidence.values.reshape(-1,1)   # shape is crucial for sklearn models since they don't work well with "(n,)" style shapes.
y_train = data_2C.sacral_slope.values.reshape(-1,1)

x_test = np.arange(min(x_train),max(x_train)).reshape(-1,1)   # to predict each possible value between lowest and highest x values
plt.clf()
plt.figure(figsize=(10,6))
plt.scatter(x_train,y_train,c="orange")
plt.xlabel("pelvic incidence")
plt.ylabel("sacral slope")
plt.show()   # it's very sutiable especially for regression.
# importing the model
from sklearn.linear_model import LinearRegression
# declaring the model
lr_model = LinearRegression()
# training the model
lr_model.fit(x_train,y_train)

# predicting for all x_test (for graph) values (sequential) and storing in lr_y_head
lr_y_head = lr_model.predict(x_test)

# predicting real x values for score evaluation
lr_y_head_real = lr_model.predict(x_train)
# importing the model
from sklearn.tree import DecisionTreeRegressor
# declaring the model
dtr_model = DecisionTreeRegressor()
# training the model
dtr_model.fit(x_train,y_train)

# predicting for all x_test values and storing in dtr_y_head
dtr_y_head = dtr_model.predict(x_test)

# predicting for all x_test (for graph) values (sequential) and storing in dtr_y_head
dtr_y_head_real = dtr_model.predict(x_train)
# importing the model
from sklearn.ensemble import RandomForestRegressor
# declaring the model and setting amount of trees to the 128
rf_model = RandomForestRegressor(n_estimators=128,random_state=42)
# training the model
rf_model.fit(x_train,y_train)

# predicting for all x_test values and storing in rf_y_head
rf_y_head = rf_model.predict(x_test)

# predicting for all x_test (for graph) values (sequential) and storing in rf_y_head
rf_y_head_real = rf_model.predict(x_train)
plt.figure(figsize=(20,10))
plt.scatter(x_train,y_train,c="gray")
plt.xlabel("pelvic incidence")
plt.ylabel("sacral slope")
plt.plot(x_test,lr_y_head,c="red",label="Linear Regression",linewidth=4)
plt.plot(x_test,rf_y_head,c="green",label="RandomForest Regression",linewidth=4)
plt.plot(x_test,dtr_y_head,c="blue",label="DecisionTree Regression",linewidth=4)
plt.legend()
plt.suptitle("COMPARISON OF THE REGRESSION MODELS",fontsize=20)
plt.show()   
from sklearn.metrics import r2_score

print("score of linear regressor:",          r2_score(y_train,lr_y_head_real))
print("score of decisiontree regressor:",    r2_score(y_train,dtr_y_head_real))
print("score of randomforest regressor:",    r2_score(y_train,rf_y_head_real))
algorithms = ("Linear Regression","Random Forest Regression","Decision Tree Regression")
scores = (r2_score(y_train,lr_y_head_real), r2_score(y_train,dtr_y_head_real), r2_score(y_train,rf_y_head_real))
y_pos = np.arange(1,4)
colors = ("red","green","blue")

plt.figure(figsize=(24,12))
plt.xticks(y_pos,algorithms,fontsize=18)
plt.yticks(np.arange(0.00, 1.01, step=0.1))
plt.bar(y_pos,scores,color=colors)
plt.grid()
plt.suptitle("Bar Chart Comparison of Models",fontsize=24)
plt.show()
# Your Votes and Comments does matter to me. Please share your ideas or advices.       Regards,efe.
