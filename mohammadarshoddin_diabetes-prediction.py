import pandas as pd



import numpy as np



import matplotlib.pyplot as plt



import seaborn as sns
df = pd.read_csv('../input/diabetes-dataset/data.csv')



df.head()
df.shape
df.diabetes.value_counts()
df.info()

df.describe()
sns.set_style('darkgrid')



sns.countplot(x = 'diabetes', data = df, palette = 'CMRmap')



plt.show()
plt.figure(figsize = (16, 16))

 

sns.heatmap(df.corr(), annot = True)



plt.show()
xdata = df.drop(columns = 'diabetes')



ydata = df['diabetes']
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size = 0.10, random_state = 1)



print("Shape of xtrain :: ", xtrain.shape)



print("Shape of xtest :: ", xtest.shape)



print("Shape of ytrain :: ", ytrain.shape)



print("Shape of ytest :: ", ytest.shape)
from sklearn.preprocessing import StandardScaler

 

standard_scalar = StandardScaler()



xtrain = standard_scalar.fit_transform(xtrain)



xtest = standard_scalar.transform(xtest)
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier(n_estimators = 100)



rf_model
rf_model.fit(xtrain, ytrain)
rf_predicted_value = rf_model.predict(xtest)



rf_predicted_value
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
rf_score = accuracy_score(ytest, rf_predicted_value)



print("Score of the prediction by RandomForestClassifier is :: ", rf_score)



rf_accuracy = rf_score*100



print("Accuracy of the prediction by RandomForestClassifier is :: ", rf_accuracy, "%")
rf_report = classification_report(ytest, rf_predicted_value)



print("The Classification report for the Random Forest Classifier")



print("\n")



print(rf_report)
rf_conf_matrix = confusion_matrix(ytest, rf_predicted_value)



print(rf_conf_matrix)
sns.heatmap(rf_conf_matrix, annot = True, cmap = 'cool', linewidths = 0.2, yticklabels = ['False', 'True'], xticklabels = ['Predicted_False', 'Predicted_True'])
feature_imp = pd.Series(rf_model.feature_importances_)



print(feature_imp)



indices = list(xdata.columns)



print(indices)
new_xdata = xdata.drop(columns = ['thickness', 'insulin', 'skin'])



new_xdata.head()
new_xtrain, new_xtest, new_ytrain, new_ytest = train_test_split(new_xdata, ydata, test_size = 0.10, random_state = 1)



print("Shape of new xtrain :: ", new_xtrain.shape)



print("Shape of new xtest :: ", new_xtest.shape)



print("Shape of new ytrain :: ", new_ytrain.shape)



print("Shape of new ytest :: ", new_ytest.shape)
new_rf_model = RandomForestClassifier(n_estimators = 100)



new_rf_model
new_rf_model.fit(new_xtrain, new_ytrain)
new_predicted_value = new_rf_model.predict(new_xtest)



new_predicted_value
new_score = accuracy_score(new_ytest, new_predicted_value)



print("New Score of the prediction by RandomForestClassifier is :: ", new_score)



new_accuracy = new_score*100



print("New Accuracy of the prediction by RandomForestClassifier is :: ", new_accuracy, "%")
new_rf_report = classification_report(ytest, new_predicted_value)



print("The Classification report for the New Random Forest Classifier")



print("\n")



print(new_rf_report)
new_conf_matrix = confusion_matrix(new_ytest, new_predicted_value)



print(new_conf_matrix)
sns.heatmap(new_conf_matrix, annot = True, cmap = 'cool', linewidths = 0.2, yticklabels = ['False', 'True'], xticklabels = ['Predicted_False', 'Predicted_True'])
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors = 10)



knn_model
knn_model.fit(xtrain, ytrain)
knn_predicted_value = knn_model.predict(xtest)



knn_predicted_value
knn_score = accuracy_score(ytest, knn_predicted_value)



print("Score of the prediction by KNeighborsClassifier is :: ", knn_score)



knn_accuracy = knn_score*100



print("Accuracy of the prediction by KNeighborsClassifier is :: ", knn_accuracy, "%")
knn_report = classification_report(ytest, knn_predicted_value)



print("The Classification report for the KNeighbors Classifier")



print("\n")



print(knn_report)
knn_conf_matrix = confusion_matrix(ytest, knn_predicted_value)



print(knn_conf_matrix)
sns.heatmap(knn_conf_matrix, annot = True, cmap = 'cool', linewidths = 0.2, yticklabels = ['False', 'True'], xticklabels = ['Predicted_False', 'Predicted_True'])
from sklearn.tree import DecisionTreeClassifier



dt_model = DecisionTreeClassifier()



dt_model
dt_model.fit(xtrain, ytrain)
dt_predicted_value = dt_model.predict(xtest)



dt_predicted_value
dt_score = accuracy_score(ytest, dt_predicted_value)



print("Score of the prediction by DecisionTreeClassifier is :: ", dt_score)



dt_accuracy = dt_score*100



print("Accuracy of the prediction by DecisionTreeClassifier is :: ", dt_accuracy, "%")
dt_report = classification_report(ytest, dt_predicted_value)



print("The Classification report for the Decision Tree Classifier")



print("\n")



print(dt_report)
dt_conf_matrix = confusion_matrix(ytest, dt_predicted_value)



print(dt_conf_matrix)
sns.heatmap(dt_conf_matrix, annot = True, cmap = 'cool', linewidths = 0.2, yticklabels = ['False', 'True'], xticklabels = ['Predicted_False', 'Predicted_True'])
from sklearn.tree import export_graphviz



from IPython.display import Image



from io import StringIO



import pydot
indices
dot_data = StringIO()



export_graphviz(dt_model, out_file = dot_data, feature_names = indices, filled = True, rounded = True)



graph = pydot.graph_from_dot_data(dot_data.getvalue())



Image(graph[0].create_png())
x = [('rf_accuracy', rf_accuracy), ('new_rf_accuracy', new_accuracy), ('knn_accuracy', knn_accuracy), ('dt_accuracy', dt_accuracy)]



labels, values = zip(*x)

 

plt.figure(figsize = (10, 6))



plt.bar(labels, values, color = 'b')



plt.xticks(color = 'm', fontsize = 16)



plt.yticks(color = 'm', fontsize = 16)



plt.xlabel('Classifier', color = 'g', fontsize = 18)



plt.ylabel('Accuarcy in %', color = 'g', fontsize = 18)



plt.title('Accuracy Obtained for Different Classifiers', color = 'r', fontsize = 20)



plt.show()