import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data= pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.shape
data.info()
data.describe()
data_modified= data.copy()
data_modified.head()
data_modified[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = data_modified[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)
data_modified.head(10)
corr=data_modified.corr()
corr
plt.figure(figsize=(8,7))
sns.heatmap(corr, annot=True)
import missingno as msno
msno.matrix(data_modified,figsize=(13,5),fontsize=10)
msno.bar(data_modified,figsize=(12,4),fontsize=10)
data_modified.hist(figsize=(11,11))
data_modified["Glucose"].fillna(data_modified["Glucose"].mean(), inplace=True)
data_modified["BloodPressure"].fillna(data_modified["BloodPressure"].mean(), inplace= True)
data_modified["SkinThickness"].fillna(data_modified["SkinThickness"].median(),inplace=True)
data_modified["Insulin"].fillna(data_modified["Insulin"].median(), inplace=True)
data_modified["BMI"].fillna(data_modified["BMI"].median(), inplace=True)
data_modified.hist(figsize=(11,11))
msno.matrix(data_modified,figsize=(12,4),fontsize=10)
msno.bar(data_modified,figsize=(12,4),fontsize=10)
sns.countplot(data_modified["Outcome"])
data_modified.Outcome.value_counts()
#p= data_modified.Outcome.value_counts().plot(kind="bar")
#p
plt.figure(figsize=(30,30))
p_plot= sns.pairplot(data_modified, hue="Outcome")
new_corr= data_modified.corr()
new_corr
plt.figure(figsize=(8,7))
sns.heatmap(new_corr, annot=True)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
data_modified_X = data_modified.drop("Outcome",axis=1)
data_modified_X
X = pd.DataFrame(scaler.fit_transform(data_modified_X), columns = ["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction","Age"])
X.head()
y= data_modified["Outcome"]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
test_scores= []
train_scores= []


for i in range(1,20):
    model= KNeighborsClassifier(i)
    model.fit(X_train,y_train)

    test_scores.append(model.score(X_test,y_test))
    train_scores.append(model.score(X_train,y_train))
    
    
test_scores, train_scores
max(test_scores),max(train_scores)
max_index_test= test_scores.index(max(test_scores))+1
print(max_index_test)

max_index_train= train_scores.index(max(train_scores))+1
print(max_index_train)
print("max train score is {}% at k=[{}]".format(max(train_scores)*100,max_index_train))
print("max test score is {}% at k=[{}]".format(max(test_scores).round(3)*100,max_index_test))
plt.figure(figsize=(11,4))
sns.lineplot(range(1,20),train_scores,marker='o',label='Train Score')
sns.lineplot(range(1,20),test_scores,markers="*",label="Test Score")
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_predicted)
df_confusion_matrix = pd.DataFrame(confusion_matrix)
sns.heatmap(df_confusion_matrix, annot=True,fmt="g")

plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))
