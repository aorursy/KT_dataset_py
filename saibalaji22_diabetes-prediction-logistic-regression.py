import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
dataDF = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')
dataDF.columns
dataDF.info()
dataDF.head(2)
import seaborn as sn
sn.heatmap(dataDF.corr(),annot=True)
X = dataDF[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataDF.Outcome
X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
model.score(X_test,y_test)
y_predict = model.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize=(5,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

