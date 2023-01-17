import pandas as pd
nombres = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']
dataset=pd.read_csv("../input/Heart_Disease_Data.csv", na_values="?")
#dataset = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleve.mod', skiprows=20, header=None, sep='\s+', names=nombres, index_col=False, na_values="?")
dataset["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
dataset
%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
#Boxplots
continuas=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
dataset[continuas].boxplot(return_type='axes', figsize=(12,8))
plt.show()
continuas=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
dataset[continuas].describe()
#Sex: sex (1 = male; 0 = female) 
tempo5 = dataset['sex']
tempo5.value_counts().plot(kind="bar")
#Fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
tempo6 = dataset['fbs']
tempo6.value_counts().plot(kind="bar")
#Slope: the slope of the peak exercise ST segment  
#Value 1: upsloping 
#Value 2: flat 
#Value 3: downsloping
tempo7 = dataset['slop']
tempo7.value_counts().plot(kind="bar")
#Cp: chest pain type
#Value 1: typical angina 
#Value 2: atypical angina 
#Value 3: non-anginal pain 
#Value 4: asymptomatic 
tempo8 = dataset['cp']
tempo8.value_counts().plot(kind="bar")
#Exang: exercise induced angina (1 = yes; 0 = no) 
tempo9 = dataset['exang']
tempo9.value_counts().plot(kind="bar")
#Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
tempo10 = dataset['thal']
tempo10.value_counts().plot(kind="bar")
#Restecg: resting electrocardiographic results 
#Value 0: normal 
#Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
#Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
tempo11 = dataset['restecg']
tempo11.value_counts().plot(kind="bar")
#Class: diagnosis of heart disease (angiographic disease status) 
#Value 0: < 50% diameter narrowing (Healthy)
#Value 1: > 50% diameter narrowing (Sick)
tempo12 = dataset['pred_attribute']
tempo12.value_counts().plot(kind="bar")
dataset.dropna(inplace=True, axis=0, how="any")
X=dataset.loc[:, "age":"thal" ]
Y=dataset["pred_attribute"]
# evaluate the model by splitting into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

freqs = pd.DataFrame({ "Training dataset": Y_train.value_counts().tolist(), "Test dataset":Y_test.value_counts().tolist(), "Total": Y.value_counts().tolist()}, index=["Healthy", "Sick"])
freqs[["Training dataset", "Test dataset", "Total"]]
# instantiate a logistic regression model, and fit with X and y (with training data in X,y)
model = LogisticRegression()
model.fit(X_train, Y_train)

# check the accuracy on the training set
model.score(X_train, Y_train)



# check the accuracy on the test set
model.score(X_test, Y_test)


# predict class labels for the training set
predicted1 = model.predict(X_train)

# predict class labels for the test set
predicted2 = model.predict(X_test)
pd.crosstab(Y_train, predicted1, rownames=['Predicted'], colnames=['Reality'], margins=True)
pd.crosstab(Y_test, predicted2, rownames=['Predicted'], colnames=['Reality'], margins=True)