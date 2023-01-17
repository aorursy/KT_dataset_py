# Kaggle
# Task to classify diabetic patients from glucose level, BMI, etc.

# https://www.kaggle.com/uciml/pima-indians-diabetes-database
# https://www.kaggle.com/hkthirano/logisticregression-and-randomforestclassifier/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))
diabetes_df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
col_name = diabetes_df.columns
diabetes_df.head()
# check na
print(diabetes_df.isnull().any().sum())
diabetes_df = diabetes_df.values
X = diabetes_df[:,0:8] #Predictors
y = diabetes_df[:,8] #Target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
# LogisticRegression

# Pregnancies and BloodPressure are important
logisticl1_model = LogisticRegression(penalty='l1', solver='saga')
# DiabetesPedigreeFunction is important
logisticl2_model = LogisticRegression(penalty='l2')

model_dict = {'logisticl1':logisticl1_model, 'logisticl2':logisticl2_model}

for model_name in ['logisticl1', 'logisticl2']:
    print('=== {} ==='.format(model_name))
    model = model_dict[model_name]

    model.fit(X_train,y_train)
    predicted = model.predict(X_test)

    print("Confusion Matrix")
    matrix = confusion_matrix(y_test,predicted)
    print(matrix)
    
    df = pd.DataFrame(model.coef_.T, columns=['coefficient'], index=col_name[:-1])
    df.plot.bar()
# RandomForest
# Glucose and BMI are important
randomforest_model = RandomForestClassifier()
print('=== RandomForest ===')

randomforest_model.fit(X_train,y_train)
predicted = randomforest_model.predict(X_test)

print("Confusion Matrix")
matrix = confusion_matrix(y_test,predicted)
print(matrix)

df = pd.DataFrame(randomforest_model.feature_importances_.T, columns=['importances'], index=col_name[:-1])
df.plot.bar()