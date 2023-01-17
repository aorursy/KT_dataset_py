import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.describe()
features = df.iloc[:, df.columns != 'Attrition']
labels = df.iloc[:,1]
indexes = df.index.values
features = features.drop(
    columns=['MonthlyRate','EmployeeCount','EmployeeNumber','DailyRate','HourlyRate','Over18','OverTime'
    ,'StockOptionLevel','YearsWithCurrManager','TotalWorkingYears','TrainingTimesLastYear'], axis=1)
features.columns
transformer = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,2,4,5,6,7,8,9,10,11,12,16,17,21])], remainder="passthrough")
features = transformer.fit_transform(features)
le_labels = LabelEncoder()
labels = le_labels.fit_transform(labels)
trainFeatures, testFeatures, trainIndexes, targetIndexes = train_test_split(features, indexes, test_size=0.30, random_state=42)
trainLabels, targetLabels = labels[trainIndexes], labels[targetIndexes]
model = RandomForestClassifier(n_estimators = 200, random_state = 0, criterion='entropy', max_depth=None, min_samples_split=5)
model.fit(trainFeatures, trainLabels)
predictions = model.predict(testFeatures)
precision = accuracy_score(targetLabels, predictions)
matrix = confusion_matrix(targetLabels, predictions)
print('Acurácia do algorítmo ->', round(precision, 2) * 100, '%') 
test_probabilities = model.predict_proba(testFeatures)
train_probabilities = model.predict_proba(trainFeatures)
df_new = df.copy()
df_new.loc[targetIndexes,'probability'] = test_probabilities[:,1]
df_new.loc[trainIndexes,'probability'] = train_probabilities[:,1]
no_attrition = df_new['Attrition']=='No'
df_no_attrition = df_new[no_attrition]

df_no_attrition.head(10)
df_no_attrition.to_csv('HR-Employee-Probability-Attrition.csv', index=False)