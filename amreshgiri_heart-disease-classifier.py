## use this command to activate virtual environment in jupyter notebook 

# 1. make the virtual environment 
# 2. install ipykernel in the environment 
# 3. make the kernel of jupyter notebook for your virtual environment with the command:
#    >> python -m ipykernel install --user --name=your_virtual_environment_name

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
## loading the dataset 

df = pd.read_csv('../input/heart.csv')
df.head()
# renaming column names to be more interpretable 

updated_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'SerumCholestoral', 'FastingBloodSugar', 'RestingECG', 
               'MaxHeartRate', 'ExeriseEnducedAngina', 'OldPeak', 'SlopeOldPeak', 'MajorVessels', 'Thal', 'Output']
df.columns = updated_cols
df.head()
# since all the features are numerical, we can use df.describe() to understand the statistical measures

df.describe()
## plotting all the features to observe any interesting fact 

## 1. Age

plt.figure(figsize=(15,7))
plt.xlabel('Number of People')
plt.ylabel('Age')
plt.title('Age')
plt.plot(df.Age)
print("Min Age: ", df.Age.min())
print("Mean Age: ", df.Age.mean())
print("Max Age: ", df.Age.max())
# 2. Sex

sns.set(style="darkgrid")
plt.figure(figsize=(8,4))
plt.title('Male v/s Female')
sns.countplot(x="Sex", data=df)
print("Number of Males: ", (df.Sex==1).sum())
print("Number of Females: ", (df.Sex==0).sum())
print("Total People: ", df.Sex.count())
# 3. Chest Pain Type 

sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
plt.title('Chest Pain Type')
sns.countplot(x="ChestPainType", data=df)
print("Number of Chest Pain Type 0: ", (df.ChestPainType==0).sum())
print("Number of Chest Pain Type 1: ", (df.ChestPainType==1).sum())
print("Number of Chest Pain Type 2: ", (df.ChestPainType==2).sum())
print("Number of Chest Pain Type 3: ", (df.ChestPainType==3).sum())
# 4. Blood Pressure 

sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
plt.title('Resting Blood Pressure')
plt.plot(df.RestingBP)
print("Highest value: ", df.RestingBP.max())
print("Mean value: ", df.RestingBP.mean())
print("Lowest value: ", df.RestingBP.min())
# 5. Serum Cholestrol

sns.set(style="darkgrid")
plt.figure(figsize=(13,6))
plt.title('Serum Cholestoral')
plt.xlabel("Number of People")
plt.ylabel("Serum Cholestoral count")
plt.plot(df["SerumCholestoral"])
print("Highest value: ", df.SerumCholestoral.max())
print("Mean value: ", df.SerumCholestoral.mean())
print("Lowest value: ", df.SerumCholestoral.min())
# 6. Fasting Blood Sugar

print("Fasting Blood Sugar > 120 mg/dl ?")
print("Yes: 1")
print("No: 0")
sns.set(style="darkgrid")
plt.figure(figsize=(9,5))
plt.title('Fasting Blood Sugar')
sns.countplot(x='FastingBloodSugar', data=df)
plt.show()
# 7. Resting ECG

sns.countplot(x='RestingECG', data=df)
plt.title("Resting ECG")
plt.show()
# 8. Max Heart Rate

plt.figure(figsize=(9,4))
plt.plot(df.MaxHeartRate)
print("Highest value: ", df.MaxHeartRate.max())
print("Mean value: ", df.MaxHeartRate.mean())
print("Lowest value: ", df.MaxHeartRate.min())
plt.xlabel("Number of Poeple")
plt.ylabel("Heart Rate")
plt.title("Max Heart Rate")
plt.show()
# 9. Exerise Enduced Angina

plt.figure(figsize=(7,4))
sns.countplot(x='ExeriseEnducedAngina', data=df)
plt.title("Exercise Enduced Angina")
plt.show()
# 10. Old Peak

print("Highest value: ", df.OldPeak.max())
print("Mean value: ", df.OldPeak.mean())
print("Lowest value: ", df.OldPeak.min())
plt.figure(figsize=(9, 4))
plt.title("Old Peak")
plt.plot(df.OldPeak)
plt.show()
# 11. Slope Old Peak

plt.figure(figsize=(9, 4))
sns.countplot(x='SlopeOldPeak', data=df)
plt.title("Slope Old Peak")
plt.show()
# 12. Major Vessels

plt.figure(figsize=(9, 4))
sns.countplot(data=df, x='MajorVessels')
plt.title("Major Vessels")
plt.show()
# 13. Thal

plt.figure(figsize=(10, 6))
sns.countplot(x='Thal', data=df)
plt.title("Thal")
plt.show()
# 14. Output

sns.countplot(x='Output', data=df)
plt.title("Output")
plt.show()
# Preparing the features dataframe 

X = df.drop('Output', axis=1)
X.head()
# Preparing the depenedent variable / output

y = df.Output
y.head()
# Creating the training and test sets 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# Using Random Forest Classifier to be the first model 

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rf = model.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print("Score using Random Forest without any hyperparameter tuning: ", rf_score)
# Using grid search for hyperparameter tuning for random forest to get the best paramaters and score 

from sklearn.model_selection import GridSearchCV

grid_param = {  
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
    'bootstrap': [True, False]
}

gd_sr = GridSearchCV(estimator=rf,  
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)

gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_ 
best_score = gd_sr.best_score_

print("Best paramaters for RandomForestClassifier: ", best_parameters)
print("Best score for RandomForestClassifier: ", best_score )
# Using grid search for hyperparameter tuning for SVC to get the best paramaters and score 

from sklearn.svm import SVC

svc = SVC()

grid_param = {  
      'C':[1,10,100,1000],
      'gamma':[1,0.1,0.001,0.0001], 
      'kernel':['linear','rbf']
}

gd_sr = GridSearchCV(estimator=svc,  
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)

gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_  
best_score = gd_sr.best_score_

print("Best parameters for SVC: ", best_parameters)  
print("Best score for SVC: ", best_score)