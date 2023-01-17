import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/mushrooms.csv")
df.head()
abt = pd.get_dummies(df)
abt.head()
#Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Building everything
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

#Saving the model
import pickle
df["class"].replace(["e", "p"], [1, 0], inplace= True)
# Create separate object for target variable
y = df["class"]
# Create separate object for input features
X = abt.drop(["class_e", "class_p"], axis= 1).astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)
print(len(X_train), len(X_test), len(y_train), len(y_test))
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=123))
}
rf_hyperparameters = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingclassifier__n_estimators": [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}
# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')
for name, model in fitted_models.items():
    print(name, model.best_score_)
for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('Acc:', accuracy_score(y_test, pred))
    print("cm:\n", confusion_matrix(y_test, pred))
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)