# Imported Libraries

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

import pandas as pd

import numpy as np
diabetes_data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

diabetes_data.head()
diabetes_data[:60]

diabetes_data.shape
diabetes_data.isna().sum()

diabetes_data.dtypes
X = diabetes_data.drop("Outcome", axis=1)

y = diabetes_data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svc_model = LinearSVC(max_iter=10000)

svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_test, y_test)

if svc_score < 0.6:

    print(f"SVC Model Score is Less : {svc_score}".format())

    

else:

    random_clf = RandomForestClassifier(n_estimators=100)

    random_clf.fit(X_train, y_train)



    # This is the data from a dummy patient who we are going to predict for if he or she has diabetes

    patient_sample = np.array([[0, 137, 40, 35, 168, 43.1, 2.244, 30]])

    prediction = random_clf.predict(patient_sample)

    

    if prediction == 0:

        print("You are not expected to be diabetic")

        

    elif prediction == 1:

        print("You are expected to be diabetic")