import numpy as np

import pandas as pd



import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

%matplotlib inline





import sklearn

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
input_ = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"

data = pd.read_csv(input_)

df = data.copy()



data.head(10)
df.describe()
fig = px.histogram(data, "age", title="Age Distribution", width=750)

fig.show()
fig = px.histogram(data, "time", title="Time Distribution", width=750)

fig.show()
fig = px.histogram(data, "creatinine_phosphokinase", title="Creatinine Phosphokinase Distribution", width=750)

fig.show()
fig = px.histogram(data, "ejection_fraction", title="Ejection Fraction Distribution", width=750)

fig.show()
fig = px.histogram(data, "platelets", title="Platelets Distribution", width=750)

fig.show()
fig = px.histogram(data, "serum_creatinine", title="Serum Creatinine Distribution", width=750)

fig.show()
fig = px.histogram(data, "serum_sodium", title="Serum Sodium Distribution", width=750)

fig.show()
anaemia_dis = data["anaemia"].value_counts().reset_index()

fig = px.bar(anaemia_dis, x="index", y="anaemia", title="Anaemia Distribution",

             width=750, labels={"index": "Anaemia", "anaemia": "Count"})

fig.show()
diabetes_dis = data["diabetes"].value_counts().reset_index()

fig = px.bar(diabetes_dis, x="index", y="diabetes", title="Diabetes Distribution", 

             width=750, labels={"index": "Diabetes", "diabetes": "Count"})

fig.show()
hbp_dis = data["high_blood_pressure"].value_counts().reset_index()

fig = px.bar(hbp_dis, x="index", y="high_blood_pressure", title="High Blood Pressure Distribution",

             width=750, labels={"index": "High Blood Pressure", "high_blood_pressure": "Count"})

fig.show()
sex_dis = data["sex"].value_counts().reset_index()

fig = px.bar(sex_dis, x="index", y="sex", title="Sex Distribution",

             width=750, labels={"index": "Sec", "sex": "Count"})

fig.show()
smooking_dis = data["smoking"].value_counts().reset_index()

fig = px.bar(smooking_dis, x="index", y="smoking", title="Sex Distribution",

             width=750, labels={"index": "Smooking", "smoking": "Count"})

fig.show()
death_dis = data["DEATH_EVENT"].value_counts().reset_index()

fig = px.bar(death_dis, x="index", y="DEATH_EVENT", title="DEATH EVENT Distribution",

             width=750, labels={"index": "DEATH_EVENT", "DEATH_EVENT": "Count"})

fig.show()
fig = px.pie(data, values='DEATH_EVENT',names='sex', title='GENDER',

      width=680, height=480)

fig.show()
f, ax = plt.subplots(figsize=(14,14))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)

plt.show()
sns.pairplot(data[['age', 'creatinine_phosphokinase',

       'ejection_fraction', 'platelets',

       'serum_creatinine', 'serum_sodium','time',

       'DEATH_EVENT']], hue="DEATH_EVENT")
inp_data = data.drop(data[['DEATH_EVENT']], axis=1)

out_data = data[['DEATH_EVENT']]



scaler = StandardScaler()

inp_data = scaler.fit_transform(inp_data)



X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, random_state=42)
print("X_train Shape : ", X_train.shape)

print("X_test Shape  : ", X_test.shape)

print("y_train Shape : ", y_train.shape)

print("y_test Shape  : ", y_test.shape)
def weightInitialization(n_features):

    w = np.zeros((1, n_features))

    b = 0

    return w,b
def sigmoid_activation(result):

    final_result = 1/(1 + np.exp(-result))

    return final_result
def model_optimize(w, b, X, Y):

    m = X.shape[0]

    

    # Prediction

    final_result = sigmoid_activation(np.dot(w,X.T) + b)

    cost = (-1/m)*(np.sum(Y.T * np.log(final_result)) + ((1-Y.T) * (np.log(1-final_result))))

    

    # Gradient Calculation

    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T)) # look down (photo)

    db = (1/m)*(np.sum(final_result-Y.T))

    

    grads = {

        "dw": dw,

        "db": db

    }

    

    return grads, cost
def model_predict(w, b, X, Y, learning_rate, no_iterations):

    costs = []

    for i in range(no_iterations):

        grads, cost = model_optimize(w, b, X, Y)

        dw = grads['dw']

        db = grads['db']

        

        w = w - (learning_rate * dw.T) # look up (photo)

        b = b - (learning_rate * db)

        

        if (i % 100 == 0):

            costs.append(cost)

            

    # final parameters

    coeff = {"w":w, "b":b}

    gradient = {"dw":dw, "db":db}

    

    return coeff, gradient, costs
def predict(final_pred, m):

    y_pred = np.zeros((1,m))

    for i in range(final_pred.shape[1]):

        if final_pred[0][i] > 0.5:

            y_pred[0][i] = 1

    return y_pred
# Get number of features

n_features = X_train.shape[1]

print('Number of Features: {}'.format(n_features))



w, b = weightInitialization(n_features)

# Gradient Descent

coeff, gradient, costs = model_predict(w, b, X_train, y_train.values.reshape(-1,1), learning_rate=0.0001,no_iterations=4500)

# Final Prediction

w = coeff['w']

b = coeff['b']

print('Optimized weights: {}'.format(w))

print('Optimized intercept: {}'.format(b))



final_train_pred = sigmoid_activation(np.dot(w,X_train.T)+b)

final_test_pred = sigmoid_activation(np.dot(w,X_test.T)+b)



print("="*60)



y_train_pred = predict(final_train_pred, X_train.shape[0])

print('Training Accuracy             : {:.4f}'.format(accuracy_score(y_train_pred.T, y_train)))



y_test_pred = predict(final_test_pred, X_test.shape[0])

print('Test Accuracy                 : {:.4f}'.format(accuracy_score(y_test_pred.T, y_test)))



print('Logistic Regression f1-score  : {:.4f}'.format(f1_score(y_test_pred.T, y_test)))

print('Logistic Regression precision : {:.4f}'.format(precision_score(y_test_pred.T, y_test)))

print('Logistic Regression recall    : {:.4f}'.format(recall_score(y_test_pred.T, y_test)))

print("\n",classification_report(y_test_pred.T, y_test))
cf_matrix = confusion_matrix(y_test_pred.T, y_test)

sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression



sms = SMOTE(random_state=12345)

X_res, y_res = sms.fit_sample(inp_data, out_data)



print("X_train Shape : ", X_train.shape)

print("X_test Shape  : ", X_test.shape)

print("y_train Shape : ", y_train.shape)

print("y_test Shape  : ", y_test.shape)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {}'.format(logreg.score(X_test, y_test)))

print('Logistic Regression f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))

print('Logistic Regression precision : {:.4f}'.format(precision_score(y_pred, y_test)))

print('Logistic Regression recall    : {:.4f}'.format(recall_score(y_pred, y_test)))

print("\n",classification_report(y_pred, y_test))
cf_matrix = confusion_matrix(y_pred, y_test)

sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure(figsize=(10,6))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Reporting')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()