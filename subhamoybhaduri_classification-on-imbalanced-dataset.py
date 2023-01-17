import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
%matplotlib inline
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")
data.head()
data.info()
data['Class'].value_counts()
sns.countplot(x='Class', data=data)
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(data, test_size=0.25, random_state=42)
print(train_data.shape, val_data.shape)
train_label = train_data['Class']
val_label   = val_data['Class']
train_data  = train_data.drop(['Class'], axis=1)
val_data    = val_data.drop(['Class'], axis=1)
print(train_data.shape, val_data.shape, train_label.shape, val_label.shape)
train_data.head(2)
val_data.head(2)
from sklearn.preprocessing import StandardScaler

std_scaler_Time   = StandardScaler()
std_scaler_Amount = StandardScaler()

train_data['Time']   = std_scaler_Time.fit_transform(train_data[['Time']])
train_data['Amount'] = std_scaler_Amount.fit_transform(train_data[['Amount']])

val_data['Time']   = std_scaler_Time.transform(val_data[['Time']])
val_data['Amount'] = std_scaler_Amount.transform(val_data[['Amount']])
train_data.head(2)
val_data.head(2)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, classification_report

def model_def(model, model_name, m_train_data, m_train_label):
    model.fit(m_train_data, m_train_label)
    s = "predict_"
    p = s + model_name
    p = model.predict(m_train_data)
    cm = confusion_matrix(m_train_label, p)
    print("Confusion Matrix: \n", cm)
    cr = classification_report(m_train_label, p, target_names=['Not Fraud', 'Fraud'])
    print("Classification Report: \n", cr)
    precision = np.diag(cm)/np.sum(cm, axis=0)
    recall    = np.diag(cm)/np.sum(cm, axis=1)
    F1 = 2 * np.mean(precision) * np.mean(recall)/(np.mean(precision) + np.mean(recall))
    cv_score = cross_val_score(model, m_train_data, m_train_label, cv=10, scoring='recall')
    print("Mean CV Score     :", cv_score.mean())
    print("Std Dev CV Score  :", cv_score.std())
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=0.5)
model_def(logreg, "logreg", train_data, train_label)
val_data_logreg = logreg.predict(val_data)
print("Logistic Regression: \n", confusion_matrix(val_label, val_data_logreg))
from imblearn.under_sampling import NearMiss

print("Before Undersampling, counts of label '1': {}".format(sum(train_label == 1))) 
print("Before Undersampling, counts of label '0': {} \n".format(sum(train_label == 0))) 
  
nr = NearMiss() 
  
train_data_miss, train_label_miss = nr.fit_sample(train_data, train_label.ravel()) 
  
print("After Undersampling, counts of label '1': {}".format(sum(train_label_miss == 1))) 
print("After Undersampling, counts of label '0': {}".format(sum(train_label_miss == 0))) 
logreg_miss = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=0.5)
model_def(logreg_miss, "logreg_miss", train_data_miss, train_label_miss)
val_data_miss   = logreg_miss.predict(val_data)
print("Logistic Regression - Undersampling: \n", confusion_matrix(val_label, val_data_miss))
from imblearn.over_sampling import SMOTE

print("Before Oversampling, counts of label '1': {}".format(sum(train_label == 1))) 
print("Before Oversampling, counts of label '0': {} \n".format(sum(train_label == 0))) 
  
sm = SMOTE(random_state=42) 
  
train_data_SMOTE, train_label_SMOTE = sm.fit_sample(train_data, train_label.ravel()) 
  
print("After Oversampling, counts of label '1': {}".format(sum(train_label_SMOTE == 1))) 
print("After Oversampling, counts of label '0': {}".format(sum(train_label_SMOTE == 0))) 
logreg_SMOTE = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=0.5)
model_def(logreg_SMOTE, "logreg_SMOTE", train_data_SMOTE, train_label_SMOTE)
val_data_SMOTE  = logreg_SMOTE.predict(val_data)
print("Logistic Regression - Oversampling: \n", confusion_matrix(val_label, val_data_SMOTE))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# Applying Neural Network
def build_classifier():
    classifier = Sequential([Dense(128, activation='relu', input_shape=(train_data_SMOTE.shape[1], )),
                             Dropout(rate=0.1),
                             Dense(64, activation='relu'),
                             Dropout(rate=0.1),
                             Dense(1, activation='sigmoid')])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision', 'Recall'])
    print(classifier.summary())
    return classifier

model = KerasClassifier(build_fn=build_classifier)
history = model.fit(train_data_SMOTE, train_label_SMOTE,
                    batch_size=30,
                    epochs=10,
                    validation_data=(val_data, val_label))
val_data_Neural = model.predict(val_data)
print("Artificial Neural Network: \n", confusion_matrix(val_label, val_data_Neural))
