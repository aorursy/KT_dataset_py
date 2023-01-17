import pandas as pd  # data processing
import numpy as np   # linear algebra
import matplotlib.pyplot as plt  #Plotting
import seaborn as sns
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
data[['Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']] = data[['Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)
data.head()
data_nan = data.isna().sum()
data_nan = pd.DataFrame(data_nan, columns=['NaN count'])
data_nan
data_nan = data_nan.reset_index()
plt.figure(figsize = (12,8))
plot = sns.barplot(x = 'index', y = 'NaN count', data = data_nan, palette = 'rocket')
for p in plot.patches:
    plot.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize = 12)

plt.xticks(fontsize = 12, rotation=40)
plt.xlabel("Variable", fontsize=15)
plt.yticks(fontsize = 12)
plt.ylabel("NaN Count", fontsize=15)
plt.title('NaN Count of variables', fontsize=20)
plt.show()
data.info()
plt.figure(figsize=(12,8))
plot_outcome = sns.countplot(x = 'Outcome', data = data, palette="husl")
for p in plot_outcome.patches:
    plot_outcome.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize = 12)

plt.title('Count of Outcome', fontsize = 20)
plt.xlabel('Outcome', fontsize = 15)
plt.xticks(np.arange(2), ('No', 'Yes'), fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
plt_preg = sns.countplot(x = 'Pregnancies', data = data, palette="husl")
for p in plt_preg.patches:
    plt_preg.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize = 12)

plt.title('Count of Number of Pregnancies', fontsize = 20)
plt.xlabel('Number of Pregnancies', fontsize = 15)
plt.xticks(np.arange(18), fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['Glucose'], kde = True, color = 'Orange')
plt.title('Histogram of Glucose', fontsize = 20)
plt.xlabel('Glucose Level', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['BloodPressure'], kde = True, color = 'Purple')
plt.title('Histogram of BloodPressure', fontsize = 20)
plt.xlabel('BloodPressure Level', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['SkinThickness'], kde = True, color = 'Red')
plt.title('Histogram of SkinThickness', fontsize = 20)
plt.xlabel('SkinThickness Level', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['Insulin'], kde = True, color = 'Orange')
plt.title('Histogram of Insulin', fontsize = 20)
plt.xlabel('Insulin Level', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['BMI'], kde = True, color = 'Blue')
plt.title('Histogram of BMI', fontsize = 20)
plt.xlabel('BMI Value', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['DiabetesPedigreeFunction'], kde = True, color = 'Brown')
plt.title('Histogram of Diabetes Pedigree Function', fontsize = 20)
plt.xlabel('Diabetes Pedigree Function Value', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(data['Age'], kde = True, color = 'Black')
plt.title('Histogram of Age', fontsize = 20)
plt.xlabel('Age in years', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['Age'],  kde = True, label = 'No', color='red')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['Age'],  kde = True, label = 'Yes', color='blue')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Age Vs. Outcome', fontsize = 20)
plt.xlabel('Age', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['DiabetesPedigreeFunction'],  kde = True, label = 'No', color='green')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['DiabetesPedigreeFunction'],  kde = True, label = 'Yes', color='orange')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Diabetes Pedigree Function Vs. Outcome', fontsize = 20)
plt.xlabel('Diabetes Pedigree Function Value', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['BMI'],  kde = True, label = 'No', color='purple')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['BMI'],  kde = True, label = 'Yes', color='red')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('BMI Vs. Outcome', fontsize = 20)
plt.xlabel('BMI Value', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['Insulin'],  kde = True, label = 'No', color='black')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['Insulin'],  kde = True, label = 'Yes', color='yellow')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Insulin Vs. Outcome', fontsize = 20)
plt.xlabel('Insulin Level', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['SkinThickness'],  kde = True, label = 'No', color='red')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['SkinThickness'],  kde = True, label = 'Yes', color='green')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('SkinThickness Vs. Outcome', fontsize = 20)
plt.xlabel('SkinThickness', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['BloodPressure'],  kde = True, label = 'No', color='crimson')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['BloodPressure'],  kde = True, label = 'Yes', color='gold')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Blood Pressure level Vs. Outcome', fontsize = 20)
plt.xlabel('Blood Pressure Level', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
df1 = data[data['Outcome'] == 0]
sns.distplot(df1['Glucose'],  kde = True, label = 'No', color='teal')
df2 = data[data['Outcome'] == 1]
sns.distplot(df2['Glucose'],  kde = True, label = 'Yes', color='peru')

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Glucose level Vs. Outcome', fontsize = 20)
plt.xlabel('Glucose Level', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
plt.figure(figsize=(12,8))
plt_preg = sns.countplot(x = 'Pregnancies', data = data, palette="rocket", hue = 'Outcome')
for p in plt_preg.patches:
    plt_preg.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize = 12)

plt.title('Number of Pregnancies Vs. Outcome', fontsize = 20)
plt.xlabel('Number of Pregnancies', fontsize = 15)
plt.xticks(np.arange(18), fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend(['No','Yes'], loc = 'upper right', fontsize = 15)
plt.show()
plt.subplots(figsize = (12,8))
sns.set(font_scale = 1.5)
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Plot', fontsize = 20)
plt.show()
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2, weights="uniform")
data_imputed = imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
data_imputed.head()
data_imputed['Outcome'].value_counts()
from sklearn.utils import resample

df_majority = data_imputed[data_imputed.Outcome==0]
df_minority = data_imputed[data_imputed.Outcome==1]

data_minority_upsampled = resample(df_minority, replace=True, n_samples=500, random_state=123) 
data_upsampled = pd.concat([df_majority, data_minority_upsampled])

data_upsampled['Outcome'].value_counts()
x = data_upsampled.drop(columns = ['Outcome'], axis=1)
y = data_upsampled.Outcome
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
accuracy = pd.DataFrame(columns=['classifiers', 'accuracy','auc'])
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
alphas = np.linspace(1,10,100)
ridgeClassifiercv = LogisticRegressionCV(penalty = 'l2', Cs = 1/alphas, solver = 'liblinear')
ridgeClassifiercv.fit(x_train, y_train)
ridgeClassifiercv.C_  #Inverse of best alpha value
LR = LogisticRegression(penalty = 'l2', C = ridgeClassifiercv.C_[0], solver = 'liblinear')
LR.fit(x_train, y_train)
y_predLR = LR.predict(x_test)
accLR = accuracy_score(y_test, y_predLR)
clf_roc_auc = roc_auc_score(y_test, LR.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, LR.predict_proba(x_test)[:,1])
result_table = result_table.append({'classifiers':'Logistics Ridge', 'fpr':fpr, 'tpr':tpr, 'auc':clf_roc_auc}, ignore_index=True)
plt.figure(figsize = (12,8))
plt.plot(fpr, tpr, label='Logistic Ridge (area = %0.2f)' % clf_roc_auc, lw = 2)
plt.plot([0, 1], [0, 1],'r--', lw = 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('Receiver operating characteristic curve for Logistic Ridge Classification', fontsize = 20)
plt.legend(loc="lower right", fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
accuracy = accuracy.append({'classifiers':'Logistic Ridge', 'accuracy':accLR, 'auc':clf_roc_auc}, ignore_index=True)
lassoClassifiercv = LogisticRegressionCV(penalty = 'l1', Cs = 1/alphas, solver = 'liblinear')
lassoClassifiercv.fit(x_train, y_train)
lassoClassifiercv.C_
LL = LogisticRegression(penalty = 'l1', C = lassoClassifiercv.C_[0], solver = 'liblinear')
LL.fit(x_train, y_train)
y_predLL = LL.predict(x_test)
accLL = accuracy_score(y_test, y_predLL)
clf_roc_auc = roc_auc_score(y_test, LL.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, LL.predict_proba(x_test)[:,1])
result_table = result_table.append({'classifiers':'Logistics Lasso', 'fpr':fpr, 'tpr':tpr, 'auc':clf_roc_auc}, ignore_index=True)
plt.figure(figsize = (12,8))
plt.plot(fpr, tpr, label='Logistic Lasso (area = %0.2f)' % clf_roc_auc, lw = 2)
plt.plot([0, 1], [0, 1],'r--', lw = 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('Receiver operating characteristic curve for Logistic Lasso Classification', fontsize = 20)
plt.legend(loc="lower right", fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
accuracy = accuracy.append({'classifiers':'Logistic Lasso', 'accuracy':accLL, 'auc':clf_roc_auc}, ignore_index=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
np.random.seed(123)
model = Sequential()
model.add(Dense(8, activation = 'tanh', input_dim = 8))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 10, epochs = 10, verbose = 1)
y_pred = model.predict_classes(x_test)
accNN = accuracy_score(y_test, y_pred)
clf_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
result_table = result_table.append({'classifiers':'Neural Network', 'fpr':fpr, 'tpr':tpr, 'auc':clf_roc_auc}, ignore_index=True)
plt.figure(figsize = (12,8))
plt.plot(fpr, tpr, label='Neural Network (area = %0.2f)' % clf_roc_auc, lw = 2)
plt.plot([0, 1], [0, 1],'r--', lw = 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('Receiver operating characteristic curve for Neural Network', fontsize = 20)
plt.legend(loc="lower right", fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
accuracy = accuracy.append({'classifiers':'Neural Network', 'accuracy':accNN, 'auc':clf_roc_auc}, ignore_index=True)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
k = range(1,30)
for i in k:
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train, y_train)
  pred_i = knn.predict(x_test)
  error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12,8))
plt.plot(k,error_rate, color='black', linestyle='dashed', marker='o', markerfacecolor='pink')
plt.title('Error Rate vs. K Value', fontsize = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('K', fontsize = 15)
plt.ylabel('Error Rate', fontsize = 15)
plt.show()
KNN = KNeighborsClassifier(n_neighbors=1, p=2, metric='euclidean')
KNN.fit(x_train,y_train)
y_predKNN = KNN.predict(x_test)
accKNN = accuracy_score(y_test, y_predKNN)
clf_roc_auc = roc_auc_score(y_test, KNN.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, KNN.predict_proba(x_test)[:,1])
result_table = result_table.append({'classifiers':'KNN', 'fpr':fpr, 'tpr':tpr, 'auc':clf_roc_auc}, ignore_index=True)
plt.figure(figsize = (12,8))
plt.plot(fpr, tpr, label='K Nearest Neighbour (area = %0.2f)' % clf_roc_auc, lw = 2)
plt.plot([0, 1], [0, 1],'r--', lw = 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('Receiver operating characteristic curve for K Nearest Neighbour Classification', fontsize = 20)
plt.legend(loc="lower right", fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
accuracy = accuracy.append({'classifiers':'KNN', 'accuracy':accKNN, 'auc':clf_roc_auc}, ignore_index=True)
from sklearn.ensemble import RandomForestClassifier
error_rate_RF = []
n = range(1,20)
for i in n:
  RFC = RandomForestClassifier(n_estimators=i)
  RFC.fit(x_train, y_train)
  pred_i = RFC.predict(x_test)
  error_rate_RF.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12,8))
plt.plot(n,error_rate_RF, color='black', linestyle='dashed', marker='o', markerfacecolor='maroon')
plt.title('Error Rate vs. Number of estimators(Trees)', fontsize = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Number of estimators(Trees)', fontsize = 15)
plt.ylabel('Error Rate', fontsize = 15)
plt.show()
RFC = RandomForestClassifier(n_estimators=18)
RFC.fit(x_train, y_train)
y_predRFC = RFC.predict(x_test)
accRF = accuracy_score(y_test, y_predRFC)
clf_roc_auc = roc_auc_score(y_test, RFC.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, RFC.predict_proba(x_test)[:,1])
result_table = result_table.append({'classifiers':'Random Forest', 'fpr':fpr, 'tpr':tpr, 'auc':clf_roc_auc}, ignore_index=True)
plt.figure(figsize = (12,8))
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % clf_roc_auc, lw = 2)
plt.plot([0, 1], [0, 1],'r--', lw = 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('Receiver operating characteristic curve for Random Forest Classification', fontsize = 20)
plt.legend(loc="lower right", fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
accuracy = accuracy.append({'classifiers':'Random Forest', 'accuracy':accRF, 'auc':clf_roc_auc}, ignore_index=True)
result_table.set_index('classifiers', inplace=True)
accuracy.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(12,8))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='black', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1), fontsize = 12)
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1), fontsize = 12)
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontsize=20)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()
accuracy