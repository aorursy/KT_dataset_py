import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from xgboost import plot_importance

import xgboost as xgb



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve, auc
data = pd.read_csv('../input/creditcard.csv')
data.head()
data.describe().transpose()
data.info()
# Next, Class feature will be examined.

plt.figure(figsize=(10,10))

sns.countplot(

    y="Class", 

    data=data,

    facecolor=(0, 0, 0, 0),

    linewidth=5, 

    edgecolor=sns.color_palette("dark", 2))



plt.title('Fraudulent Transaction Summary')

plt.xlabel('Count')

plt.ylabel('Fraudulent Transaction   Non-Fraudulent Transaction', fontsize=12)
data_value= data["Class"].value_counts()
print(data_value)

print(data_value/284807)
data['Class']= data['Class'].astype('category')
#Distribution of Time

plt.figure(figsize=(15,10))

sns.distplot(data['Time'])
#Distribution of Amount

plt.figure(figsize=(10,10))

sns.distplot(data['Amount'])
data['Hour'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)
#Class vs Amount vs Hour

pd.pivot_table(

    columns="Class", 

    index="Hour", 

    values= 'Amount', 

    aggfunc='count', 

    data=data)
#Hour vs Class

fig, axes = plt.subplots(2, 1, figsize=(15, 10))



sns.countplot(

    x="Hour",

    data=data[data['Class'] == 0], 

    color="#98D8D8",  

    ax=axes[0])

axes[0].set_title("Non-Fraudulent Transaction")





sns.countplot(

    x="Hour",

    data=data[data['Class'] == 1],

    color="#F08030", 

    ax=axes[1])

axes[1].set_title("Fraudulent Transaction")
#Amount vs Hour vs Class

fig, axees = plt.subplots(2, 1, figsize=(15, 10))



plt.title("Non-Fraudulent Transactions")

sns.barplot(

    x='Hour',

    y='Amount', 

    data=data[data['Class'] == 0], 

    palette="ocean", 

    ax=axees[0])



plt.title("Fraudulent Transactions")

sns.barplot(

    x='Hour', 

    y='Amount', 

    data=data[data['Class'] == 1], 

    palette="Reds", 

    ax=axees[1])
#Drop hour feature before continues next analysis.

data=data.drop(['Hour'], axis=1)
data_nonfraud = data[data['Class'] == 0].sample(2000)

data_fraud  = data[data['Class'] == 1]



data_new = data_nonfraud.append(data_fraud).sample(frac=1)

X = data_new.drop(['Class'], axis = 1).values

y = data_new['Class'].values
tsne = TSNE(n_components=2, random_state=42)

X_transformation = tsne.fit_transform(X)
plt.figure(figsize=(10, 10))

plt.title("t-SNE Dimensionality Reduction")



def plot_data(X, y):

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Non_Fraudulent", alpha=0.5, linewidth=0.15, c='#17becf')

    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Fraudulent", alpha=0.5, linewidth=0.15, c='#d62728')

    plt.legend()

    return plt.show()



plot_data(X_transformation, y)
data[['Time', 'Amount']] = StandardScaler().fit_transform(data[['Time', 'Amount']])
corr=data.corr(method='pearson')
plt.figure(figsize=(18, 18))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(

    corr, 

    xticklabels=corr.columns,

    yticklabels=corr.columns, 

    cmap="coolwarm", 

    annot=True, 

    fmt=".2f",

    mask=mask, 

    vmax=.2, 

    center=0,

    square=True, 

    linewidths=1, 

    cbar_kws={"shrink": .5})
# First train and label data created. 

train_data, label_data = data.iloc[:,:-1],data.iloc[:,-1]



#Convert to matrix

data_dmatrix = xgb.DMatrix(data=train_data, label= label_data)
#Split data randomly to train and test subsets.

X_train, X_test, y_train, y_test = train_test_split(

                                    train_data, label_data, test_size=0.3,random_state=42)
## Defining parameters



#grid_param = {'n_estimators': [50, 100, 500],'max_depth': [4, 8], 

            #'max_features': ['auto', 'log2'], 

            #'criterion': ['gini', 'entropy'],

            #'bootstrap': [True, False]}



## Building Grid Search algorithm with cross-validation and F1 score.



#grid_search = GridSearchCV(estimator=xg_class,  

                     #param_grid=grid_param,

                     #scoring='f1',

                     #cv=5,

                     #n_jobs=-1)



## Lastly, finding the best parameters.



#grid_search.fit(X_train, y_train)

#best_parameters = grid_search.best_params_  

#print(best_parameters)
params = {

    'objective':'reg:logistic',

    'colsample_bytree': 0.3,

    'learning_rate': 0.1,

    'bootstrap': True, 

    'criterion': 'gini', 

    'max_depth': 4, 

    'max_features': 'auto', 

    'n_estimators': 50

}

xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)



#Feature importance graph

plt.rcParams['figure.figsize'] = [20, 10]

xgb.plot_importance(xg_reg)
data_model = data.drop(['V13', 'V25', 'Time', 'V20', 'V22', 'V8', 'V15', 'V19', 'V2'], axis=1)
data_under_nonfraud = data_model[data_model['Class'] == 0].sample(15000)

data_under_fraud  = data_model[data_model['Class'] == 1]



data_undersampling = data_under_nonfraud.append(data_under_fraud, 

                                                ignore_index=True, sort=False)
plt.figure(figsize=(10,10))

sns.countplot(y="Class", data=data_undersampling,palette='Dark2')

plt.title('Fraudulent Transaction Summary')

plt.xlabel('Count')

plt.ylabel('Fraudulent Transaction,        Non-Fraudulent Transaction')
# New data will be split randomly to train and test subsets. Train data proportion is 70% and the test data proportion is 30%.



model_train, model_label = data_undersampling.iloc[:,:-1],data_undersampling.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(

                                        model_train, model_label, test_size=0.3, random_state=42)
#5-fold Cross Validation method will be used.



kfold_cv=KFold(n_splits=5, random_state=42, shuffle=True)



for train_index, test_index in kfold_cv.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]
# Define the model as the Random Forest

modelRF = RandomForestClassifier(

    n_estimators=500, 

    criterion = 'gini', 

    max_depth = 4, 

    class_weight='balanced', 

    random_state=42

).fit(X_train, y_train)



# Obtain predictions from the test data 

predict_RF = modelRF.predict(X_test)
# Define the model as the Support Vector Machine

modelSVM = svm.SVC(

    kernel='rbf', 

    class_weight='balanced', 

    gamma='scale', 

    probability=True, 

    random_state=42

).fit(X_train, y_train)



# Obtain predictions from the test data 

predict_SVM = modelSVM.predict(X_test)
# Define the model as the Logistic Regression

modelLR = LogisticRegression(

    solver='lbfgs', 

    multi_class='multinomial',

    class_weight='balanced', 

    max_iter=500, 

    random_state=42

).fit(X_train, y_train)



# Obtain predictions from the test data 

predict_LR = modelLR.predict(X_test)
# Define the model as the Multilayer Perceptron

modelMLP = MLPClassifier(

    solver='lbfgs', 

    activation='logistic', 

    hidden_layer_sizes=(100,),

    learning_rate='constant', 

    max_iter=1500, 

    random_state=42

).fit(X_train, y_train)



# Obtain predictions from the test data 

predict_MLP = modelMLP.predict(X_test)
RF_matrix = confusion_matrix(y_test, predict_RF)

SVM_matrix = confusion_matrix(y_test, predict_SVM)

LR_matrix = confusion_matrix(y_test, predict_LR)

MLP_matrix = confusion_matrix(y_test, predict_MLP) 
fig, ax = plt.subplots(1, 2, figsize=(15, 8))



sns.heatmap(RF_matrix, annot=True, fmt="d",cbar=False, cmap="Paired", ax = ax[0])

ax[0].set_title("Random Forest", weight='bold')

ax[0].set_xlabel('Predicted Labels')

ax[0].set_ylabel('Actual Labels')

ax[0].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

ax[0].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])



sns.heatmap(SVM_matrix, annot=True, fmt="d",cbar=False, cmap="Dark2", ax = ax[1])

ax[1].set_title("Support Vector Machine", weight='bold')

ax[1].set_xlabel('Predicted Labels')

ax[1].set_ylabel('Actual Labels')

ax[1].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

ax[1].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])





fig, axe = plt.subplots(1, 2, figsize=(15, 8))



sns.heatmap(LR_matrix, annot=True, fmt="d",cbar=False, cmap="Pastel1", ax = axe[0])

axe[0].set_title("Logistic Regression", weight='bold')

axe[0].set_xlabel('Predicted Labels')

axe[0].set_ylabel('Actual Labels')

axe[0].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

axe[0].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])



sns.heatmap(MLP_matrix, annot=True, fmt="d",cbar=False, cmap="Pastel1", ax = axe[1])

axe[1].set_title("Multilayer Perceptron", weight='bold')

axe[1].set_xlabel('Predicted Labels')

axe[1].set_ylabel('Actual Labels')

axe[1].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

axe[1].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

print("Classification_RF:")

print(classification_report(y_test, predict_RF))

print("Classification_SVM:")

print(classification_report(y_test, predict_SVM))

print("Classification_LR:")

print(classification_report(y_test, predict_LR))

print("Classification_MLP:")

print(classification_report(y_test, predict_MLP))
#RF AUC

rf_predict_probabilities = modelRF.predict_proba(X_test)[:,1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_predict_probabilities)

rf_roc_auc = auc(rf_fpr, rf_tpr)



#SVM AUC

svm_predict_probabilities = modelSVM.predict_proba(X_test)[:,1]

svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_predict_probabilities)

svm_roc_auc = auc(svm_fpr, svm_tpr)



#LR AUC

lr_predict_probabilities = modelLR.predict_proba(X_test)[:,1]

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_predict_probabilities)

lr_roc_auc = auc(lr_fpr, lr_tpr)



#MLP AUC

mlp_predict_probabilities = modelMLP.predict_proba(X_test)[:,1]

mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_predict_probabilities)

mlp_roc_auc = auc(mlp_fpr, mlp_tpr)

plt.figure()

plt.plot(rf_fpr, rf_tpr, color='red',lw=2,

         label='Random Forest (area = %0.2f)' % rf_roc_auc)



plt.plot(svm_fpr, svm_tpr, color='blue',lw=2, 

         label='Support Vector Machine (area = %0.2f)' % svm_roc_auc)



plt.plot(lr_fpr, lr_tpr, color='green',lw=2, 

         label='Logistic Regression (area = %0.2f)' % lr_roc_auc)



plt.plot(mlp_fpr, mlp_tpr, color='orange',lw=2, 

         label='Multilayer Perceptron (area = %0.2f)' % mlp_roc_auc)



plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()

print("Average precision score of Logistic Regression", average_precision_score(y_test, modelLR.predict_proba(X_test)[:,1]))

print("Average precision score of Random Forest", average_precision_score(y_test, modelRF.predict_proba(X_test)[:,1]))

print("Average precision score of Multilayer Perceptron", average_precision_score(y_test, modelMLP.predict_proba(X_test)[:,1]))