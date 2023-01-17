import math

import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

# Read our data

df = pd.read_csv("../input/palo-it-tech/PALO_IT_heart.csv")
# Our data

df
print("The numbers of data is " + str(df.shape[0]))

print("Each data has " + str(df.shape[1]) + " dimensions")

def clean(i, df, count):

    

    #age

    if type(df.at[i, 'age']) == str and not df.at[i, 'age'].isdigit:

        df = df.drop([i])

        return df, count

    if type(df.at[i, 'age']) == float and math.isnan(df.at[i, 'age']):

        df = df.drop([i])

        return df, count



    #sex:

    if df.at[i, 'sex'] == "female":

        df.at[i, 'sex'] = 0

    elif df.at[i, 'sex'] == "male":

        df.at[i, 'sex'] = 1

    else:

        df = df.drop([i])

        return df, count





    #cp:

    if type(df.at[i, 'cp']) == str and not df.at[i, 'cp'].isdigit:

        df = df.drop([i])

        return df, count





    #trestbps:

    if type(df.at[i, 'trestbps']) == str and not df.at[i, 'trestbps'].isdigit:

        df = df.drop([i])

        return df, count





    #chol:

    if type(int(df.at[i, 'chol'])) != int:

        df = df.drop([i])

        return df, count



        

    #fbs:    

    if df.at[i, 'fbs'] == "FALSE":

        df.at[i, 'fbs'] = 0

    elif df.at[i, 'fbs'] == "TRUE":

        df.at[i, 'fbs'] = 1

    else:

        df = df.drop([i])

        return df, count



        

    #restecg:

    if type(int(df.at[i, 'restecg'])) != int:

        df = df.drop([i])

        return df, count





    #thalach:

    if type(int(df.at[i, 'thalach'])) != int:

        df = df.drop([i])

        return df, count



    #exang:

    if type(int(df.at[i, 'exang'])) != int:

        df = df.drop([i])   

        return df, count





    #oldpeak:

    if type(int(df.at[i, 'oldpeak'])) != int or type(float(df.at[i, 'oldpeak'])) != float:

        df = df.drop([i])

        return df, count





    #slope:

    if type(int(df.at[i, 'slope'])) != int:

        df = df.drop([i])

        return df, count





    #ca:

    if type(int(df.at[i, 'ca'])) != int:

        df = df.drop([i])

        return df, count





    #thal:

    if df.at[i, 'thal'] == "normal":

        df.at[i, 'thal'] = 1

    elif df.at[i, 'thal'] == "fixed defect":

        df.at[i, 'thal'] = 2

    elif df.at[i, 'thal'] == "reversable defect":

        df.at[i, 'thal'] = 3

    else:

        df = df.drop([i])

        return df, count





    #target:

    if df.at[i, 'target'] == "FALSE":

        df.at[i, 'target'] = 0

    elif df.at[i, 'target'] == "TRUE":

        df.at[i, 'target'] = 1

    else:

        df = df.drop([i])

    

    count -= 1 

    return df, count



        

        

        
count = 0

for index in range(df.shape[0]):

    ori_count = count

    df, count = clean(index, df, count)

    if ori_count == count:

        print("DATA " + "{:>3d}".format(index) + " has been deleted.")

    count += 1



print()

print("Total deleted data : " + str(count))

df= df.astype(float)
sns.countplot(x="target", data=df, palette="bwr").set_title('Target numbers')

plt.show()

print(df.target.value_counts())

countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients don't have Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
sns.countplot(x='sex', data=df).set_title('Disease distribution in sex')

plt.xlabel("Sex (0 : female, 1= : male)")

plt.show()

countFemale = len(df[df.sex == 0])

countMale   = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.legend(["Female", "Male"])

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(df.slope,df.target).plot(kind="bar")

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The slope of the Peak exercise ST segment ')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.legend(["Disease", "No Disease"])

plt.show()
pd.crosstab(df.fbs,df.target).plot(kind="bar")

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.legend(["Don't have Disease", "Have Disease"])

plt.show()
pd.crosstab(df.cp,df.target).plot(kind="bar")

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.legend(["Don't have Disease", "Have Disease"])

plt.show()
a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")

d = pd.get_dummies(df['ca'], prefix = "ca")
frames = [df, a, b, c, d]

df = pd.concat(frames, axis = 1)

df.head()
df = df.drop(columns = ['cp', 'thal', 'slope', 'ca'])

df.head()
# Store original df for later feature engineering

df_ori = df
y = df.target.values

x_data = df.drop(['target'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)



#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test  = x_test.T

y_test  = y_test.T



print(x_train.shape)
#initialize

def initialize(dimension):

    

    weight = np.full((dimension,1),0.01)

    bias   = 0.0

    return weight,bias
def sigmoid(x):

    return 1. / (1. + np.exp(-x))
def forwardBackward(weight,bias,x_train,y_train):

    # Forward

    

    y_head = sigmoid(np.dot(weight.T,x_train) + bias)

    loss   = -(y_train * np.log(y_head) + (1-y_train) * np.log(1 - y_head))

    cost   = np.sum(loss) / x_train.shape[1]

    

    # Backward

    derivative_weight = np.dot(x_train, ((y_head - y_train).T)) / x_train.shape[1]

    derivative_bias   = np.sum( y_head-y_train) / x_train.shape[1]

    gradients         = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}

    

    return cost,gradients
def update(weight,bias,x_train,y_train,learningRate,iteration) :

    costList = []

    index = []

    

    #for each iteration, update weight and bias values

    for i in range(iteration):

        cost, gradients = forwardBackward(weight,bias,x_train,y_train)

        weight          = weight - learningRate * gradients["Derivative Weight"]

        bias            = bias   - learningRate * gradients["Derivative Bias"]

        

        costList.append(cost)

        index.append(i)



    parameters = {"weight": weight,"bias": bias}

    

    print("iteration:",iteration)

    print("cost:",cost)



    plt.plot(index,costList)

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()



    return parameters, gradients
def predict(weight,bias,x_test):

    z = np.dot(weight.T,x_test) + bias

    y_head = sigmoid(z)



    y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(y_head.shape[1]):

        if y_head[0,i] <= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):

    dimension   = x_train.shape[0]

    weight,bias = initialize(dimension)

    

    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)



    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)

    

    print("Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
logistic_regression(x_train,y_train,x_test,y_test,1,100)
accuracies_train = {}

accuracies_test = {}



lr = LogisticRegression(max_iter=100)

lr.fit(x_train.T,y_train.T)



acc_train = lr.score(x_train.T,y_train.T)*100

acc_test = lr.score(x_test.T,y_test.T)*100



accuracies_train['Logistic Reg'] = acc_train

accuracies_test['Logistic Reg'] = acc_test

print("Train Accuracy {:.2f}%".format(acc_train))

print("Test Accuracy {:.2f}%".format(acc_test))
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Train Accuracy: {:.2f}%".format(2, knn.score(x_train.T, y_train.T)*100))

print("{} NN Test Accuracy: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# Find best k value

scoreList_train = []

scoreList_test = []

n_max = 20

for i in range(2,n_max):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(x_train.T, y_train.T)

    scoreList_train.append(knn2.score(x_train.T, y_train.T))

    scoreList_test.append(knn2.score(x_test.T, y_test.T))

    

plt.plot(range(2,n_max), scoreList_train)

plt.plot(range(2,n_max), scoreList_test)

plt.xticks(np.arange(1,n_max,1))

plt.xlabel("K value")

plt.ylabel("Accuracy")

plt.legend(["Train", "Test"])

plt.show()



acc_train = max(scoreList_train)*100

acc_test = max(scoreList_test)*100

accuracies_train['KNN'] = acc_train

accuracies_test['KNN'] = acc_test



print("Maximum KNN Accuracy is {:.2f}%".format(acc_test))
from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state = 1, probability = True, max_iter = 100)

svm.fit(x_train.T, y_train.T)



acc_train = svm.score(x_train.T,y_train.T)*100

acc_test  = svm.score(x_test.T,y_test.T)*100



accuracies_train['SVM'] = acc_train

accuracies_test['SVM']  = acc_test

print("Train Accuracy of SVM Algorithm: {:.2f}%".format(acc_train))

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc_test))
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc_train = nb.score(x_train.T, y_train.T) * 100

acc_test  = nb.score(x_test.T, y_test.T)   * 100



accuracies_train['Naive Bayes'] = acc_train

accuracies_test['Naive Bayes']  = acc_test



print("Train Accuracy of Naive Bayes: {:.2f}%".format(acc_train))

print("Test Accuracy of Naive Bayes: {:.2f}%".format(acc_test))
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier(max_depth = 3)

dtc.fit(x_train.T, y_train.T)



acc_train = dtc.score(x_train.T, y_train.T) * 100

acc_test  = dtc.score(x_test.T, y_test.T)    * 100



accuracies_train['Decision Tree'] = acc_train

accuracies_test['Decision Tree']  = acc_test



print("Decision Tree Test Accuracy {:.2f}%".format(acc_train))

print("Decision Tree Test Accuracy {:.2f}%".format(acc_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_depth = 5 ,n_estimators = 50, random_state = 1)

rf.fit(x_train.T, y_train.T)



acc_train = rf.score(x_train.T,y_train.T) * 100

acc_test  = rf.score(x_test.T,y_test.T) * 100



accuracies_train['Random Forest'] = acc_train

accuracies_test['Random Forest']  = acc_test



print("Random Forest Algorithm Train Accuracy Score : {:.2f}%".format(acc_train))

print("Random Forest Algorithm Test Accuracy Score : {:.2f}%".format(acc_test))
# applying XGBoost



from xgboost import XGBClassifier



param_dist = {'n_estimators': 10,

              'learning_rate': 0.05}



xg = XGBClassifier(**param_dist)



# Trian

xg.fit(x_train.T, y_train.T)



# Prediction

y_pred_train = xg.predict(x_train.T)

y_pred       = xg.predict(x_test.T)



from sklearn.metrics import accuracy_score



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test  = accuracy_score(y_test.T, y_pred.T) * 100



accuracies_train['XGBoost'] = acc_train

accuracies_test['XGBoost']  = acc_test



print("XGBoost Train Accuracy Score : {:.2f}%".format(acc_train))

print("XGBoost Test Accuracy Score : {:.2f}%".format(acc_test))

import lightgbm as lgb



params = {'learning_rate':0.1,

          'n_estimators': 10}



LGB = lgb.LGBMClassifier(**params)



# Trian

model = LGB.fit(x_train.T, y_train.T )



# Prediction

y_pred_train = LGB.predict(x_train.T)

y_pred = LGB.predict(x_test.T)



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test = accuracy_score(y_test.T, y_pred.T) * 100



accuracies_train['LightGBM'] = acc_train

accuracies_test['LightGBM'] = acc_test



print("LightGBM Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM Test Accuracy Score : {:.2f}%".format(acc_test))
Total_accuracy = pd.DataFrame([accuracies_train,accuracies_test] , index=['Train', 'Test'])

print(Total_accuracy)
acc_diagram = Total_accuracy.T

ax = acc_diagram.plot(kind='bar', legend=["Train", "Test"], figsize =[15, 4], fontsize =12, width=0.7, rot=0)

ax.get_legend().set_bbox_to_anchor((1, 1))

ax.set_xlabel("Algorithm")

ax.set_ylabel("Accuracy %")

for p in ax.patches:

    ax.annotate("%.2f" % (p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    

plt.show()

# Predicted values



knn11 = KNeighborsClassifier(n_neighbors = 11)

knn11.fit(x_train.T, y_train.T)



y_head_lr       = lr.predict(x_test.T)

y_head_knn      = knn11.predict(x_test.T)

y_head_svm      = svm.predict(x_test.T)

y_head_nb       = nb.predict(x_test.T)

y_head_dtc      = dtc.predict(x_test.T)

y_head_rf       = rf.predict(x_test.T)

y_head_xgboost  = xg.predict(x_test.T)

y_head_lightgbm = LGB.predict(x_test.T)

ALL_MODEL = {'lr':lr, 'knn11':knn11, 'svm':svm, 'nb':nb, 'dtc':dtc, 'rf':rf, 'LightGBM':LGB, "XGBoost":xg}
from sklearn.metrics import confusion_matrix



cm_lr       = confusion_matrix(y_test,y_head_lr)

cm_knn      = confusion_matrix(y_test,y_head_knn)

cm_svm      = confusion_matrix(y_test,y_head_svm)

cm_nb       = confusion_matrix(y_test,y_head_nb)

cm_dtc      = confusion_matrix(y_test,y_head_dtc)

cm_rf       = confusion_matrix(y_test,y_head_rf)

cm_xgboost  = confusion_matrix(y_test,y_head_xgboost)

cm_lightgbm = confusion_matrix(y_test,y_head_lightgbm)





plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(3,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,7)

plt.title("LightGBM Matrix")

sns.heatmap(cm_lightgbm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,8)

plt.title("XGBoost Matrix")

sns.heatmap(cm_xgboost,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()
from sklearn import metrics



# Set up for store [name, acc, p, r, f1, auc]

df_result = pd.DataFrame(columns=('Model', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'))

row = 0



# Set up for ROC figure

plt.figure(figsize=(24,12))

plt.suptitle("ROC CURVE",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



for name, clf in ALL_MODEL.items():

    y_test_pred = clf.predict(x_test.T)



    acc = metrics.accuracy_score(y_test, y_test_pred)

    p   = metrics.precision_score(y_test, y_test_pred)

    r   = metrics.recall_score(y_test, y_test_pred)

    f1  = metrics.f1_score(y_test, y_test_pred)



    y_test_proba = clf.predict_proba(x_test.T)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba[:, 1])

    auc = metrics.auc(fpr, tpr)



    df_result.loc[row] = [name, acc, p, r, f1, auc]

    

    row += 1     

    

    plt.subplot(3,3,row)

    plt.title(name)

    

    #Plot curve

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)

    

    # diagonal

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    

    # Figure limitation

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic of '+name)

    plt.legend(loc="lower right")



plt.show()
# applying lightGBM

import lightgbm as lgb





params = {'learning_rate':0.5,

          'n_estimators': 100,

          'metric': ['logloss']}



LGB_100 = lgb.LGBMClassifier(**params)



model_100 = LGB_100.fit(x_train.T, y_train.T, eval_set = [(x_train.T, y_train.T), (x_test.T, y_test.T)], eval_names = ['training', 'val'] ,eval_metric= 'rmse', verbose = False)



lgb.plot_metric(model_100, 'rmse')

plt.show();





#Prediction

y_pred_train = LGB_100.predict(x_train.T)

y_pred       = LGB_100.predict(x_test.T)



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test  = accuracy_score(y_test.T, y_pred.T) * 100



accuracies_train['LightGBM'] = acc_train

accuracies_test['LightGBM'] = acc_test



print("LightGBM Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM Test Accuracy Score : {:.2f}%".format(acc_test))
params = {'learning_rate':0.5,

          'n_estimators': 10,

          'metric': ['logloss']}



LGB_10 = lgb.LGBMClassifier(**params)



model_10 = LGB_10.fit(x_train.T, y_train.T, eval_set = [(x_train.T, y_train.T), (x_test.T, y_test.T)], eval_names = ['training', 'val'] ,eval_metric= 'rmse', verbose = False)



lgb.plot_metric(model_10, 'rmse')

plt.show();



#Prediction

y_pred = LGB_10.predict(x_test.T)



y_pred_train = LGB_10.predict(x_train.T)



from sklearn.metrics import accuracy_score



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test = accuracy_score(y_test.T, y_pred.T) * 100



accuracies_train['LightGBM'] = acc_train

accuracies_test['LightGBM'] = acc_test



print("LightGBM Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM Test Accuracy Score : {:.2f}%".format(acc_test))
# Plot importance

lgb.plot_importance(model_10)

plt.show();



import shap #for SHAP values



explainer = shap.TreeExplainer(model_10)

shap_values = explainer.shap_values(x_test.T)



# Display SHAP in bar

shap.summary_plot(shap_values, x_test.T, plot_type="bar")



# Display SHAP

shap.summary_plot(shap_values[1], x_test.T)
from pdpbox import pdp, info_plots #for partial plots



base_features = df.columns.values.tolist()

base_features.remove('target')



feat_name = 'oldpeak'

pdp_dist = pdp.pdp_isolate(model=model_10, dataset=x_test.T, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()



feat_name = 'trestbps'

pdp_dist = pdp.pdp_isolate(model=model_10, dataset=x_test.T, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()

import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - An object to be used as a cross-validation generator.

          - An iterable yielding train/test splits.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : integer, optional

        Number of jobs to run in parallel (default 1).

    """

    plt.figure(figsize=(10,6))  # Resize

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold,StratifiedKFold



lgbr = LGBMClassifier(learning_rate = 0.2)

cv = KFold(n_splits=5, random_state=2, shuffle=True)

estimator = lgbr



plot_learning_curve(estimator, "lgbclassifier", x_train.T, y_train.T, cv=cv, train_sizes=np.linspace(0.2, 1.0, 5))    





lgbr = LGBMClassifier(learning_rate = 0.1)

estimator = lgbr

plot_learning_curve(estimator, "lgbclassifier", x_train.T, y_train.T, cv=cv, train_sizes=np.linspace(0.2, 1.0, 5))   

# applying lightGBM



# We modify the label which label 1 => 0.7 ~ 1, label 0 => 0 ~ 0.3 

y_train_soft = y_train.copy()



for i in range(len(y_train)):

    if y_train_soft[i] == 1:

         y_train_soft[i] = random.uniform(0.7, 1)

    else:

         y_train_soft[i] = random.uniform(0, 0.3)



            

lgb_train      = lgb.Dataset(x_train.T, y_train.T)

lgb_train_soft = lgb.Dataset(x_train.T, y_train_soft.T)





params = {'learning_rate': 0.01,

          'n_estimator' : 100}





def loglikelihood(preds, train_data):

    labels = train_data.get_label()

    preds = sigmoid(preds)

    grad = preds - labels

    hess = preds * (1. - preds)

    return grad, hess



gbm = lgb.train(params,

                lgb_train,

                fobj=loglikelihood)



gbm_soft = lgb.train(params,

                lgb_train_soft,

                fobj=loglikelihood)





#Prediction



y_pred      = gbm.predict(x_test.T)

y_pred_soft = gbm_soft.predict(x_test.T)



# Transfer into final prediction

for i in range(len(y_pred)):

    if sigmoid(y_pred[i]) > 0.5:

        y_pred[i] = 1

    else:

        y_pred[i] = 0

        

for i in range(len(y_pred_soft)):

    if sigmoid(y_pred_soft[i]) > 0.5:

        y_pred_soft[i] = 1

    else:

        y_pred_soft[i] = 0



acc            = accuracy_score(y_test, y_pred) * 100

acc_soft_label = accuracy_score(y_test, y_pred_soft) * 100



print("LightGBM                 Test Accuracy Score : {:.2f}%".format(acc))

print("LightGBM with Soft label Test Accuracy Score : {:.2f}%".format(acc_soft_label))
from sklearn.model_selection import GridSearchCV



param_grid = {'learning_rate': list(np.arange(0.01, 0.2, 0.01)),

              'n_estimators':list(range(1,20))}





LGB = lgb.LGBMClassifier()





from sklearn.metrics import fbeta_score, make_scorer



model_GS = GridSearchCV(LGB, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)



model_GS.fit(x_train.T, y_train.T, eval_set = [(x_test.T, y_test.T)], eval_names = ['val'] ,eval_metric= 'rmse', verbose = False)

print('Best parameters found by grid search are:', model_GS.best_params_)

# Prediction

y_pred_train = model_GS.predict(x_train.T)

y_pred = model_GS.predict(x_test.T)



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test = accuracy_score(y_test.T, y_pred.T) * 100



print("LightGBM_GS Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM_GS Test Accuracy Score : {:.2f}%".format(acc_test))
extra_1 = df_ori['chol'] * df_ori['trestbps']

extra_1.name = 'extra_1'



extra_2 = df_ori['thalach'] * df_ori['trestbps']

extra_2.name = 'extra_2'



extra_3 = df_ori['age'] * df_ori['trestbps']

extra_3.name = 'extra_3'



extra_4 = df_ori['age'] * df_ori['thalach']

extra_4.name = 'extra_4'



extra_5 = df_ori['age'] * df_ori['oldpeak']

extra_5.name = 'extra_5'
# Merge with the original dataset

frames = [df_ori, extra_1, extra_2, extra_3, extra_4, extra_5]

df_new = pd.concat(frames, axis = 1)

df_new.head()
y = df_new.target.values

x_data = df_new.drop(['target'], axis = 1)
# Normalize and split

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)



#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test  = x_test.T

y_test  = y_test.T
params = {'learning_rate':0.05,

          'n_estimators': 40,

          'metric': ['logloss']}



LGB_40 = lgb.LGBMClassifier(**params)



model_40 = LGB_40.fit(x_train.T, y_train.T, eval_set = [(x_train.T, y_train.T), (x_test.T, y_test.T)], eval_names = ['training', 'val'] ,eval_metric= 'rmse', verbose = False)



lgb.plot_metric(model_40, 'rmse')

plt.show();



#Prediction

y_pred = LGB_40.predict(x_test.T)



y_pred_train = LGB_40.predict(x_train.T)



from sklearn.metrics import accuracy_score



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test = accuracy_score(y_test.T, y_pred.T) * 100



accuracies_train['LightGBM'] = acc_train

accuracies_test['LightGBM'] = acc_test



print("LightGBM Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM Test Accuracy Score : {:.2f}%".format(acc_test))
lgb.plot_importance(model_40)

plt.show();
from sklearn.feature_selection import VarianceThreshold





y = df_ori.target.values

x_data = df_ori.drop(['target'], axis = 1)



print('Before threshold, dataset')

print(x_data.shape)



#remove features with variance below the threshold

min_feature_variance = 0.1

feature_selector = VarianceThreshold(threshold = min_feature_variance * (1 - min_feature_variance))

x_data = feature_selector.fit_transform(x_data)



print('After threshold, dataset:')

print(x_data.shape)



# Normalize and split

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)



#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test  = x_test.T

y_test  = y_test.T
params = {'learning_rate':0.05,

          'n_estimators': 40,

          'metric': ['logloss']}



LGB_40 = lgb.LGBMClassifier(**params)



model_40 = LGB_40.fit(x_train.T, y_train.T, eval_set = [(x_train.T, y_train.T), (x_test.T, y_test.T)], eval_names = ['training', 'val'] ,eval_metric= 'rmse', verbose = False)



lgb.plot_metric(model_40, 'rmse')

plt.show();



#Prediction

y_pred = LGB_40.predict(x_test.T)



y_pred_train = LGB_40.predict(x_train.T)



from sklearn.metrics import accuracy_score



acc_train = accuracy_score(y_train.T, y_pred_train.T) * 100

acc_test = accuracy_score(y_test.T, y_pred.T) * 100



accuracies_train['LightGBM'] = acc_train

accuracies_test['LightGBM'] = acc_test



print("LightGBM Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM Test Accuracy Score : {:.2f}%".format(acc_test))


from sklearn.feature_selection import SelectKBest, chi2



y = df_ori.target.values

x_data = df_ori.drop(['target'], axis = 1)



# Normalize and split

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)





print('Before threshold, dataset:')

print(x_train.shape)



# apply ch2

ch2 = SelectKBest(chi2)

x_train = ch2.fit_transform(x_train, y_train)

x_test = ch2.transform(x_test)



print('After threshold, dataset:')

print(x_train.shape)



params = {'learning_rate':0.05,

          'n_estimators': 40,

          'metric': ['logloss']}



LGB_40 = lgb.LGBMClassifier(**params)



model_40 = LGB_40.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], eval_names = ['training', 'val'] ,eval_metric= 'rmse', verbose = False)



lgb.plot_metric(model_40, 'rmse')

plt.show();



#Prediction

y_pred = LGB_40.predict(x_test)



y_pred_train = LGB_40.predict(x_train)



from sklearn.metrics import accuracy_score



acc_train = accuracy_score(y_train, y_pred_train) * 100

acc_test = accuracy_score(y_test, y_pred) * 100



accuracies_train['LightGBM'] = acc_train

accuracies_test['LightGBM'] = acc_test



print("LightGBM Train Accuracy Score : {:.2f}%".format(acc_train))

print("LightGBM Test Accuracy Score : {:.2f}%".format(acc_test))