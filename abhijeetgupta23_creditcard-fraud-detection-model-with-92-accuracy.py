import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix,f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import validation_curve
credit_card_dataset = pd.read_csv("../input/creditcardfraud/creditcard.csv")
def understand_variables(dataset):

    print("Type = " +str(type(dataset))+"\n")

    print("Shape = "+str(dataset.shape)+"\n")

    print("Head : \n\n"+str(dataset.head())+"\n\n")

    print("Columns:\n"+str(dataset.columns)+"\n\n")

    print("No.of unique values :\n\n"+str(dataset.nunique(axis=0))+"\n\n")

    print("Description :\n\n"+str(dataset.describe())+"\n\n")

    

    #print(dataset.describe(exclude=[np.number]))

    #Since no categorical variables, no need to have the above line

    

    print("Null count :\n\n"+str(dataset.isnull().sum()))

    

understand_variables(credit_card_dataset)
credit_card_dataset['Time'] = ((credit_card_dataset['Time']/3600)%24).sort_values(ascending=False)
def variable_distribution_analysis(dataset):



        

    numerical_features=[feature for feature in dataset.columns if dataset[feature].dtype!='O']



    for feature in numerical_features:

        plt.figure(figsize=(10,4))

        try:

            sns.distplot(dataset[feature])

            plt.show()

        except:

            continue





    #sns.pairplot(dataset,kind="reg")

    #plt.show()

    

variable_distribution_analysis(credit_card_dataset)
for col in list(credit_card_dataset.columns):

    if col!='Class':

        plt.figure(figsize=(10,4))

        sns.boxplot(data=credit_card_dataset,x='Class',y=col)

        plt.show()
credit_card_corr = credit_card_dataset.corr()

plt.figure(figsize=(25,25))

sns.heatmap(data=credit_card_corr, annot=True,fmt='.1f')
X = credit_card_dataset.drop(["Class"],axis=1)

y = credit_card_dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1,stratify=y)

forest_model = RandomForestClassifier(n_estimators=20,random_state=1)

forest_model.fit(X_train, y_train)

Y_pred = pd.Series(forest_model.predict(X_test))



print("Training Accuracy :", forest_model.score(X_train, y_train))

print("Testing Accuracy :", forest_model.score(X_test, Y_pred))



conf = confusion_matrix(y_test, Y_pred)

print("\nConfusion matrix\n"+str(conf))

print("\nF1 score = "+str(round(f1_score(y_test, Y_pred)*100,2))+" %")

print("\nClassification report\n\n"+str(classification_report(y_test, Y_pred)))
features = pd.Series(forest_model.feature_importances_)

features.index = X_train.columns

features = features.sort_values(ascending=False)

print("Feature Importance in Random Forest:\n"+ str(features))
from sklearn import datasets

import xgboost as xgb

xgb_classifier = xgb.XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),random_state=1,base_score=0.3)
X = credit_card_dataset.drop(["Class"],axis=1)

y = credit_card_dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.245,random_state=1,stratify=y)



xgb_model = xgb_classifier.fit(X_train, y_train)

xgb_y_pred = xgb_model.predict(X_test)

#best_preds = np.asarray([np.argmax(line) for line in xgb_y_pred])

conf = confusion_matrix(y_test, xgb_y_pred)

print("\nConfusion matrix\n"+str(conf))

print("\nF1 score = "+str(round(f1_score(y_test, xgb_y_pred)*100,2))+" %")

print(classification_report(y_test, xgb_y_pred))
f = 'gain'

x = xgb_model.get_booster().get_score(importance_type= f)



print("Feature importance for xgboost model : \n")

{k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc
## Get predcition probabilities (range of 0 to 1) instead of binary values (0/1)

xgb_y_pred_prob = xgb_model.predict_proba(X_test)
y_pred_prob_df = pd.DataFrame(xgb_y_pred_prob)[[1]]

print(y_pred_prob_df.sort_values(by=[1],ascending=False).head(118)) # 117 predicted as fraud, and last probability in the list (index = 44160) is for a class predicted as non-fraud



#This verifies the fact that probablity threshold is 0.5. We can look at how recall changes with change in probablity threshold



xgb_y_pred_prob = pd.DataFrame(xgb_y_pred_prob)
import sklearn

auprc = sklearn.metrics.average_precision_score(y_test, xgb_y_pred)

print(auprc)
xgb_precision, xgb_recall,threshold = precision_recall_curve(y_test, y_pred_prob_df)

xgb_f1, xgb_auc = f1_score(y_test, xgb_y_pred), auc(xgb_recall, xgb_precision)
print(xgb_f1)

print(xgb_auc)
#threshold = np.append(threshold,[1])

precision_recall_threshold_df = pd.DataFrame([xgb_precision, xgb_recall,threshold]).transpose()

precision_recall_threshold_df.columns = ['Precision','Recall','Threshold']

precision_recall_threshold_df['F1-Score'] = 2*precision_recall_threshold_df['Precision']*precision_recall_threshold_df['Recall']/(precision_recall_threshold_df['Precision']+precision_recall_threshold_df['Recall'])

print(precision_recall_threshold_df)
plt.figure(figsize=(20,4))

sns.lineplot(x="Recall", y="Precision", data=precision_recall_threshold_df)

plt.show()
plt.figure(figsize=(20,4))

sns.lineplot(x="Threshold", y="F1-Score", data=precision_recall_threshold_df)

plt.show()



## We look at thresholds where Precision is >=0.85 and Recall >= 0.9



print(precision_recall_threshold_df[(precision_recall_threshold_df['Precision']>=0.85) & (precision_recall_threshold_df['Recall']>=0.9)].sort_values(by='Recall',ascending=False))



##Then, after the threshold observations, we select threshold with highest f1-score (since increasing recall will decrease precision drastically)

max_f1_threshold = (precision_recall_threshold_df[precision_recall_threshold_df["F1-Score"]==max(precision_recall_threshold_df["F1-Score"])]["Threshold"]).iloc[0]
## Classify as fraud for probability greater than max_f1_threshold (~0.33)



y_pred_prob_df.columns = ['Prob of 1']

y_pred_prob_df['class'] = [1 if float(prob)>=max_f1_threshold else 0 for prob in y_pred_prob_df['Prob of 1']]
conf = confusion_matrix(y_test, y_pred_prob_df['class'])

print("\nConfusion matrix\n"+str(conf))

print("\nF1 score = "+str(round(f1_score(y_test, y_pred_prob_df['class'])*100,2))+" %")

print(classification_report(y_test, y_pred_prob_df['class']))