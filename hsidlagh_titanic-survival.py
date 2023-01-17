# Main imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/titanic/train.csv')

df.head()
df.info()
df.describe()
df.isnull().sum()
df.Embarked.value_counts()
df.loc[df.Embarked.isnull(), :]
df.pivot_table(columns=['Survived'], index='Embarked', values='Ticket' , aggfunc='count')
df.pivot_table(columns=['Survived'], index='Sex', values='Ticket' , aggfunc='count')
df = df.loc[~df.Embarked.isnull(),:]
df = df.drop('Cabin', axis=1)
df.isnull().sum()
import math

# Find percent of survival based on age range .. 5 means 5-10years, -1 means age was null

df['AgeRange'] = df.Age.apply(lambda x: -1 if np.isnan(x) else math.floor(x/5)*5)

agedf1 = df.groupby(by='AgeRange')['Survived'].mean()

sns.barplot(agedf1.index, agedf1.values)
#Capture current number of records

df.shape
df = df.loc[~df.Age.isnull(),:]
df.isnull().sum()
df.shape
df = df.drop(['Name', 'Ticket'], axis=1)
df.head()
df.Sex.value_counts()
df['male'] = df['Sex'].apply(lambda x: 1 if (x == 'male') else 0)

df = df.drop('Sex', axis=1)
df_dum1 = pd.get_dummies(df['Embarked'], prefix='Emb')
df = pd.concat([df, df_dum1], axis=1)
df = df.drop('Embarked', axis=1)
df = df.drop('Emb_S', axis=1)
df = df.drop('AgeRange', axis=1)
df.head()
y_train = df.pop('Survived')
X_train = df
# We dont need Passenger ID in the model

X_train = X_train.drop('PassengerId', axis=1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']] )
sns.heatmap(X_train.corr(), annot=True)
import statsmodels.api as sm



X_train_sm = sm.add_constant(X_train)

modl = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

modl.fit().summary()
X_train = X_train.drop('Parch', axis=1)

X_train_sm = sm.add_constant(X_train)



modl2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

modl2.fit().summary()

X_train = X_train.drop('Fare', axis=1)

X_train_sm = sm.add_constant(X_train)



modl3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

modl3.fit().summary()
X_train = X_train.drop('Emb_Q',  axis=1)

X_train_sm = sm.add_constant(X_train)



modl4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

modl4.fit().summary()
X_train = X_train.drop('Emb_C',  axis=1)

X_train_sm = sm.add_constant(X_train)



modl5 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

modl5.fit().summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
res = modl5.fit()



y_train_pred = res.predict(X_train_sm)

# Convert predicted to a one-d array  

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
# Join the predicted values, training values together

y_train_pred_final = pd.DataFrame({'Survived': y_train.values, "Survived_Prob": y_train_pred})

y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
from sklearn import metrics



cm = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted) 

cm
TP = cm[1,1] 

TN = cm[0,0] 

FP = cm[0,1] 

FN = cm[1,0]

print("Accuracy=", metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted)*100)

print('TP=',TP)

print('TN=',TN)

print('FP=',FP)

print('FN=',FN)

#Sensitivity - how many negatives caught out of actually negative

print('Specificity=', (TN/(TN+FP)))

#Sensitivity - how many positives caught out of actually positive

print('Sensitivity=', (TP/(TP+FN)))

print('Accuracy=', ((TP+TN)/(TP+TN+FP+FN)))

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
#Get ROC data

fpr, tpr, thresholds = metrics.roc_curve( 

    y_train_pred_final.Survived, y_train_pred_final.predicted, drop_intermediate = False )

#Draw using function

draw_roc(y_train_pred_final.Survived, y_train_pred_final.predicted)
# For numbers from 0, 0.1, 0.2, 0.3 ......1 thresolds

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
opt_cutoff = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])

l1 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for n in l1:

    c_mat= metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[n] )

    tot=sum(sum(c_mat))

    accuracy = (c_mat[0,0]+c_mat[1,1])/tot

    

    specificity = c_mat[0,0]/(c_mat[0,0]+c_mat[0,1])

    sensitivity = c_mat[1,1]/(c_mat[1,0]+c_mat[1,1])

    opt_cutoff.loc[n] =[ n ,accuracy,sensitivity,specificity]

print(opt_cutoff)
opt_cutoff.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])

plt.plot([0.395,0.395], [0,1], 'k--')

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.395 else 0)

y_train_pred_final.head()
cm = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted) 

TP = cm[1,1] 

TN = cm[0,0] 

FP = cm[0,1] 

FN = cm[1,0]

print('TP=',TP)

print('TN=',TN)

print('FP=',FP)

print('FN=',FN)

print("Accuracy=", metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)*100)

#Sensitivity - how many negatives caught out of actually negative

print('Specificity=', (TN/(TN+FP)))

#Sensitivity - how many positives caught out of actually positive

print('Sensitivity=', (TP/(TP+FN)))

# Calculate false postive rate - predicting churn when customer does not have churned

print('FPR=', FP/ float(TN+FP))

# Positive predictive value 

print ('Positive predictive value=', TP / float(TP+FP))

# Negative predictive value

print ('Negative predictive value', TN / float(TN+ FN))

from sklearn.metrics import precision_score, recall_score



ps = precision_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)

rs = recall_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)

print('Precision=', ps)

print('Recall=', rs)
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)

plt.plot(thresholds, p[:-1], "b-")

plt.plot(thresholds, r[:-1], "y-")

plt.show()
dft = pd.read_csv('../input/titanic/test.csv')

dft.head()
dft.info()
dft.loc[dft.Embarked.isnull(), :]

dft = dft.loc[:,  ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

dft['male'] = dft['Sex'].apply(lambda x: 1 if (x == 'male') else 0)

dft = dft.drop('Sex', axis=1)

dft_dum1 = pd.get_dummies(dft['Embarked'], prefix='Emb')

dft = pd.concat([dft, dft_dum1], axis=1)

dft = dft.drop('Embarked', axis=1)

dft = dft.drop('Emb_S', axis=1)



# dft = dft.loc[~dft.Age.isnull(),:] ---

maleagegavg = dft.loc[dft.male == 1, ['Age']].mean()

femaleagegavg = dft.loc[dft.male == 0, ['Age']].mean()

print('Male Avg ', maleagegavg)

print('Female Avg ', femaleagegavg)
dft.Age.fillna(30.272362, inplace=True)
dft.info()
dft.loc[dft.Fare.isnull(),:]
dft.Fare = dft.Fare.fillna(0)
dft.info()
X_test = dft

X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']] )

X_test = X_test.drop('Parch', axis=1)

X_test = X_test.drop('Fare', axis=1)

X_test = X_test.drop('Emb_Q',  axis=1)

X_test = X_test.drop('Emb_C',  axis=1)

X_test_passngrId = X_test.loc[:,['PassengerId']]

X_test = X_test.drop('PassengerId', axis=1)

X_test_sm = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_sm)

y_test_pred = y_test_pred.values.reshape(-1)

# Join the predicted values, training values together

y_test_pred_final = pd.DataFrame({"Survived_Prob": y_test_pred})

y_test_pred_final['predicted'] = y_test_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.395 else 0)

y_test_pred_final.head()
y_test_pred_final = pd.concat([y_test_pred_final, X_test_passngrId], axis=1)

y_test_pred_final.head()
answer = y_test_pred_final.loc[:,['PassengerId', 'predicted']]

answer.columns = ['PassengerId','Survived']
answer.set_index('PassengerId')
answer.to_csv('Titanic_Pred.csv', index=False)