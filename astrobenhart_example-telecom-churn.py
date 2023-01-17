import pandas as pd

import numpy as np

import seaborn as sn

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv("/kaggle/input/telecom-churn/telecom_churn.csv")
df.head()
label_ratio = df.Churn.value_counts()

print('There are {} churn and {} current customers, or {}% have churned.'.format(label_ratio[1], label_ratio[0], np.round((label_ratio[1]/label_ratio[0])*100, 1)))
df.describe()
df.info()
df_hue = df.copy()

df_hue["Churn"] = np.where(df_hue["Churn"] == 0, "S", "C")

sn.pairplot(df_hue.drop(['DataPlan', 'ContractRenewal'], axis=1), hue="Churn", palette="husl")
def calc_vif(X):

    vif = pd.DataFrame()

    vif["variables"] = X.columns

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(df)
df['average_call_dura'] = df.apply(lambda x: x['DayMins'] / x['DayCalls'] if x['DayCalls'] > 0 else 0,axis=1)

# df['total_monthly_charge'] = df.apply(lambda x: x['MonthlyCharge'] + x['OverageFee'],axis=1)

df2 = df.drop(['MonthlyCharge', 'DayCalls', 'DataUsage', 'RoamMins', 'DayMins'],axis=1)

calc_vif(df2)
final_model_cols = ['AccountWeeks', 'ContractRenewal', 'DataPlan', 'CustServCalls', 'OverageFee', 'average_call_dura']
plt.figure(figsize=(12,9))

corrMatrix = df.drop('average_call_dura', axis=1).corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
X0 = df[df.DataPlan==0].MonthlyCharge.values.reshape(-1,1)

y0 = df[df.DataPlan==0].DayMins.values.reshape(-1,1)

X1 = df[df.DataPlan==1].MonthlyCharge.values.reshape(-1,1)

y1 = df[df.DataPlan==1].DayMins.values.reshape(-1,1)



X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.33, random_state=42)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=42)



regr0 = LinearRegression()

regr1 = LinearRegression()

regr0.fit(X0_train, y0_train)

regr1.fit(X1_train, y1_train)

y0_pred = regr0.predict(X0_test)

y1_pred = regr1.predict(X1_test)

print('Coefficients: \n No data plan y = {}x + {} \n With data plan y = {}x + {}'.format(regr0.coef_[0][0], regr0.intercept_[0], regr1.coef_[0][0], regr1.intercept_[0]))

print('Mean squared error for no data plan: {}; Mean squared error for data plan: {}'.format(mean_squared_error(y0_test, y0_pred), mean_squared_error(y1_test, y1_pred)))

print('Coefficient of determination for no data plan: {}; Coefficient of determination for data plan: {}'.format(r2_score(y0_test, y0_pred), r2_score(y1_test, y1_pred)))





fig, ax = plt.subplots()

ax.scatter(X0_test, y0_test, color='red', alpha=0.6, label='No data plan train')

ax.plot(X0_test, y0_pred, color='orange', linewidth=2, label='No data plan pred')

ax.scatter(X1_test, y1_test, color='green', alpha=0.6, label='Data plan train')

ax.plot(X1_test, y1_pred, color='blue', linewidth=2, label='Data plan pred')

ax.set_xlabel('MonthlyCharge')

ax.set_ylabel('DayMins')

plt.legend()

plt.show()
X = df.drop('Churn', axis=1)

y = df.Churn



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



X_filt = X[final_model_cols]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_filt_train, X_filt_test, y_train, y_test = train_test_split(X_filt, y, test_size=0.33, random_state=42)



feature_names = list(X_train.columns.values)
from sklearn.dummy import DummyClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve
maj_class = DummyClassifier(strategy="most_frequent")
maj_class.fit(X_train, y_train)

preds = maj_class.predict(X_test)

maj_class.score(X_test,y_test)

print('Score: {}'.format(maj_class.score(X_test, y_test)))

print('Cross Val score: {}'.format(np.round(cross_val_score(maj_class, X_test, y_test).mean(),2)))

print('ROC AUC Score: {}'.format(roc_auc_score(y_test, preds)))

fpr, tpr, thresholds = roc_curve(y_test, preds)

print('AUC Score: {}'.format(auc(fpr, tpr)))
confusion_matrix(y_test, preds)
from sklearn.ensemble import RandomForestClassifier
RF_class = RandomForestClassifier(random_state=42, class_weight='balanced').fit(X_train, y_train)

preds = RF_class.predict(X_test)

print('Score: {}'.format(RF_class.score(X_test, y_test)))

print('Cross Val score: {}'.format(np.round(cross_val_score(RF_class, X_test, y_test).mean(),2)))

print('ROC AUC Score: {}'.format(roc_auc_score(y_test, preds)))

fpr, tpr, thresholds = roc_curve(y_test, preds)

print('AUC Score: {}'.format(auc(fpr, tpr)))

print("\nFeature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), RF_class.feature_importances_), feature_names), 

             reverse=True)))
confusion_matrix(y_test, preds)
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score
SVC_class = make_pipeline(StandardScaler(), SVC(class_weight='balanced')).fit(X_train, y_train)

preds = SVC_class.predict(X_test)

print('Score: {}'.format(SVC_class.score(X_test, y_test)))

print('Cross Val score: {}'.format(np.round(cross_val_score(SVC_class, X_test, y_test).mean(),2)))

print('ROC AUC Score: {}'.format(roc_auc_score(y_test, preds)))

fpr, tpr, thresholds = roc_curve(y_test, preds)

print('AUC Score: {}'.format(auc(fpr, tpr)))
confusion_matrix(y_test, preds)
from sklearn.ensemble import ExtraTreesClassifier
ET_class = ExtraTreesClassifier(random_state=42, class_weight='balanced').fit(X_train, y_train)

preds = ET_class.predict(X_test)

print('Score: {}'.format(ET_class.score(X_test, y_test)))

print('Cross Val score: {}'.format(np.round(cross_val_score(ET_class, X_test, y_test).mean(),2)))

print('ROC AUC Score: {}'.format(roc_auc_score(y_test, preds)))

fpr, tpr, thresholds = roc_curve(y_test, preds)

print('AUC Score: {}'.format(auc(fpr, tpr)))

print("\nFeature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), ET_class.feature_importances_), feature_names), 

             reverse=True)))
confusion_matrix(y_test, preds)
from xgboost import XGBClassifier
XG_class = XGBClassifier(eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc').fit(X_train, y_train)

preds = XG_class.predict(X_test)

print('Score: {}'.format(XG_class.score(X_test, y_test)))

print('Cross Val score: {}'.format(np.round(cross_val_score(XG_class, X_test, y_test).mean(),2)))

print('ROC AUC Score: {}'.format(roc_auc_score(y_test, preds)))

fpr, tpr, thresholds = roc_curve(y_test, preds)

print('AUC Score: {}'.format(auc(fpr, tpr)))

print("\nFeature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), XG_class.feature_importances_), feature_names), 

             reverse=True)))
import scipy.stats as ss

        

def compare_filt_scores(classifier):

    clf = classifier.fit(X_train, y_train)

    print('no filter: {}'.format(np.round(clf.score(X_test, y_test), 3)))

    print('Cross Val score: {}'.format(np.round(cross_val_score(clf, X_test, y_test).mean(),3)))

    feature_names = list(X_train.columns.values)

    try:

        print("Feature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True))+"\n")

    except AttributeError as a:

        pass

    

    clf_filt = classifier.fit(X_filt_train, y_train)

    print('with filter: {}'.format(np.round(clf_filt.score(X_filt_test, y_test), 3)))

    print('Cross Val score: {}'.format(np.round(cross_val_score(clf_filt, X_filt_test, y_test).mean(),3)))

    feature_names = list(X_filt_train.columns.values)

    try:

        print("Feature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True))+"\n")

    except AttributeError as a:

        pass

    
print('majority class: 0.85')

print('')

print('RF classifier')

compare_filt_scores(RandomForestClassifier(random_state=42, class_weight='balanced'))

print('')

print('SVC classifier')

compare_filt_scores(make_pipeline(StandardScaler(), SVC(class_weight='balanced')))

print('')

print('ET classifier')

compare_filt_scores(ExtraTreesClassifier(random_state=42, class_weight='balanced'))

print('')

print('XG classifier')

compare_filt_scores(XGBClassifier())

print('')
df.Churn.value_counts()
56.22 * 258
threshold = 0.2



predicted_proba = XG_class.predict_proba(X_test)

predicted = (predicted_proba [:,1] >= threshold).astype('int')



accuracy = accuracy_score(y_test, predicted)

print(accuracy)

cm = confusion_matrix(y_test, predicted)

print(cm)



print("correctly classified {} churners and {} non churners.".format(cm[0][0], cm[1][1]))

print("{} classified as churners when not, represents {} in potential churn driven loss".format(cm[1][0], cm[1][0]*56.22))

print("{} classified as not churners when they were, represents {} missed retention opppertunity loss".format(cm[0][1], cm[0][1]*56.22))
predicted_proba = XG_class.predict_proba(X_test)

pos = [x[1] for x in predicted_proba]

ns_probs = [0 for _ in range(len(y_test))]



ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(y_test, pos)



lr_auc = roc_auc_score(y_test, pos)

print('XGBoost: ROC AUC=%.3f' % (lr_auc))



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='XGBoost')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
from sklearn.metrics import precision_recall_curve, f1_score
yhat = XG_class.predict(X_test)



lr_precision, lr_recall, _ = precision_recall_curve(y_test, pos)

lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)



no_skill = len(y_test[y_test==1]) / len(y_test)



print('XGBoost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))



plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
def to_labels(pos_probs, threshold):

	return (pos_probs >= threshold).astype('int')
predicted_proba = XG_class.predict_proba(X_test)

probs = predicted_proba[:, 1]

thresholds = np.arange(0, 1, 0.001)

scores = [f1_score(y_test, to_labels(probs, t)) for t in thresholds]

ix = np.argmax(scores)

print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

plt.plot(thresholds, scores)

plt.vlines(thresholds[ix], 0, scores[ix]+0.05*scores[ix])

plt.hlines(scores[ix], 0, 1)

plt.xlabel('Thresholds')

plt.ylabel('F1 score')
threshold = 0.614



predicted_proba = XG_class.predict_proba(X_test)

predicted = (predicted_proba [:,1] >= threshold).astype('int')



accuracy = accuracy_score(y_test, predicted)

print(accuracy)

cm = confusion_matrix(y_test, predicted)

print(cm)



print('potential to save: {}; ${}'.format(len(y_test[y_test==1]), len(y_test[y_test==1])*56.22))

print("correctly classified TN: {} churners and TP: {} non churners.".format(cm[0][0], cm[1][1]))

print("FP: {} classified as churners when not, represents {} in potential churn driven loss".format(cm[1][0], cm[1][0]*56.22))

print("FN: {} classified as not churners when they were, represents {} missed retention opppertunity loss".format(cm[0][1], cm[0][1]*56.22))

print('If we assume you always save 50% of TP, drive 20% of FP customers away, and lose all FN we can see the new loss')

saved = (int(0.5*cm[1][1]) - int(0.2*cm[1][0]) - cm[0][1])

saved_d = (int(0.5*cm[1][1]) - int(0.2*cm[1][0]) - cm[0][1])*56.22

print('Saved: {}; ${}; {}%'.format(saved, saved_d, (saved/len(y_test[y_test==1]))*100))