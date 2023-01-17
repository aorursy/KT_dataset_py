import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from collections import Counter
df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")
df_train.head()
df_train.describe().T
plt.figure(figsize=(15,5))
sns.countplot(df_train.Survived);
plt.figure(figsize=(15,5))
sns.countplot(df_train.Sex);
plt.figure(figsize=(15,5))
sns.countplot(df_train.Pclass);
plt.figure(figsize=(15,5))
sns.countplot(df_train.Embarked);
plt.figure(figsize=(15,5))
sns.countplot(df_train.SibSp);
plt.figure(figsize=(15,5))
sns.countplot(df_train.Parch);
plt.figure(figsize=(15,5))
sns.distplot(df_train.Age, hist=True, kde=True, color='r');
plt.show()
plt.figure(figsize=(15,5))
sns.barplot(x='Sex', y='Survived', data=df_train)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(df_train.Sex, hue=df_train.Survived, palette='pastel');
plt.figure(figsize=(15,5))
sns.countplot(df_train.Pclass, hue=df_train.Survived, palette='pastel');
df = pd.concat([df_train, df_test])
df[df['Embarked'].isnull()]
sns.catplot(x="Embarked", y="Fare", kind="box", data=df);
df['Embarked'] = df["Embarked"].fillna("C")
df[df['Fare'].isnull()]
df['Fare'] = df["Fare"].fillna(df.Fare.mean())
plt.figure(figsize=(15,5))
sns.distplot(df.Age);
plt.plot()
df['Age'] = df["Age"].fillna(df.Age.mean())
df.head()
df = df.drop(['Name', "PassengerId", "Ticket", "Cabin"], axis=1)
df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df["Sex"])
df['Embarked'] = le.fit_transform(df['Embarked'])
df.head()
train = df.iloc[0: (df.shape[0] - df_test.shape[0])]
test = df.iloc[df_train.shape[0]:]
X = train.drop(['Survived'], axis=1)
y = train.Survived
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_params = {'n_estimators': [400,500,600,700],
            'max_features': [5,6,7,8,9,10],
            'min_samples_split':[5,6,7,8,9,10]}
from sklearn.model_selection import GridSearchCV
rf_cv_model = GridSearchCV(rf, rf_params, cv=21, n_jobs=-1, verbose=1).fit(X, y)
best_params = rf_cv_model.best_params_
print(best_params)
rf = RandomForestClassifier(
    max_features=best_params['max_features'], 
    min_samples_split=best_params['min_samples_split'], 
    n_estimators=best_params['n_estimators']
).fit(X, y)
y_pred_rf = rf.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred_rf)
rf.feature_importances_
feature_imp = pd.Series(rf.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.show()
from sklearn.model_selection import cross_val_score
cross_val_score(rf, X, y, cv=7).mean()
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support
print('sensitivity and specificity:', sensitivity_specificity_support(y, y_pred_rf, average='micro', labels=pd.unique(df_train.Survived)))
print(classification_report_imbalanced(y, y_pred_rf))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title('Confusion Matrix')
plt.savefig('con_mat')
plt.show()
from sklearn.metrics import roc_auc_score, roc_curve
rf_roc_auc = roc_auc_score(y, rf.predict(X))
fpr , tpr, thresholds = roc_curve(y, rf.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % rf_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()
test = test.drop(['Survived'], axis=1)
predictions = rf.predict(test)
submission = df_test.PassengerId.copy().to_frame()
predictions = [int(i) for i in predictions]
submission['Survived'] = predictions
submission.to_csv("submission.csv", index = False)
submission
