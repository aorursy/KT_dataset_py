import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report, recall_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

sns.set()
df_test = pd.read_csv("../input/carinsurance/carInsurance_test.csv")

df_train = pd.read_csv("../input/carinsurance/carInsurance_train.csv")
print("The training data has {0} samples and {1} features.".format(df_train.shape[0], df_train.shape[1]-1))

print("The testing data has {0} samples and {1} features.".format(df_test.shape[0], df_test.shape[1]-1))
df_train.isnull().sum()
df_test.isnull().sum()
def hist_matrix(data):

    numeric_cols = [col for col in data if data[col].dtype!= "O"]

    fig, ax = plt.subplots(nrows = 4, ncols = 3,figsize = (16,10))

    fig.subplots_adjust(hspace = 0.5)

    x=0

    y=0

    for i in numeric_cols:

        ax[y,x].hist(data[i])

        ax[y,x].set_title("{}".format(i))

        x+=1

        if x == 3:

            x-=3

            y+=1

    return
hist_matrix(df_train)
def preprocessing(data):

    data = data.drop('Id', axis = 1)

    data['Education'] = data['Education'].fillna(data['Education'].mode()[0])

    data['Job'] = data['Job'].fillna(data['Job'].mode()[0])

    for i in ['CallStart', 'CallEnd']:

        data[i] = pd.to_datetime(data[i])

    data['CallDur'] = ((data['CallEnd']-data['CallStart']).dt.seconds)/60

    data['CallHour'] = data['CallStart'].dt.hour

    data = data.drop(['CallStart', 'CallEnd'], axis = 1)

    for i in ['Balance', 'NoOfContacts', 'PrevAttempts', 'DaysPassed']:

        val = data[i].quantile(.99).astype(int)

        data.loc[data[i]>val, i] = val

    data.loc[data['DaysPassed']<0, 'DaysPassed'] = 0

    data['Communication'] = data['Communication'].fillna('Missing')

    data['Outcome'] = data['Outcome'].fillna('Mising')

    return data
df_train = preprocessing(df_train)
df_train.isnull().sum()
hist_matrix(df_train)
y = df_train['CarInsurance'].copy()

x_cols = [col for col in df_train.columns if col != "CarInsurance"]

x = df_train[x_cols].copy()
numeric_cols = [col for col in x if x[col].dtype != "O"]

numeric_transformer = Pipeline(steps = [(

                        'scaler', MinMaxScaler())])



categorical_cols = [col for col in x if col not in numeric_cols]

categorical_transformer = Pipeline(steps=[(

                            'ohe', OneHotEncoder(drop = 'first'))])



preprocessor = ColumnTransformer(transformers = [

                ('num', numeric_transformer, numeric_cols),

                ('cat', categorical_transformer, categorical_cols)])
models = ['Random Forest: ', 'Logistic Regression: ', 'XGBoost: ']

suffixes = ['_rf', '_logit', '_xgb']

names = ['fpr', 'tpr', 'thresholds', 'auc']

j = 0

for i in [RandomForestClassifier(n_estimators = 100, random_state=0), LogisticRegression(solver = 'lbfgs'), XGBClassifier(random_state=0)]:

    clf = Pipeline(steps = [

            ('preprocessor', preprocessor),

            ('classifier', i)])

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)

    print(models[j])

    print("Model Accuracy: {}".format(round(accuracy_score(y_test, preds),4)*100))

    print(classification_report(y_test, preds))

    print("*****************************************************")   

    exec("{0}, {1}, {2} = roc_curve(y_test, clf.predict_proba(x_test)[:,1])".format(names[0]+suffixes[j], names[1]+suffixes[j], names[2]+suffixes[j]))

    exec("{} = roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])".format(names[3]+suffixes[j]))

    j+=1
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18,6))

ax[0].plot(fpr_rf,tpr_rf, label = "Random Forest")

ax[0].plot([0,1], [0,1], label = 'Base Rate')

ax[0].set_ylabel("True Positive Rate")

ax[0].set_title("Random Forest")

ax[0].text(0.3, 0.7, "AUC = {}".format(round(auc_rf, 2)))

ax[1].plot(fpr_logit,tpr_logit, label = "Random Forest")

ax[1].plot([0,1], [0,1], label = 'Base Rate')

ax[1].set_xlabel("False Positive Rate")

ax[1].set_title("Logistic Regression")

ax[1].text(0.3, 0.7, "AUC = {}".format(round(auc_logit, 2)))

ax[2].plot(fpr_xgb,tpr_xgb, label = "Random Forest")

ax[2].plot([0,1], [0,1], label = 'Base Rate')

ax[2].set_title("XGBoost")

ax[2].text(0.3, 0.7, "AUC = {}".format(round(auc_xgb, 2)))

fig.suptitle("ROC Graph")
clf = Pipeline(steps = [

        ('preprocessor', preprocessor),

        ('classifier', RandomForestClassifier(random_state=0))])



param_grid = {'classifier__n_estimators' : [10, 50, 75, 100, 150, 200, 250, 300],

             'classifier__criterion' : ['gini', 'entropy']}



search = GridSearchCV(clf, param_grid, n_jobs = -1, cv = 7)

search.fit(x_train, y_train)

print(search.best_params_)
clf = Pipeline(steps = [

        ('preprocessor', preprocessor),

        ('classifier', RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state=0))])

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

print("Tuned Random Forest: ")

print("Model Accuracy: {}".format(round(accuracy_score(y_test, preds),4)*100))

print(classification_report(y_test, preds))
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt = ".0f")

plt.yticks([1.5,0.5], ['Did not Buy', 'Did Buy'])

plt.xticks([1.5,0.5], ['Did Buy', 'Did not Buy'])

plt.ylabel("Actual")

plt.xlabel("Predicted")

plt.title("Confusion Matrix")
names = {"feature" : numeric_cols + list(clf['preprocessor'].transformers_[1][1]['ohe'].get_feature_names(categorical_cols))}

imp = {'importances' : list(clf.steps[1][1].feature_importances_)}

feature_importances = {**names, **imp}

feature_importances_df = pd.DataFrame(feature_importances) 
feature_importances_df = feature_importances_df.sort_values(by = 'importances', ascending = True)

feature_importances_df
plt.figure(figsize = (16,10))

plt.title("Feature Importances from Random Forest Classifier Model")

plt.barh(feature_importances_df['feature'], feature_importances_df['importances'])