import pandas as pd

import numpy as np



# model prep

from sklearn import cluster

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

from sklearn.model_selection import train_test_split

!pip install prince

from prince import PCA

from statsmodels.regression import linear_model



# models

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# model evaluation

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import cross_val_predict



# visualization

import matplotlib.pyplot as plt

import matplotlib.style as style

%matplotlib inline

style.use('seaborn-white')

import seaborn as sns

sns.set_style('white')



!pip install jupyterthemes

from jupyterthemes import jtplot

jtplot.style(theme = 'onedork', grid = False)
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.info()
print('Blanks in TotalCharges')

print(df.TotalCharges.str.isspace().value_counts())



print('\nTenure where TotalCharges is blank')

print(df[df.TotalCharges.str.isspace()][['tenure', 'TotalCharges']])



# clean missing values

# when TotalCharges is blank, the tenure is 0

df.TotalCharges = df.TotalCharges.replace(' ', '0')



# TotalCharges is read initially as a string because of the blank values. This field should be a float.

df.TotalCharges = pd.to_numeric(df.TotalCharges, downcast='float')
# customerID is a unique identifier that provides no predictive value

df = df.drop(['customerID', 'gender'], axis = 1)



categorical_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', \

                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', \

                      'Contract', 'PaperlessBilling', 'PaymentMethod']
df = df.replace({'No phone service': 'No'}, regex=True)

df.head()

df.to_excel(r'kaggle\working\clean_data.xlsx', index = False)
plt.subplots_adjust(left=1, bottom=1, right=3, top=2, wspace=.3, hspace=None)

plt.subplot(1,3,1)

sns.distplot(df['MonthlyCharges'], bins = 7)

plt.subplot(1,3,2)

sns.distplot(df['TotalCharges'], bins = 7)

plt.subplot(1,3,3)

sns.distplot(df['tenure'], bins = 7)
plt.subplots_adjust(left=1, bottom=1, right=3, top=2, wspace=.3, hspace=None)

plt.subplot(1, 3, 1)

sns.boxplot(y=df['MonthlyCharges'])

plt.subplot(1, 3, 2)

sns.boxplot(y=df['TotalCharges'])

plt.subplot(1, 3, 3)

sns.boxplot(y=df['tenure'])

plt.show()
sns.pairplot(df[['MonthlyCharges', 'TotalCharges', 'tenure']], diag_kind = 'kde')
corr = df[['MonthlyCharges', 'TotalCharges', 'tenure']].corr().round(2)

sns.heatmap(corr,fmt='', annot=True, cmap='Blues')
# remove TotalCharges since it's highly correlated with tenure and MonthlyCharges

df = df.drop('TotalCharges', 1)

corr = df[['MonthlyCharges', 'tenure']].corr().round(2)

sns.heatmap(corr,fmt='', annot=True, cmap='Blues')
# encode categorical features

le = LabelEncoder()

X = df.drop('Churn', 1).apply(le.fit_transform)

y = df.Churn



# prepare target variable 'Churn' for decision tree model

y.replace('Yes', 1, inplace = True)

y.replace('No', 0, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = RandomForestClassifier(class_weight='balanced', 

                           n_estimators=100).fit(X_train,y_train)



feature_imp = pd.DataFrame(clf.feature_importances_, index=X.columns)

feature_imp = feature_imp[0].sort_values(ascending=False)



sns.barplot(x=feature_imp[:20], y=feature_imp[:20].index, color = '#3274A1')

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Important Features - Random Forest")

plt.legend()

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=2)

plt.subplots_adjust(left=1, bottom=1, right=3, top=2, wspace=.5, hspace=.5)

df.Contract.value_counts().plot(kind = 'barh', ax = axes[0,0], title = 'Contract')

df.OnlineSecurity.value_counts().plot(kind = 'barh', ax = axes[0,1], title = 'OnlineSecurity')

df.PaymentMethod.value_counts().plot(kind = 'barh', ax = axes[1,0], title = 'PaymentMethod')

df.TechSupport.value_counts().plot(kind = 'barh', ax = axes[1,1], title = 'TechSupport')
print(round(100. * df.Churn.value_counts() / len(df.Churn)))
def cluster_features(n):

    agglo = cluster.FeatureAgglomeration(n_clusters = n)

    agglo.fit(X)

    for i, label in enumerate(set(agglo.labels_)):

        features_with_label = [j for j, lab in enumerate(agglo.labels_) if lab == label]

        clustered_features = []

        for feature in features_with_label:

            clustered_features.append(X.columns[feature])

        print('Cluster {}: {}'.format(i + 1, clustered_features))



print('5 Clusters')

cluster_features(5)

print('\n4 Clusters')

cluster_features(4)



clustered_features = ['Contract', 'MonthlyCharges', 'tenure', 'PaymentMethod']
#generating interaction terms

x_interaction = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit(X)

interaction_df = pd.DataFrame(x_interaction.transform(X), columns = x_interaction.get_feature_names(X.columns))

interaction_model = linear_model.OLS(y, interaction_df).fit()



X = interaction_df

len(X.columns)
interaction_pvalues_05 = interaction_model.pvalues[interaction_model.pvalues < 0.05].sort_values(ascending=True)

interaction_pvalues_01 = interaction_model.pvalues[interaction_model.pvalues < 0.01].sort_values(ascending=True)

interaction_features_05 = interaction_pvalues_05.index[:]

interaction_features_01 = interaction_pvalues_01.index[:]

print(len(interaction_features_05))

print(len(interaction_features_01))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = RandomForestClassifier(class_weight='balanced', 

                           n_estimators=100).fit(X_train,y_train)



feature_imp = pd.DataFrame(clf.feature_importances_, index=X.columns)

feature_imp = feature_imp[0].sort_values(ascending=False)



sns.barplot(x=feature_imp[:20], y=feature_imp[:20].index, color = '#3274A1')

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Important Features - Random Forest")

plt.legend()

plt.show()



sns.barplot(x=interaction_pvalues_05, y=interaction_features_05, color = '#3274A1')

plt.xlabel('P-Values')

plt.ylabel('Features')

plt.title("Important Features - Polinomial Features")

plt.legend()

plt.show()

# improve performance by limiting to the most important features

X_regduced = X[interaction_features_05] 



# Dummy variables for logistic regression

X_dummy = pd.get_dummies(X, drop_first = True)

X_dummy_reduced = pd.get_dummies(X[interaction_features_05], drop_first = True)
scaler = StandardScaler()

scaler.fit(X)

scaled_X = scaler.transform(X)



pca = PCA(n_components=2)

pca_X = pca.fit(scaled_X).transform(scaled_X)



plt.figure()

plt.figure(figsize=(8,8))

plt.xticks(fontsize=12)

plt.yticks(fontsize=14)

plt.xlabel('Principal Component - 1',fontsize=20)

plt.ylabel('Principal Component - 2',fontsize=20)

plt.title("Principal Component Analysis of Customer Churn",fontsize=20)

targets = [0, 1]

for target in targets:

    indicesToKeep = df['Churn'] == target

    plt.scatter(pca_X.loc[indicesToKeep, 0]

               , pca_X.loc[indicesToKeep, 1], cmap = 'coolwarm', s = 10)





plt.legend(['No Churn', 'Churn'],prop={'size': 15})
def create_confusion_matrix(y_test, y_pred, predicted_proba, title):

    cf_matrix = confusion_matrix(y_test, y_pred)



    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ['{0:0.0f}'.format(value) for value in

                    cf_matrix.flatten()]

    group_percentages = ['{0:.2%}'.format(value) for value in

                         cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in

              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    

    plt.title(title +  

              '\nRecall ' + str(round(recall_score(y_test, y_pred), 2)) +

              '\nPrecision ' + str(round(precision_score(y_test, y_pred), 2)) + 

              '\nAUC ' + str(round(roc_auc_score(y_test, predicted_proba[:,1]), 2)) +

              '\nF1 ' + str(round(f1_score(y_test, y_pred), 2)) +

              '\nAccuracy ' + str(round(accuracy_score(y_test, y_pred), 2)))

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

svm = SVC(class_weight='balanced', probability=True)



plt.subplots_adjust(left=1, bottom=.5, right=4, top=1, wspace=.5, hspace=None)



plt.subplot(1, 4, 1)

predicted_proba = cross_val_predict(svm, pca_X, y, cv=5, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'PCA')



plt.subplot(1, 4, 2)

predicted_proba = cross_val_predict(svm, X[interaction_features_05], y, cv=5, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Interaction Features p<.05')



plt.subplot(1, 4, 3)

predicted_proba = cross_val_predict(svm, X[interaction_features_01], y, cv=5, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Interaction Features p<.01')



plt.subplot(1, 4, 4)

predicted_proba = cross_val_predict(svm, X[clustered_features], y, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Important Features Clustered')
lr = LogisticRegression(class_weight='balanced', 

                            max_iter=1000)



plt.subplots_adjust(left=1, bottom=.5, right=4, top=1, wspace=.5, hspace=None)



plt.subplot(1, 4, 1)

predicted_proba = cross_val_predict(lr, pca_X, y, cv=5, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'PCA')



plt.subplot(1, 4, 2)

predicted_proba = cross_val_predict(lr, X[interaction_features_05], y, cv=5, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Interaction Features p<.05')



plt.subplot(1, 4, 3)

predicted_proba = cross_val_predict(lr, X[interaction_features_01], y, cv=5, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Interaction Features p<.01')



plt.subplot(1, 4, 4)

predicted_proba = cross_val_predict(lr, X[clustered_features], y, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Important Features Clustered')
rf = RandomForestClassifier(class_weight='balanced', 

                               max_depth=5, 

                               n_estimators=50)



plt.subplots_adjust(left=1, bottom=.5, right=4, top=1, wspace=.5, hspace=None)



plt.subplot(1, 4, 1)

predicted_proba = cross_val_predict(rf, pca_X, y, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'PCA')



plt.subplot(1, 4, 2)

predicted_proba = cross_val_predict(rf, X[interaction_features_05], y, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Interaction Features p<.05')



plt.subplot(1, 4, 3)

predicted_proba = cross_val_predict(rf, X[interaction_features_01], y, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Interaction Features p<.01')



plt.subplot(1, 4, 4)

predicted_proba = cross_val_predict(rf, X[clustered_features], y, method='predict_proba')

y_pred = (predicted_proba[:,1] >= 0.5).astype('int')

create_confusion_matrix(y, y_pred, predicted_proba, 'Important Features Clustered')
model_params = {

    'logistic_regression': {

        'model': LogisticRegression(class_weight = 'balanced', max_iter=1000),

        'params': {

            'C': range(1, 20)

        }

    },

    'random_forest': {

        'model': RandomForestClassifier(class_weight = 'balanced'),

        'params': {

            'max_depth': range(1, 20),

            'n_estimators': range(1, 100)

        }

    }

}



scores = []



from sklearn.model_selection import RandomizedSearchCV



for model_name, mp in model_params.items():

    clf = RandomizedSearchCV(mp['model'], mp['params'], scoring = 'roc_auc', return_train_score = False)

    clf.fit(X[interaction_features_01], y)

    scores.append({

        'model': model_name,

        'best_auc_score': clf.best_score_,

#         'results': clf.cv_results_

        'best_params': clf.best_params_

    })

    

for model_name, mp in model_params.items():

    clf = RandomizedSearchCV(mp['model'], mp['params'], scoring = 'recall', return_train_score = False)

    clf.fit(X[interaction_features_01], y)

    scores.append({

        'model': model_name,

        'best_recall_score': clf.best_score_,

#         'results': clf.cv_results_

        'best_params': clf.best_params_

    })
scores
# chosen features

print(interaction_features_01)

print(len(interaction_features_01))
plt.subplots_adjust(left=1, bottom=.5, right=4, top=1, wspace=.5, hspace=None)



for i in range(3):

    threshold = round((i / 10) + 0.4, 1)

    plt.subplot(1, 3, i + 1)

    predicted_proba = cross_val_predict(rf, X[interaction_features_01], y, method='predict_proba')

    y_pred = (predicted_proba[:,1] >= threshold).astype('int')

    create_confusion_matrix(y, y_pred, predicted_proba, str(threshold) + ' Threshold')