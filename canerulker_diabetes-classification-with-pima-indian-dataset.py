#close warnings
import warnings
warnings.simplefilter(action='ignore')

#import libraries
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
#read dataset
df_diabetes=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
#copy dataset in case of reloading
df=df_diabetes.copy()
#lets look at dataset
print(df.shape)
df.head()
df.info()
df.isnull().sum()
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
#look at outcome values
df["Outcome"].value_counts()
#Suspicious values:)
#Zero value is possible or NaN? >> NaN
df[df[['BloodPressure', 'Glucose', 
       'SkinThickness', 'Insulin', 'BMI']]==0].count()
#corr matrix
plt.subplots(figsize = (12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix', fontsize = 20)
plt.show()

#Comment: we don't observe strong correlation between features (max 0.54)
#change 0 values to NaN
df[['Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness',
                            'Insulin','BMI']].replace(0, np.nan)
df.isnull().sum()
#Feature Engineering-01
#Age groups: 18-25: YoungAdults, 26-35: Adults, 36-45: MiddleAged, 46+: Elderly
print("Age Min: ", df['Age'].min())
print("Age Max: ", df['Age'].max())
df['New_Age']=pd.cut(x=df['Age'], bins=[18, 25, 35, 45, 100],
                         labels=['YoungAdults', 'Adults', 'MiddleAged', 'Elderly'])
print(df['New_Age'].value_counts())
print("New_Age is created")
print('----------')
df.head()
#in my previous notebooks, AGE seems one of the most important features on test models
#then i finally prefer it to fill missing values (temporarily categorical)
spec_fill_cols=['Insulin', 'SkinThickness', 'Glucose','BloodPressure', 'BMI']

for col in spec_fill_cols:
    for cat in df['New_Age'].unique():
            temp_fill = df.groupby(['New_Age', 'Outcome'])[col].mean()
            df.loc[(df[col].isnull()) & (df['Outcome'] == 0 ) & (df['New_Age']==cat), col] = temp_fill[cat, 0]
            df.loc[(df[col].isnull()) & (df['Outcome'] == 1 ) & (df['New_Age']==cat), col] = temp_fill[cat, 1]
    print(temp_fill)
print("Missing values were filled automatically.")
df.isnull().sum()
df.drop(['New_Age'], axis=1, inplace=True)
df.info()
#check outliers with quantiles
for feature in df:
    Q1 = df[feature].quantile(0.05)
    Q3 = df[feature].quantile(0.95)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature, "**UPPER: YES**", "LIMIT:", upper, "QTY:", df[(df[feature] > upper)][feature].count())
    else:
        print(feature, "UPPER: NO")

    if df[(df[feature] < lower)].any(axis=None):
        print(feature, "**UPPER: YES**", "LIMIT:", lower, "QTY:", df[(df[feature] > upper)][feature].count())
    else:
        print(feature, "LOWER: NO")
        
#suppress outliers slightly
for feature in df:
    Q1 = df[feature].quantile(0.05)
    Q3 = df[feature].quantile(0.95)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df.loc[df[feature] < lower, feature] = lower
    df.loc[df[feature] > upper, feature] = upper
#check outliers with LOF
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 8)
lof.fit_predict(df)
lof_scores = lof.negative_outlier_factor_
print(np.sort(lof_scores)[0:20])

#how to visualize lof scores
df_lof_scores = pd.DataFrame(np.sort(lof_scores))
df_lof_scores.plot(stacked=True, xlim=[0,20], style='.-')
#drop LOF outliers
threshold = np.sort(lof_scores)[4]
outliers = lof_scores < threshold
print(df[outliers])
df=df[~outliers].reset_index(drop = True)

print(df.shape)
# Feature Engineering-02
new_cols=[]
old_cols=[]

# <140: Normal, <200: Prediabetes, 200+: Diabetes 
print("Glucose Max: ", df['Glucose'].min())
print("Glucose Max: ", df['Glucose'].max())
df['New_Glucose']=pd.cut(x=df['Glucose'], bins=[0, 139.99, 199.99, 1000],
                         labels=['Normal', 'Prediabetes', 'Diabetes'])
print(df['New_Glucose'].value_counts())
print("New_Glucose is created")
new_cols.append('New_Glucose')
old_cols.append('Glucose')
print('----------')

# <80: Normal/Elevated, <89: Hyper Stage 1, <120: Hyper Stage 2, 120+: Hyper Crisis
print("BloodPressure Max: ", df['BloodPressure'].min())
print("BloodPressure Max: ", df['BloodPressure'].max())
df['New_BloodPressure'] = pd.cut(x=df['BloodPressure'], bins=[0, 79.99, 88.99, 1000], 
                         labels=['Normal', 'HyperStage1', 'HyperStage2'])
print(df['New_BloodPressure'].value_counts())
print("New_BloodPressure is created")
new_cols.append('New_BloodPressure')
old_cols.append('BloodPressure')
print('----------')

# 16-166: Normal
print("Insulin Max: ", df['Insulin'].min())
print("Insulin Max: ", df['Insulin'].max())
df['New_Insulin'] = pd.cut(x=df['Insulin'], bins=[0, 15.99, 159.99, 1000], 
                         labels=['Low Anormal', 'Normal', 'High Anormal'])
print(df['New_Insulin'].value_counts())
print("New_Insulin is created")
new_cols.append('New_Insulin')
old_cols.append('Insulin')
print('----------')

# <18.5: underweight, <= 24.9 : normal/healthy, <= 29.9: overweight, 30+: obese
# 30 - 34.9: obese class 1, 35 - 39.9: obese class 2, 40+: obese class 3
print("BMI Max: ", df['BMI'].min())
print("BMI Max: ", df['BMI'].max())
df['New_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.49, 24.99, 29.99, 34.9, 39.9, 1000], 
                         labels=['Underweight', 'Normal', 'Overweight', 'Obese CL.1', 'Obese CL.2', 'Obese CL.3'])
print(df['New_BMI'].value_counts())
print("New_BMI is created")
new_cols.append('New_BMI')
old_cols.append('BMI')
print('----------')

print("New Cols: ", new_cols)
print("Old Cols: ", old_cols)
df.head()
#Feature Encoding-01
df.drop(old_cols, axis=1, inplace=True)
#continuous features that already transformed categorical features drop before encoding
df = pd.get_dummies(df, columns = new_cols, drop_first = True)
df.head()
df.info()
#Feature Scaling-02
cat_cols=[col for col in df.columns if df[col].dtype.name=='uint8']
print("Categorical Columns: ", cat_cols)
y = df[['Outcome']]
toBeSc_X = df.drop(cat_cols, axis = 1)
toBeSc_X = toBeSc_X.drop('Outcome', axis=1)
print("y Shape:", y.shape)
print("To Be Scaled X Shape: ", toBeSc_X.shape)
print("To Be Scaled X Columns: ", toBeSc_X.columns)
#Normalization
normalized_X=preprocessing.normalize(toBeSc_X)
normalized_X=pd.DataFrame(normalized_X, columns=toBeSc_X.columns)
normalized_X.shape

#StandardScaling
stdScaled_X=StandardScaler().fit_transform(toBeSc_X)
stdScaled_X=pd.DataFrame(stdScaled_X, columns=toBeSc_X.columns)
stdScaled_X.shape

#RobustScaling
robScaled_X=RobustScaler().fit_transform(toBeSc_X)
robScaled_X=pd.DataFrame(robScaled_X, columns=toBeSc_X.columns)
robScaled_X.shape

#MinMaxScaling
minMaxScaled_X=MinMaxScaler().fit_transform(toBeSc_X)
minMaxScaled_X=pd.DataFrame(minMaxScaled_X, columns=toBeSc_X.columns)
minMaxScaled_X.shape

#RobustScaling was selected because of best accuracy score on primitive model scores
#concat df after scaling
#X=pd.concat([robScaled_X.reset_index(drop=False), df[cat_cols].reset_index(drop=False)], axis = 1)
X=pd.concat([robScaled_X, df[cat_cols]], axis = 1)
print(X.shape)
X.tail(10)
# Validation scores of all primitive models
pr_models = []
pr_models.append(('LOG', LogisticRegression(random_state = 42)))
pr_models.append(('KNN', KNeighborsClassifier()))
pr_models.append(('SVM', SVC(random_state = 42)))
#pr_models.append(('ANN', MLPClassifier(random_state = 42)))
pr_models.append(('CART', DecisionTreeClassifier(random_state = 42)))
pr_models.append(('RF', RandomForestClassifier(random_state = 42)))
#pr_models.append(('GBM', GradientBoostingClassifier(random_state = 42)))
pr_models.append(('XGB', GradientBoostingClassifier(random_state = 42)))
pr_models.append(("LGBM", LGBMClassifier(random_state = 42)))
#pr_models.append(('CATB', CatBoostClassifier(random_state = 42)))

# evaluate each model in turn
pr_names = []
pr_results = []

for pr_name, pr_model in pr_models:
        pr_cv_results = cross_val_score(pr_model, X, y, cv = 10, scoring= "accuracy")
        pr_results.append(pr_cv_results)
        pr_names.append(pr_name)
        msg = "%s: %f (%f)" % (pr_name, pr_cv_results.mean(), pr_cv_results.std())
        print(msg)
        
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,10))
fig.suptitle('PRIMITIVE ALGORITHMS COMPARISON')
ax = fig.add_subplot(111)
plt.boxplot(pr_results)
ax.set_xticklabels(pr_names)
plt.show()
#lOG: CV Model
log_params={'solver': ['newton-cg', 'lbfgs', 'liblinear']}
log_cv_model = GridSearchCV(LogisticRegression(random_state = 42), 
                            log_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("log_cv_model.best_params_:", log_cv_model.best_params_)
#LOG: Final Model
log_tuned_model=LogisticRegression(**log_cv_model.best_params_, random_state = 42).fit(X, y)
print("log_tuned_model:", cross_val_score(log_tuned_model, X, y, cv = 10).mean())
#KNN: CV Model
knn_params={"n_neighbors": np.arange(2,20,1)}
knn_cv_model = GridSearchCV(KNeighborsClassifier(), 
                            knn_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("knn_cv_model.best_params_:", knn_cv_model.best_params_)
#KNN: Final Model
knn_tuned_model=KNeighborsClassifier(**knn_cv_model.best_params_).fit(X, y)
print("knn_tuned_model:", cross_val_score(knn_tuned_model, X, y, cv = 10).mean())
#SVM: CV Model
svm_params = {"kernel": ['linear', 'poly', 'rbf']}
svm_cv_model = GridSearchCV(SVC(random_state = 42), 
                             svm_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("svm_cv_model.best_params_:", svm_cv_model.best_params_)
#SVM: Final Model
svm_tuned_model=SVC(**svm_cv_model.best_params_, random_state = 42).fit(X, y)
print("svm_tuned_model:", cross_val_score(svm_tuned_model, X, y, cv = 10).mean())
#CART: CV Model
cart_params = {"max_depth": [2, 3, 5, None],
              "min_samples_split": [2, 3, 5, 10]}
cart_cv_model = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                             cart_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("cart_cv_model.best_params_:", cart_cv_model.best_params_)
#CART: Final Model
cart_tuned_model=DecisionTreeClassifier(**cart_cv_model.best_params_).fit(X, y)
print("cart_tuned_model:", cross_val_score(cart_tuned_model, X, y, cv = 10).mean())
#RF: CV Model
rf_params = {"max_depth": [4, 5, 6, 8, 10],
             "max_features": [3, 5, None],
             "min_samples_split": [3, 4, 5, 6],
             "n_estimators": [200, 300, 500]}
rf_cv_model = GridSearchCV(RandomForestClassifier(random_state=42), 
                             rf_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("rf_cv_model.best_params_:", rf_cv_model.best_params_)
#RF: Final Model
rf_tuned_model=RandomForestClassifier(**rf_cv_model.best_params_).fit(X, y)
print("rf_tuned_model:", cross_val_score(rf_tuned_model, X, y, cv = 10).mean())
#XGB: CV Model
xgb_params = {"learning_rate": [0.01, 0.02, 0.03],
              "max_depth":[2, 3, 4, 5],
              "subsample":[0.2, 0.3, 0.4],
              "n_estimators": [100, 200, 300]}
xgb_cv_model = GridSearchCV(GradientBoostingClassifier(random_state=42), 
                             xgb_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("xgb_cv_model.best_params_:", xgb_cv_model.best_params_)
#XGB: Final Model
xgb_tuned_model=GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X, y)
print("xgb_tuned_model:", cross_val_score(xgb_tuned_model, X, y, cv = 10).mean())
#LGBM: CV Model
lgbm_params = {"learning_rate": [0.01, 0.02, 0.03],
              "n_estimators": [500, 600, 700],
              "max_depth":[2, 3, 4],
              "colsample_bytree": [0.8, 0.9, 1, 1.1]}
lgbm_cv_model = GridSearchCV(LGBMClassifier(random_state=42), 
                             lgbm_params, cv=10, n_jobs = -1, verbose=2).fit(X, y)
print("lgbm_cv_model.best_params_:", lgbm_cv_model.best_params_)
#LGBM: Final Model
lgbm_tuned_model=LGBMClassifier(**lgbm_cv_model.best_params_).fit(X, y)
print("lgbm_tuned_model:", cross_val_score(lgbm_tuned_model, X, y, cv = 10).mean())
# Validation scores of some tuned models
td_models = []
td_models.append(('LOG', LogisticRegression(**log_cv_model.best_params_, random_state = 42)))
td_models.append(('KNN', KNeighborsClassifier(**knn_cv_model.best_params_)))
td_models.append(('SVM', SVC(**svm_cv_model.best_params_, random_state = 42)))
#td_models.append(('ANN', MLPClassifier(random_state = 42)))
td_models.append(('CART', DecisionTreeClassifier(**cart_cv_model.best_params_, random_state = 42)))
td_models.append(('RF', RandomForestClassifier(**rf_cv_model.best_params_, random_state = 42)))
#td_models.append(('GBM', GradientBoostingClassifier(random_state = 42)))
td_models.append(('XGB', GradientBoostingClassifier(**xgb_cv_model.best_params_, random_state = 42)))
td_models.append(("LGBM", LGBMClassifier(**lgbm_cv_model.best_params_, random_state = 42)))
#td_models.append(('CATB', CatBoostClassifier(random_state = 42)))

# evaluate each model in turn
td_names = []
td_results = []

for td_name, td_model in td_models:
        td_cv_results = cross_val_score(td_model, X, y, cv = 10, scoring= "accuracy")
        td_results.append(td_cv_results)
        td_names.append(td_name)
        msg = "%s: %f (%f)" % (td_name, td_cv_results.mean(), td_cv_results.std())
        print(msg)
        
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,10))
fig.suptitle('TUNED ALGORITHMS COMPARISON')
ax = fig.add_subplot(111)
plt.boxplot(td_results)
ax.set_xticklabels(td_names)
plt.show()
#LGBM: Feature Importances & Visualization
importance=pd.DataFrame({'importance': lgbm_tuned_model.feature_importances_ * 100}, index=X.columns)

importance.sort_values(by='importance', axis=0, ascending=True). plot(kind='barh', color='g')

plt.xlabel('Variable Importances')
plt.gca().legend_=None
#XGB: Feature Importances & Visualization
importance=pd.DataFrame({'importance': xgb_tuned_model.feature_importances_ * 100}, index=X.columns)

importance.sort_values(by='importance', axis=0, ascending=True). plot(kind='barh', color='g')

plt.xlabel('Variable Importances')
plt.gca().legend_=None
#Confusion matrix for LGBM Tuned Model
y_pred_lgbm = lgbm_tuned_model.predict(X)
conf_lgbm = confusion_matrix(y, y_pred_lgbm)
print(conf_lgbm)
#Confusion matrix for XGB Tuned Model
y_pred_xgb = xgb_tuned_model.predict(X)
conf_xgb = confusion_matrix(y, y_pred_xgb)
print(conf_xgb)
#Confusion matrices for RF Tuned Model
y_pred_rf = rf_tuned_model.predict(X)
conf_rf = confusion_matrix(y, y_pred_rf)
print(conf_rf)
