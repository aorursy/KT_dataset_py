import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from IPython.display import display
from scipy import stats

import warnings
%matplotlib inline 
np.random.seed(42)
warnings.filterwarnings("ignore")
hr_df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
hr_df.head()
hr_df.info()
# Changing numeric values to corresponding categorical values
hr_df['Education'] = hr_df['Education'].map({1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Masters', 5: 'Doctor'})
hr_df['EnvironmentSatisfaction'] = hr_df['EnvironmentSatisfaction'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
hr_df['JobInvolvement'] = hr_df['JobInvolvement'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
hr_df['JobSatisfaction'] = hr_df['JobSatisfaction'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
hr_df['RelationshipSatisfaction'] = hr_df['RelationshipSatisfaction'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
hr_df['PerformanceRating'] = hr_df['PerformanceRating'].map({1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'})
hr_df['WorkLifeBalance'] = hr_df['WorkLifeBalance'].map({1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'})
hr_df = hr_df.drop(['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], axis = 1)
#making categorical and numerical data frames
hr_categorical = []
hr_numerical = []
for column in hr_df:
    if type(hr_df[column][1]) == str:
        hr_categorical.append(column)
    
    else:
        hr_numerical.append(column)
        
numerical_df = hr_df[hr_numerical]
categorical_df = hr_df[hr_categorical]
# histograms of the numerical data
fig = plt.figure(figsize = (15,30))
i = 0

for column in numerical_df:
    i += 1
    fig.add_subplot(6,3,i)
    plt.hist(numerical_df[column])
    plt.title(column)
plt.tight_layout()
hr_df['Attrition'] = hr_df['Attrition'].map({'Yes': 1, 'No': 0})
fig = plt.figure(figsize = (30,40))
i = 0

for col in categorical_df:
    i += 1
    fig.add_subplot(5,3,i)
    sns.countplot(categorical_df[col])
    plt.xticks(rotation=35, fontsize = 20)
    plt.title(col, fontsize = 20)
    
plt.tight_layout()
cor = numerical_df.corr()
plt.figure(figsize = (15,15))
sns.heatmap(cor, annot = True)
plt.show()
plt.figure(figsize = (10,8))
big_cor = cor.where(abs(cor) > .6)
sns.heatmap(big_cor.replace(np.nan, 0))
plt.title('Correlation Heat Map with High Correlations Highlighted')
plt.show()
sns.boxplot(hr_df['YearsAtCompany'])
plt.title('Box Plot for Years at Company')
plt.show()
threshold = np.std(hr_df['YearsAtCompany']) * 3 # 3 std above mean
hr_df = hr_df[hr_df['YearsAtCompany'] < np.mean(hr_df['YearsAtCompany']) + threshold]
from lifelines import KaplanMeierFitter
sns.countplot(hr_df['Attrition'])
plt.text(x = -.15, y = 800, s = str(np.round(1233/1470.0, 4) * 100) + '%', fontsize = 16)
plt.text(x = .85, y = 100, s = str(np.round(237/1470.0, 4) * 100) + '%', fontsize = 16)
plt.xticks(np.arange(2),('No', 'Yes'))
plt.show()
from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report, confusion_matrix, \
                        precision_recall_fscore_support
hr_df = pd.get_dummies(hr_df, drop_first = True) #to avoid multicolinearity
X = hr_df.drop('Attrition', axis = 1)
y = hr_df['Attrition']
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

#scale the data
scaler = StandardScaler()
# Fit_transform
X_train_scaled = scaler.fit_transform(X_train)
# transform
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
xgbparams = {
    'max_depth':[1,3,5],
    'learning_rate':[.1,.5,.7,.8],
    'n_estimators':[25,50,100]
}

xgb_gs = GridSearchCV(XGBClassifier(), param_grid = xgbparams, cv=5, n_jobs=-1, verbose = 1)
xgb_gs.fit(X_train_scaled, y_train)
xgb_gs.best_params_
print('Train acc =', xgb_gs.score(X_train_scaled, y_train))
print('Test acc = ', xgb_gs.score(X_test_scaled, y_test))
y_pred = xgb_gs.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
from sklearn.neural_network import MLPClassifier
mlpparams = {
            'learning_rate': ["constant", "invscaling", "adaptive"],
            'hidden_layer_sizes': [(30,), (60,), (50,), (40,)],
            'alpha': [.1],
            'activation': ["logistic", "relu", "tanh"]
            }

mlp_gs = GridSearchCV(MLPClassifier(), param_grid = mlpparams, cv = 5, verbose = 1)
mlp_gs.fit(X_train_scaled, y_train)
mlp_gs.best_params_
print('Train acc =', mlp_gs.score(X_train_scaled, y_train))
print('Test acc =', mlp_gs.score(X_test_scaled, y_test))
y_pred = mlp_gs.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
xgb = XGBClassifier(max_depth = 1, learning_rate = .8, n_estimators = 50)
xgb.fit(X_train_scaled, y_train)
zipped = list(zip(X.columns, xgb.feature_importances_))
zipped = pd.DataFrame(zipped)
zipped = zipped.sort_values(by = 1)
zipped = zipped.iloc[27:]
plt.figure(figsize=(10,10))
plt.barh(np.arange(30), zipped[1],)
plt.yticks(np.arange(30), (list(zipped[0])))
plt.ylabel('Feature', fontsize=15)
plt.xlim(xmin = .015, xmax = .14)
plt.xlabel('Importance')
plt.show()
from sklearn.metrics import precision_score, recall_score
mlp = MLPClassifier(activation = 'logistic', hidden_layer_sizes = (60,), alpha = .1, \
                    learning_rate = 'adaptive')
mlp.fit(X_train_scaled, y_train)
def best_threshold (model, steps, X, y, p):
    salary = 50000.0
    bonus = 5000.0
    TN_cost = 0
    TP_cost = p*bonus + (1-p)*.5*salary
    FP_cost = bonus
    FN_cost = .5*salary
    
    cost = []
    threshold = 0
    
    m = model
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    #scale the data
    scaler = StandardScaler()
    # Fit_transform
    X_train_scaled = scaler.fit_transform(X_train)
    # transform
    X_test_scaled = scaler.transform(X_test)
    
    m.fit(X_train_scaled, y_train)
    
    for i in range(steps + 1):
        y_pred_train = (model.predict_proba(X_train_scaled)[:,1] > threshold)
        y_pred_test = (model.predict_proba(X_test_scaled)[:,1] > threshold)
        
        cm = confusion_matrix(y_test, y_pred_test)
        TN = cm[0,0]
        TP = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]
        
        total_cost = TN_cost*TN + TP_cost*TP + FP_cost*FP + FN_cost*FN
        results_dict = {
                'threshold' : threshold,
                'cost' : total_cost,
                'precision_score_test': precision_score(y_test, y_pred_test),
                'recall_score_test': recall_score(y_test, y_pred_test),
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TP': TP,
                        }
        cost.append(results_dict)
        threshold += (1.0/steps) 
    
    thresh_results = pd.DataFrame(cost, columns=['cost', 'threshold', 'precision_score_test',\
                        'recall_score_test','FN', 'FP','TN','TP'])
    return thresh_results
fig = plt.figure(figsize = (15,90))
j = 0
probabilities = -(np.sort(-np.linspace(0,1,11)))

for i in probabilities:
    df1 = best_threshold(xgb,20,X,y, i)
    df2 = best_threshold(mlp,20,X,y, i)
    
    j += 1
    fig.add_subplot(12,1,j)
    plt.plot(df2['threshold'], df2['cost'], label = 'ANN')
    plt.plot(df1['threshold'], df1['cost'], label = 'XGBoost')
    plt.plot(np.linspace(0,1,9), 1375000*np.ones(9), label = 'Overhead Cost')
    plt.ylabel('Cost')
    plt.xlabel('Probability Threshold')
    plt.title('Comparing Models to Minimze Cost')
    plt.text(x = .325, y = 1500000, s = 'Probability that bonus is successful = ' + str(i), fontsize = 14)

    m = df2['cost'].min()
    profit = 1375000 - m
    plt.text(x = .325, y = 1400000, s = 'Savings = $' + str(profit), fontsize = 14)
    plt.legend()
plt.tight_layout()
