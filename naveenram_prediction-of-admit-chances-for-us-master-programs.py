import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV

from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,classification_report,roc_auc_score,recall_score, precision_score, f1_score, accuracy_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', index_col=0)

df.head(20)
df.shape
df.info()
df.isnull().sum()
df.drop('Research', axis=1).describe()
plt.figure(figsize=(10,6))

plt.hist(df['Chance of Admit '])

plt.title('Distribution of Target (Chance of Admit)')

pd.DataFrame(df['Chance of Admit '].describe()).T
plt.figure(figsize=(10,6))

plt.hist(df['GRE Score'], bins=20)

plt.title('GRE Score Distribution')

pd.DataFrame(df['GRE Score'].describe()).T
plt.figure(figsize=(10,6))

plt.hist(df['TOEFL Score'], bins=12)

plt.title('TOEFL Score Distribution')

pd.DataFrame(df['TOEFL Score'].describe()).T
plt.figure(figsize=(8,6))

sns.countplot(df['University Rating'])

pd.DataFrame(df['University Rating'].astype('O').describe()).T
plt.figure(figsize=(10,6))

sns.countplot(df['SOP'])

pd.DataFrame(df['SOP'].astype('O').describe()).T
plt.figure(figsize=(10,6))

sns.countplot(df['LOR '])

pd.DataFrame(df['LOR '].astype('O').describe()).T
plt.figure(figsize=(10,6))

plt.hist(df['CGPA'])

plt.title('GCPA Score Distribution')

pd.DataFrame(df['CGPA'].describe()).T
plt.figure(figsize=(6,6))

sns.countplot(df['Research'])

pd.DataFrame(df['Research'].astype('O').describe()).T
pd.DataFrame(df.corr()['Chance of Admit '])[:-1]
sns.pairplot(df, hue='Research')
plt.subplots(1,2 , figsize = (16,6))

plt.subplot(1,2,1)

sns.scatterplot(df['GRE Score'], df['Chance of Admit '])

plt.subplot(1,2,2)

sns.lineplot(df['GRE Score'], df['Chance of Admit '])
plt.subplots(1,2 , figsize = (16,6))

plt.subplot(1,2,1)

sns.scatterplot(df['TOEFL Score'], df['Chance of Admit '])

plt.subplot(1,2,2)

sns.lineplot(df['TOEFL Score'], df['Chance of Admit '])
Uni_Mean = df['Chance of Admit '].groupby(df['University Rating']).agg(['mean'])

sns.lineplot(df['University Rating'],df['Chance of Admit '])

Uni_Mean.plot(kind='bar')

plt.ylabel('Mean Chances of Admit')

Uni_Mean
plt.subplots(2,2, figsize = (16,6))

plt.subplot(1,2,1)

sns.scatterplot(df['SOP'], df['Chance of Admit '])

plt.subplot(1,2,2)

sns.lineplot(df['SOP'], df['Chance of Admit '])
plt.subplots(2,2, figsize = (16,6))

plt.subplot(1,2,1)

sns.scatterplot(df['LOR '], df['Chance of Admit '])

plt.subplot(1,2,2)

sns.lineplot(df['LOR '], df['Chance of Admit '])

df['LOR '].value_counts().tail(1)
plt.figure(figsize=(10,6))

sns.scatterplot(df['CGPA'], df['Chance of Admit '])
Research_Mean = df['Chance of Admit '].groupby(df['Research']).agg(['mean'])

Research_Mean.plot(kind='bar')

plt.ylabel('Mean Chance of Admit')

Research_Mean
pd.DataFrame(df.corr()['GRE Score'][1:-1])
plt.subplots(1,2,figsize=(16,6))

plt.subplot(1,2,1)

sns.scatterplot(df['GRE Score'],df['TOEFL Score'], color='r')

plt.subplot(1,2,2)

sns.lineplot(df['GRE Score'],df['University Rating'], color='g')

plt.subplots(1,2,figsize=(16,6))

plt.subplot(1,2,1)

sns.lineplot(df['GRE Score'],df['SOP'], color='y')

plt.subplot(1,2,2)

sns.lineplot(df['GRE Score'],df['LOR '], color='b')

plt.show()

plt.figure(figsize=(7,6))

sns.scatterplot(df['GRE Score'],df['CGPA'], color='y')

pd.DataFrame(df.groupby('Research')['GRE Score'].agg(['mean']))
x = df.drop('Chance of Admit ', axis=1)

y = df['Chance of Admit ']



x_scaled = pd.DataFrame(StandardScaler().fit_transform(x), columns = x.columns)



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)

x_train_s, x_test_s, y_train, y_test = train_test_split(x_scaled,y, test_size = 0.3, random_state = 1)



print("Training Set :", x_train.shape[0], "entries")

print("Testing Set :", x_test.shape[0], "entries")
l=[]



def models_lr(x,y):

    mod = {}

    model = LinearRegression().fit(x,y)

    ypred = model.predict(x_test)

    mod['Model'] = 'Linear Regression'

    mod['Train_Score'] = model.score(x,y)

    mod['Test_accuracy'] = r2_score(y_test,ypred)

    mod['RMSE'] = np.sqrt(mean_squared_error(y_test,ypred))

    return mod

l.append(models_lr(x_train,y_train))



def models_lr(x,y):

    mod = {}

    model = LinearRegression().fit(x,y)

    ypred = model.predict(x_test_s)

    mod['Model'] = 'Linear Regression with Scaled Data'

    mod['Train_Score'] = model.score(x,y)

    mod['Test_accuracy'] = r2_score(y_test,ypred)

    mod['RMSE'] = np.sqrt(mean_squared_error(y_test,ypred))

    return mod

l.append(models_lr(x_train_s,y_train))



def models_dt(x,y):

    mod = {}

    model = DecisionTreeRegressor().fit(x,y)

    ypred = model.predict(x_test)

    mod['Model'] = 'Decison Tree'

    mod['Train_Score'] = model.score(x,y)

    mod['Test_accuracy'] = r2_score(y_test,ypred)

    mod['RMSE'] = np.sqrt(mean_squared_error(y_test,ypred))

    return mod

l.append(models_dt(x_train,y_train))



def models_rf(x,y):

    mod = {}

    model = RandomForestRegressor().fit(x,y)

    ypred = model.predict(x_test)

    mod['Model'] = 'Random Forest'

    mod['Train_Score'] = model.score(x,y)

    mod['Test_accuracy'] = r2_score(y_test,ypred)

    mod['RMSE'] = np.sqrt(mean_squared_error(y_test,ypred))

    return mod

l.append(models_rf(x_train,y_train))



def models_rf(x,y):

    mod = {}

    model = KNeighborsRegressor().fit(x,y)

    ypred = model.predict(x_test_s)

    mod['Model'] = 'KNN'

    mod['Train_Score'] = model.score(x,y)

    mod['Test_accuracy'] = r2_score(y_test,ypred)

    mod['RMSE'] = np.sqrt(mean_squared_error(y_test,ypred))

    return mod

l.append(models_rf(x_train_s,y_train))



pd.DataFrame(l)
x_train_c = sm.add_constant(x_train)



vif = [ variance_inflation_factor(x_train_c.values, i ) for i in range(x_train_c.shape[1])]

vif_df = pd.DataFrame({'vif' : vif[1:]} , index = x_train.columns)

vif_df
for i in range(1,df.shape[1]):

   

    rfe = RFE(LinearRegression(),i).fit(x,y)

    print(x.columns[rfe.support_])

    x_train1,x_test1,y_train,y_test = train_test_split(rfe.transform(x),y,test_size=0.3,random_state=1)

    LR = LinearRegression()

    LR.fit(x_train1,y_train)

    y_pred1 = LR.predict(x_test1)

    print("RMSE : ",np.sqrt(mean_squared_error(y_test,y_pred1)))

    print("Accuracy : ",r2_score(y_test,y_pred1))

    print('*'*100)

    

params = [{'n_features_to_select' : list(range(1,8))}]

model_cv = GridSearchCV(estimator = rfe, param_grid= params, scoring='r2', cv=5)

model_cv.fit(x_train1, y_train)

print()

print("Number of Features : ",model_cv.best_params_);print()

print('*'*100); print()



rfe1 = RFE(LinearRegression(), n_features_to_select = 6).fit(x_train1,y_train)

y_pred2 = rfe1.predict(x_test)

print("Train Accuracy after RFE feature selection :", rfe1.score(x_train1,y_train))

print("Test Accuracy after RFE feature selection :", r2_score(y_test,y_pred2))

print("RMSE after RFE feature selection :", np.sqrt(mean_squared_error(y_test,y_pred2)))

print(); print('*'*100)



cols = pd.DataFrame({'features': x.columns, 'selection' : rfe1.support_, 'rank' : rfe1.ranking_ })

cols
pd.DataFrame(df.corr()['Chance of Admit '][:-1])
df[['GRE Score','TOEFL Score', 'Chance of Admit ']].corr()
df[['GRE Score','University Rating', 'Chance of Admit ']].corr()
df[['GRE Score','SOP', 'Chance of Admit ']].corr()
df[['GRE Score','LOR ', 'Chance of Admit ']].corr()
df[['GRE Score','CGPA', 'Chance of Admit ']].corr()
df[['CGPA','Research', 'Chance of Admit ']].corr()
x_PC = df[['CGPA']]

x_train_PC, x_test_PC, y_train, y_test = train_test_split(x_PC, y, test_size= 0.3, random_state = 1)



LR_PC = LinearRegression().fit(x_train_PC, y_train)

y_pred_PC = LR_PC.predict(x_test_PC)



print("Train Accuracy After Pearson Correlation : ", LR_PC.score(x_train_PC, y_train))

print("Test Accuracy After Pearson Correlation : ", r2_score(y_test, y_pred_PC))

print("RMSE After Pearson Correlation : ",np.sqrt(mean_squared_error(y_test, y_pred_PC)))
cols = list(x.columns)

pmax=1

while(len(cols)>0):

    p = []

    x_1 = x[cols]

    x_1 = sm.add_constant(x_1)

    model = sm.OLS(y, x_1).fit()

    p = pd.Series(model.pvalues.values[1:], index = cols)

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax > 0.05):

        cols.remove(feature_with_p_max)

    else:

        break       

selected_columns = cols

print("Choosed Features : ",selected_columns)
x_back = df[['GRE Score', 'TOEFL Score', 'LOR ', 'CGPA', 'Research']]

x_train_back, x_test_back, y_train, y_test = train_test_split(x_back, y, test_size= 0.3, random_state = 1)



LR_back = LinearRegression().fit(x_train_back, y_train)

y_pred_back = LR_back.predict(x_test_back)



print("Train Accuracy After Backward Elimination : ", LR_back.score(x_train_back, y_train))

print("Test Accuracy After Backward Elimination : ", r2_score(y_test, y_pred_back))

print("RMSE After Backward Elimination : ",np.sqrt(mean_squared_error(y_test, y_pred_back)))
for i in [100,10,1,0.1, 0.01, 0.001, 0.0001,0.00001, 0.000001]:

    rr = Ridge(alpha=i) 

    rr.fit(x_train_back, y_train) 

    print("Rigde with Learning Rate of",i,":\n")

    print("Train Accuracy After Rigde : ", rr.score(x_train_back, y_train))

    print("Test Accuracy After Rigde : ",rr.score(x_test_back, y_test),'\n')

    print("-"*100,"\n")

    
for i in [0.01, 0.001, 0.0001,0.00001, 0.000001]:

    lasso = Lasso(alpha=i) 

    lasso.fit(x_train_back, y_train) 

    print("Lasso with Learning Rate of",i,":\n")

    print("Train Accuracy After Lasso : ", lasso.score(x_train_back, y_train))

    print("Test Accuracy After Lasso : ",lasso.score(x_test_back, y_test),'\n')

    print("-"*100,"\n")
cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], 

                        normalize=True,copy_X=True, positive=False,)

cv_model.fit(x_train_back, y_train)



EN = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)

EN.fit(x_train_back, y_train)



print("Train Accuracy After ElasticNet : ", EN.score(x_train_back, y_train))

print("Test Accuracy After ElasticNet : ",r2_score(y_test, EN.predict(x_test_back)))
Bagg = BaggingRegressor().fit(x_train_back, y_train)

y_pred_bagg = Bagg.predict(x_test_back)



print("Train Accuracy After Bagging : ", Bagg.score(x_train_back, y_train))

print("Test Accuracy After Bagging : ", r2_score(y_test, y_pred_bagg))

print("RMSE After Bagging : ",np.sqrt(mean_squared_error(y_test, y_pred_bagg)))
AdaBoost = AdaBoostRegressor().fit(x_train_back, y_train)

y_pred_ada = AdaBoost.predict(x_test_back)



print("Train Accuracy After AdaBoost : ", AdaBoost.score(x_train_back, y_train))

print("Test Accuracy After AdaBoost : ", r2_score(y_test, y_pred_ada))

print("RMSE After AdaBoost : ",np.sqrt(mean_squared_error(y_test, y_pred_ada)))
Gradient = GradientBoostingRegressor().fit(x_train_back, y_train)

y_pred_grad = Gradient.predict(x_test_back)



print("Train Accuracy After Gradient Boost : ", Gradient.score(x_train_back, y_train))

print("Test Accuracy After Gradient Boost : ", r2_score(y_test, y_pred_grad))

print("RMSE After Gradient Boost: ",np.sqrt(mean_squared_error(y_test, y_pred_grad)))