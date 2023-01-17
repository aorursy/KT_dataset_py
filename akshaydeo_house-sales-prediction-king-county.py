import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectPercentile ,SelectKBest, chi2
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import random 


#Read data from CSV file using pandas and convert ['date'] column into datetime64
df_house_data=pd.read_csv("../input/housesalesprediction/kc_house_data.csv", parse_dates = ['date'])

#Check for what type of data is present
print(df_house_data.info())

#We have all numerical data except the one date
#3.1.1        Remove Features that do not contribute to the prediction
#Remove Features : Some of the features may not contribute to the prediction of the final class.  
  
df_house_data = df_house_data.drop(['id','date'], 1)
df_house_data.sample(5)
#3.1.2 Outlier Detection Seaborn boxplot 
bxplt = sns.boxplot(data=df_house_data.iloc[:,1:])
plt.setp(bxplt.get_xticklabels(), rotation=90)
plt.show()

#There are no anomilities or unexpected outliers that could have come from incorrect data. Sq feet (area) can vary quite a bit.
#3.1.3   Dealing with Missing Values
#Check for missing values
print(df_house_data.isnull().sum())

#There are no missing values
#As we do not have Categorical Data, we are not using any kind of pre processing encoding technique to convert Categorical Data to numerical form


#As we are building a regression model, we cannot use pre processing for Data imbalance


#3.1.5     Scaling Data
#Scaling the data
scaler = StandardScaler()
stdData = scaler.fit_transform(df_house_data)


X = df_house_data.iloc[:,1:]
y = df_house_data["price"]

#print(X)

#print(y)
#3.1.7    Feature Selection

#Deleting the features which have Percentile score less than 50
selector = SelectPercentile(f_regression, percentile=25) 
selector.fit(X,y)
for n,s in zip(X.columns, selector.scores_): 
    print ("Score : ", s, " for feature ", n)
    if s<50:
        X = X.drop([n], 1)
        print("deleted:", n)
        
print(X.columns)
#DecisionTreeRegressor Model

decreg = DecisionTreeRegressor()
#Cross_val_score uses the KFold or StratifiedKFold strategies by default.
accu1 = cross_val_score(decreg,X,y,cv=5)
print("___DecisionTreeRegressor____\n")
print(accu1)
sc1 =  accu1.mean()
print(sc1)
#Linear Regression Model

linreg=LinearRegression()

accu2 = cross_val_score(linreg,X,y,cv=5)
print("___Linear Regression____\n")
print(accu2)
sc2 =  accu2.mean()
print(sc2)
#KNeighborsRegressor Model

knr = KNeighborsRegressor()

accu3 = cross_val_score(knr,X,y,cv=5)
print("___KNeighborsRegressor____\n")
print(accu3)
sc3 =  accu3.mean()
print(sc3)
#Lasso Model

lasso = linear_model.Lasso(random_state = 42)
accu4 = cross_val_score(lasso,X,y,cv=5)
print("___Lasso____\n")
print(accu4)
sc4 =  accu4.mean()
print(sc4)
#Ensemble methods
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,BaggingRegressor
rf = RandomForestRegressor(random_state = 42)

accu5 = cross_val_score(rf,X,y,cv=5)
print("___RandomForestRegressor____\n")
print(accu5)
sc5 =  accu5.mean()
print(sc5)


br = BaggingRegressor(random_state = 42)

accu6 = cross_val_score(br,X,y,cv=5)
print("___BaggingRegressor____\n")
print(accu6)
sc6 =  accu6.mean()
print(sc6)
gb = GradientBoostingRegressor(random_state = 42)

accu7 = cross_val_score(gb,X,y,cv=5)
print("___GradientBoostingRegressor____\n")
print(accu7)
sc7 =  accu7.mean()
print(sc7)
et = ExtraTreesRegressor(random_state = 42)

accu8 = cross_val_score(et,X,y,cv=5)
print("___ExtraTreesRegressor____\n")
print(accu8)
sc8 =  accu8.mean()
print(sc8)
#4.1 Base Model Results:
#All the scores:
print("DecisionTreeRegressor :",sc1)
print("LinearRegression :",sc2)
print("KNeighborsRegressor :",sc3)
print("linear_model->Lasso Regressor:",sc4)
print("RandomForestRegressor :",sc5)
print("BaggingRegressor :",sc6)
print("GradientBoostingRegressor :",sc7)
print("ExtraTreesRegressor :",sc8)
# Select Top 3 Models
# To find what all parameters a model has to be set 
#ExtraTreesRegressor.get_params(et)
GradientBoostingRegressor.get_params(gb)
#RandomForestRegressor.get_params(rf)
#aggingRegressor.get_params(br)
# HPO GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [ {'n_estimators': [100,150,200] , 'max_depth' : [2,3,4,5] ,'random_state' : list(range(42,43)), 'min_samples_leaf' : [1,2,10],'min_samples_split' : [2,7,15], 'learning_rate' : [0.1,0.2,0.3]  }  ] 
clf = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3, n_jobs = -1)

clf.fit(X,y)

print("\n Best parameters set found on development set:") 
print(clf.best_params_ , "with a score of ", clf.best_score_)
# HPO BaggingRegressor

param_grid = [{ 'bootstrap': [True,False], 'bootstrap_features': [True,False],'random_state' : list(range(42,43)), 'max_features': [12,15,16], 'max_samples': [0.8,1.0,2], 'n_estimators': [10,20,30,40,50]}] 
clf = GridSearchCV(BaggingRegressor(), param_grid, cv=3, n_jobs = -1)

clf.fit(X,y)
print("\n Best parameters set found on development set:") 
print(clf.best_params_ , "with a score of ", clf.best_score_)
# HPO ExtraTreesRegressor

param_grid=[{'n_estimators': list(range(100,201,25)),'max_features': [15,16],'random_state' : list(range(42,43)),'min_samples_leaf': list(range(1,30,5)),'min_samples_split': list(range(15,36,5))}]
clf = GridSearchCV(ExtraTreesRegressor(), param_grid, cv=3, n_jobs = -1)

clf.fit(X,y)
print("\n Best parameters set found on development set:") 
print(clf.best_params_ , "with a score of ", clf.best_score_)
# 4.3.1 Tree Based Feature Selection (RandomForestRegressor)
import numpy as np

#We again build X to include all features (We had deleted 2 using the SelectPercentile Univariate Feature Selection Tool)

X = df_house_data.iloc[:,1:]
clf = RandomForestRegressor(random_state=42)
clf.fit(X, y)
print(clf.feature_importances_)

#We use  np.argsort to sort the features according to the ascending order of the indices of the feature scores. Then we  iteratively remove features from the dataset (starting with the weakest features) and calculate cross_val_score every time using Bagging Regressor with the optimal hyper parameters that we found out earlier.
imp_features_asc = np.argsort(clf.feature_importances_)
print(imp_features_asc)
print(X.shape[1])
allAccuracies = []
numberOfFeatures = []
br = BaggingRegressor(bootstrap= False, bootstrap_features= True, max_samples= 0.8, n_estimators= 50, random_state= 42)
for i in range(0,13):
    numberOfFeatures.append(i)
    X_new = X.drop(X.columns[[imp_features_asc[:i]]], axis = 1)
    
    accur9 = cross_val_score(br,X_new,y,cv=3)
    allAccuracies.append(accur9.mean())

print(allAccuracies)
    

plt.figure()
plt.xlabel("Number of features removed")
plt.ylabel("Cross validation score ")
plt.plot(numberOfFeatures, allAccuracies)
plt.show()
# 4.3.2 Feature Selection based on Pearson Coefficient (Correlation)

X = df_house_data.iloc[:,1:]
corr = df_house_data.corr()
plt.figure(figsize=(20,16))
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
plt.show()

from scipy.stats import pearsonr
#It helps to measures relationship of every feature with the target regression value

#We use the Pearson Correlation Coefficient (pearsonr) to calculate the correlations between each feature and price i.e. the target regression value. Then we select the top 16 features which have the highest Correlation values and pass them through the ML models to calculate scores.
correlations_all = {}
for f in X.columns:
    temp = df_house_data[[f,'price']]
    x1 = temp[f].values
    x2 = temp['price'].values
    key = f + ' || ' + 'price'
    correlations_all[key] = pearsonr(x1,x2)[0]
    

data_correlations = pd.DataFrame(correlations_all, index=['Value']).T
print(data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index])


top16 = list(data_correlations['Value'].abs().sort_values(ascending=False).index[:16])

top16 = [x[:-9] for x in top16]
print(top16)
br = BaggingRegressor(bootstrap= False, bootstrap_features= True, max_samples= 0.8, n_estimators= 50, random_state= 42)
X_new1 = X[top16]
print("Mean score:",cross_val_score(br,X_new1,y,cv=3).mean())
# 4.3.3 Backward Elimination
import statsmodels.api as sm
    
#Backward Elimination
#If the pvalue is above 0.05 then we remove the feature, else we keep it.
#We are using OLS model “Ordinary Least Squares” used for performing linear regression. We will remove the feature which has max pvalue and build the model once again. This is an iterative process and will keep on deleting features and checking pvalues in a loop.

cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)
    print(p.sort_values())
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
        print(feature_with_p_max," feature removed")
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
br = BaggingRegressor(bootstrap= False, bootstrap_features= True, max_samples= 0.8, n_estimators= 50, random_state= 42)
print("Mean score:",cross_val_score(br,X_new1,y,cv=3).mean())
# 4.3.4 RFE (Recursive Feature Elimination) [2]
from sklearn.feature_selection import RFE

X = df_house_data.iloc[:,1:]
cols = list(X.columns)
model = RandomForestRegressor(random_state=42)
scores = []
selected_features = []
#We increment the features one by one in a loop and calculate the RFE scores for each number of features selected. From the list of all scores, we find the maximum and the features that were used to obtain this maximum score. We then use these features in our optimized Bagging Regressor model and find the accuracy score
for i in range(4,15):
    #Initializing RFE model
    rfe = RFE(model, i)             
    #Transform data using RFE
    X_rfe = rfe.fit_transform(X,y)  
    #Fitting the data to model
    model.fit(X_rfe,y)              
    features_series = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = features_series[features_series==True].index
    print(selected_features_rfe)
    scores.append(model.score(X_rfe,y))
    selected_features.append(selected_features_rfe)
print(scores)
ind = scores.index(max(scores))
most_imp_features=selected_features[ind]
print(most_imp_features)
br = BaggingRegressor(bootstrap= False, bootstrap_features= True, max_samples= 0.8, n_estimators= 50, random_state= 42)
print("Mean score:",cross_val_score(br,X[most_imp_features],y,cv=3).mean())
