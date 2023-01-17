#Standard libs
import numpy as np 
import pandas as pd 

#Data Visualisation libs
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#Feature engineering, metrics and modeling libs
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing.imputation import Imputer
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics

#Detect Missing values
import missingno as msno


import os
print(os.listdir("../input"))


abalone = pd.read_csv('../input/abalone.csv')
abalone.head()
abalone.info()
abalone.describe()
abalone.shape
len(abalone.isnull())
rings = abalone['Rings']
plt.figure(1);plt.title('Normal')
sns.distplot(rings,kde=False,fit=st.norm)
plt.figure(2);plt.title('Johnson SU')
sns.distplot(rings,kde=False,fit=st.johnsonsu)
plt.figure(3);plt.title('Log Normal')
sns.distplot(rings,kde=False,fit=st.lognorm)
plt.hist(rings,color='green')
plt.title('Histogram of rings')
plt.xlabel('Number of Rings')
plt.ylabel('count')
numeric_features = abalone.select_dtypes(include=[np.number])
correlation = numeric_features.corr()
print(correlation['Rings'].sort_values(ascending=False))
plt.title('Correlation of numeric features', y=1,size=16)
sns.heatmap(correlation,square=True,vmax=0.8)
cols = ['Rings','Shell weight','Diameter','Height','Length']
sns.pairplot(abalone[cols],size=2,kind='scatter')
plt.show()
sns.countplot(abalone['Sex'], palette="Set3")
plt.title('Count of the Gender of Abalone')
sns.countplot(abalone['Rings'])
plt.title('Distribution of Rings')
p = sns.FacetGrid(abalone, col="Sex", hue="Sex")
p=p.map(plt.scatter,"Rings", "Shell weight")

x = sns.FacetGrid(abalone,col="Sex",hue="Sex")
x.map(plt.scatter, "Rings", "Diameter")
f = (abalone.loc[abalone['Sex'].isin(['M','F'])]
      .loc[:,['Shell weight','Rings','Sex']])

f = f[f["Rings"] >= 8]
f = f[f["Rings"] < 23]
sns.boxplot(x="Rings",y="Shell weight", hue='Sex',data=f)
w = (abalone.loc[abalone['Sex'].isin(['I'])]
    .loc[:,['Shell weight','Rings','Sex']])
w = w[w["Rings"] >= 8]
w = w[w["Rings"] < 23]
sns.boxplot(x="Rings",y="Shell weight", hue='Sex',data=w)
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,10))

ShellWeight_plot = pd.concat([abalone['Rings'],abalone['Shell weight']],axis=1)
sns.regplot(x='Rings',y='Shell weight',data=ShellWeight_plot,scatter=True,fit_reg=True,ax=ax1)

Diameter_plot = pd.concat([abalone['Rings'],abalone['Diameter']],axis=1)
sns.regplot(x='Rings',y='Diameter',data=Diameter_plot,scatter=True,fit_reg=True,ax=ax2)

from scipy import stats
z= np.abs(stats.zscore(abalone.select_dtypes(include=[np.number])))
print(z)
abalone_o = abalone[(z < 3).all(axis=1)]
print("Shape of Abalones with outliers: "+ str(abalone.shape) , 
      "Shape of Abalones without outliers: " + str(abalone_o.shape))
low_cardinality_cols = [cname for cname in abalone_o.columns if
                        abalone_o[cname].nunique() < 10 and 
                       abalone_o[cname].dtype == "object"]
numeric_cols = [cname for cname in abalone_o.columns if
                                 abalone_o[cname].dtype in ['int64','float64']]

my_cols = low_cardinality_cols + numeric_cols
abalone_predictors = abalone_o[my_cols]
abalone_predictors.dtypes.sample(7)
abalone_encoded_predictors = pd.get_dummies(abalone_predictors)
abalone_encoded_predictors.head(5)
abalone_encoded_predictors.shape
cross_cols = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Sex_F','Sex_I','Sex_M']
X = abalone_encoded_predictors[cross_cols]
y = abalone_encoded_predictors.Rings

decision_pipeline = make_pipeline(DecisionTreeRegressor())
decision_scores = cross_val_score(decision_pipeline, X,y,scoring='neg_mean_absolute_error')

print('MAE %2f' %(-1 * decision_scores.mean()))
dt_train_X,dt_test_X,dt_train_y,dt_test_y = train_test_split(X,y)
def get_mae(max_leaf_nodes,dt_train_X,dt_test_X,dt_train_y,dt_test_y ):
    model_pipeline = make_pipeline(DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0))
    model_pipeline.fit(dt_train_X,dt_train_y)
    preds_val = model_pipeline.predict(dt_test_X)
    mae = mean_absolute_error(dt_test_y,preds_val)
    return(mae)
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes,dt_train_X,dt_test_X,dt_train_y,dt_test_y)
    print("Max leaf nodes: %d \t\t MAE: %d" %(max_leaf_nodes,my_mae))
decision_split_pipeline = make_pipeline(DecisionTreeRegressor(max_leaf_nodes=5))
decision_split_pipeline.fit(dt_train_X,dt_train_y)
decision_tree_prediction = decision_split_pipeline.predict(dt_test_X)
print("MAE: " + str(mean_absolute_error(decision_tree_prediction,dt_test_y)))
acc_decision = decision_split_pipeline.score(dt_test_X,dt_test_y)
print("Acc:", acc_decision )
plt.scatter(dt_test_y,decision_tree_prediction,color='green')
plt.xlabel('Actuals')
plt.ylabel('Predictions')
plt.title('Decision Tree: Actuals vs Predictions')
plt.show()
forest_pipeline = make_pipeline(RandomForestRegressor(random_state=1))
forest_scores = cross_val_score(forest_pipeline, X,y,scoring="neg_mean_absolute_error")
print('MAE %2f' %(-1 * forest_scores.mean()))
f_train_X,f_test_X,f_train_y,f_test_y = train_test_split(X,y)
forest_split_pipeline = make_pipeline(RandomForestRegressor(random_state=1))
forest_split_pipeline.fit(f_train_X,f_train_y)
forest_predictions = forest_split_pipeline.predict(f_test_X)
print("Accuracy:",forest_split_pipeline.score(f_test_X,f_test_y))
print("MAE:",str(mean_absolute_error(forest_predictions,f_test_y)))

plt.scatter(f_test_y,forest_predictions,color='red')
plt.xlabel('Actuals')
plt.ylabel('Predictions')
plt.title('Actuals vs Predictions')
plt.show()
xgb_pipeline = make_pipeline(XGBRegressor())
xgb_scores = cross_val_score(xgb_pipeline,X.as_matrix(),y.as_matrix(),scoring="neg_mean_absolute_error")
print("MAE %2f" %(-1 * xgb_scores.mean()) )
train_X,test_X,train_y,test_y = train_test_split(X.as_matrix(),y.as_matrix(),test_size=0.25)
xgb_model = XGBRegressor()
xgb_model.fit(train_X,train_y,verbose=False)
xgb_preds = xgb_model.predict(test_X)
print("MAE: " + str(mean_absolute_error(xgb_preds,test_y)))
print("Accuracy:",xgb_model.score(test_X,test_y))
xgb_model_II = XGBRegressor(n_estimators=1000,learning_rat=0.05)
xgb_model_II.fit(train_X,train_y,early_stopping_rounds=5,
             eval_set=[(test_X,test_y)],verbose=False)
xgb_preds = xgb_model_II.predict(test_X)
print("MAE: " + str(mean_absolute_error(xgb_preds,test_y)))
print("Accuracy:",xgb_model_II.score(test_X,test_y))
plt.scatter(test_y,xgb_preds,color='blue')
plt.xlabel('Actuals')
plt.ylabel('Predictions')
plt.title('Actuals vs Predictions')
plt.show()
cols = ['Length','Diameter','Height']
ab_par_model = GradientBoostingRegressor()
ab_par_model.fit(X,y)
my_plots = plot_partial_dependence(ab_par_model,
                                  features=[0,2],
                                  X=X,
                                  feature_names=cols,
                                   grid_resolution=10)