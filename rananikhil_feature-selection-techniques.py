from PIL import Image
Image.open('roosevelt.jpg')
Image.open('forest_area.PNG')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as st
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
df=pd.read_csv('covertype.csv')
df.head(2)
plt.figure(figsize=(6,6))
df['class'].value_counts(normalize=True).plot.pie(autopct='%1.f%%')
plt.title('Distribution of classes')
plt.show()
print("Any missing sample in set:",df.isnull().values.any())
print('Number of records in dataset are {} and Features are {}.'.format(*df.shape))
anova_results={}
num_cols=df.columns.to_list()[:10]

for i in num_cols:
    
    d7=df[df['class']==7][i]
    d6=df[df['class']==6][i]
    d5=df[df['class']==5][i]
    d4=df[df['class']==4][i]
    d3=df[df['class']==3][i]
    d2=df[df['class']==2][i]
    d1=df[df['class']==1][i]

    static,p_value=st.f_oneway(d1,d2,d3,d4,d5,d6,d7)
    anova_results[i]=[static,p_value]
    
df_anova=pd.DataFrame(anova_results).T
df_anova=df_anova.rename(columns={0:'F_statistic',1:'p_value'})
df_anova=df_anova.sort_values(by=['p_value','F_statistic'],ascending=[False,False])
df_anova['Significant']=df_anova['p_value'].apply(lambda x : 'True' if x<0.05  else 'False')
df_anova
X=df.drop('class',1)
Y=df['class']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y)

cat_cols=X_train.columns.to_list()[10:-2]
not_satisfied=[]
chi2_results={}

for _ in cat_cols:
    cross_table=pd.crosstab(X_train[_],Y_train)
        
    if np.any(np.sum(cross_table,1)<5):
        not_satisfied.append(_)
    elif len(cross_table)==1:
        not_satisfied.append(_)
        
    else:
        stat,p_value,dof=(st.chi2_contingency(cross_table)[0:3])
        chi2_results[_]=[stat,p_value,dof]    
            
            
print(f'The column(s) not satisfied for chi2 test {not_satisfied}')

df_chi2=pd.DataFrame(chi2_results).T
df_chi2=df_chi2.rename(columns={0:'chi2_statistic',1:'p_value',2:'dof'})
df_chi2=df_chi2.sort_values(by=['p_value','chi2_statistic'],ascending=[False,False])
df_chi2['Significant']=df_chi2['p_value'].apply(lambda x : 'True' if x<0.05  else 'False')
df_chi2
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=10)
fit = bestfeatures.fit(X_train,Y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Columns','Score']  
print(featureScores.nlargest(10,'Score')) 
signi_cols=list(featureScores.nlargest(10,'Score')['Columns'].values)

X_train_s=X_train[signi_cols]
Y_train_s=Y_train.copy()
X_test_s=X_test[signi_cols]
Y_test_s=Y_test.copy()

V_values=[vif(sm.add_constant(X_train_s).values,i) for i in range(sm.add_constant(X_train_s).shape[1])]
l=list(zip(sm.add_constant(X_train_s).columns,V_values))
l.sort(key=lambda x : x[1],reverse=True)
df_vif=pd.DataFrame(l,columns=['Feature','vif_value'])
df_vif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

#X=df.drop('class', axis=1)
Y=df['class']

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=50), max_features=54)
embeded_rf_selector.fit(X, Y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)),'features are significant enough according to RandomForestClassifier. ')
print('\n',embeded_rf_feature)
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=54)
embeded_lgb_selector.fit(X, Y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)),'features are significant enough according to lightGBM')
print('\n',embeded_lgb_feature)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), \
                   n_features_to_select=10, step=10, verbose=5)
rfe_selector.fit(X, Y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print('\n',str(len(rfe_feature)), 'most significant features according to RFE are ')
print(rfe_feature)