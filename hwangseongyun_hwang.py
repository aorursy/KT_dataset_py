import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib.font_manager as fm
import warnings
from imblearn.combine import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import TomekLinks
font_path = 'C:/windows/Fonts/HANDotum.ttf'
font_prop = fm.FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)
xdf = pd.read_csv('mem_data.csv',encoding='cp949')
xdf.head(1)
ydf = pd.read_csv('mem_transaction.csv',encoding='cp949')
ydf.head(1)
zdf = pd.read_csv('store_info.csv',encoding='cp949')
zdf.head(1)
ord_df = pd.merge(xdf,ydf,how='left',on="MEM_ID",sort=False)
ord_df.head() #on="MEM_ID"
ord_df1 = pd.merge(ord_df,zdf,how='left',on='STORE_ID',sort=False)
ord_df1.head(1) #on='STORE_ID'
mdf = ord_df1.loc[ord_df1["GENDER"].isin(['M','F'])]
mdf.head(3)
mdf['GENDER'].value_counts()
f,ax=plt.subplots(1,2,figsize=(18,8))
mdf['GENDER'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('1', fontproperties=font_prop)
ax[0].set_ylabel('')
sns.countplot('GENDER',data=mdf,ax=ax[1])
ax[1].set_title('2', fontproperties=font_prop)
plt.show()
mdf.info()
obj1=['GENDER','BIRTH_DT','BIRTH_SL','ZIP_CD','RGST_DT','LAST_VST_DT','SMS','MEMP_STY','MEMP_DT','MEMP_TP']
mdf[obj1] = mdf[obj1].apply(lambda x: x.astype('category').cat.codes)
mdf = mdf.fillna(0)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
dfX = mdf.drop(['GENDER'],axis=1)
dfy = mdf['GENDER']
X_train, X_test, y_train, y_test = train_test_split(
     dfX,dfy,random_state=0)
tree3 = DecisionTreeClassifier(max_depth=6, random_state=0)

tree3.fit(X_train, y_train)
y_pred3 = tree3.predict(X_test)

print(classification_report(y_test, y_pred3))
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=200, random_state=0)
gbm.fit(X_train, y_train).score(X_test, y_test)
logreg = LogisticRegression()
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
voting = VotingClassifier(
    estimators = [('logreg', logreg), ('tree', tree), ('knn', knn)],
    voting = 'hard')
from sklearn.metrics import accuracy_score
for clf in (logreg, tree, knn, voting) :
    clf.fit(X_train, y_train)
    print(clf.__class__.__name__, 
          accuracy_score(y_test, clf.predict(X_test)))
ndf = ord_df1.loc[ord_df1["GENDER"] == "UNKNOWN"]
ndf.head(3)
obj2=['GENDER','BIRTH_DT','BIRTH_SL','ZIP_CD','RGST_DT','LAST_VST_DT','SMS','MEMP_STY','MEMP_DT','MEMP_TP']
ndf[obj2] = ndf[obj2].apply(lambda x: x.astype('category').cat.codes)
ndf = ndf.fillna(0)
ndf = ndf.drop(columns=['GENDER'])
ndf['GENDER'] = tree3.predict_proba(ndf.values)[:,1]
ndf.head()
df = ndf[['MEM_ID', 'GENDER']]
df.head(100)
df = df.groupby('MEM_ID').head(1)
df.to_csv('out_data_2.csv', index=False)
print('COMPLETE')
df.dtypes


