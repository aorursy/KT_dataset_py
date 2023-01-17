!ls ../input/breastcancer-dataset/
import sys; print("Python", sys.version)

import numpy; print("NumPy", numpy.__version__)

import scipy; print("SciPy", scipy.__version__)

import sklearn; print("Scikit-Learn", sklearn.__version__)




%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as s

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn import svm

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from xgboost import XGBClassifier

import pickle
data = pd.read_csv('../input/breastcancer-dataset/data.csv')
data.shape
data.head()
df = data.drop('Unnamed: 32', axis=1)
df['diagnosis'].value_counts()
df.diagnosis = df.diagnosis.astype('category')
df.head()
X = df.drop(labels='diagnosis',axis=1)

Y= df['diagnosis']

col=X.columns
col
X.isnull().sum()
X.head()


# X = mms.transform(X)



df_norm = (X - X.mean()) / (X.max() - X.min())

df_norm = pd.concat([df_norm, Y], axis=1)
# X=pd.DataFrame(X)

# X.columns=col
df_norm.head()
#Explore correlations

plt.rcParams['figure.figsize']=(12,8)

s.set(font_scale=1.4)

s.heatmap(df.drop('diagnosis', axis=1).drop('id',axis=1).corr(), cmap='coolwarm')
plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='radius_mean',data=df, ax=ax1)

s.boxplot('diagnosis',y='texture_mean',data=df, ax=ax2)

s.boxplot('diagnosis',y='perimeter_mean',data=df, ax=ax3)

s.boxplot('diagnosis',y='area_mean',data=df, ax=ax4)

s.boxplot('diagnosis',y='smoothness_mean',data=df, ax=ax5)

f.tight_layout()



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='compactness_mean',data=df, ax=ax2)

s.boxplot('diagnosis',y='concavity_mean',data=df, ax=ax1)

s.boxplot('diagnosis',y='concave points_mean',data=df, ax=ax3)

s.boxplot('diagnosis',y='symmetry_mean',data=df, ax=ax4)

s.boxplot('diagnosis',y='fractal_dimension_mean',data=df, ax=ax5)    

f.tight_layout()
g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "radius_mean", hist=False, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "texture_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "perimeter_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "area_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "smoothness_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "compactness_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "concavity_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "concave points_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "symmetry_mean", hist=True, rug=True)



g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')

g.map(s.distplot, "fractal_dimension_mean", hist=True, rug=True)


plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='radius_se',data=df, ax=ax1, palette='cubehelix')

s.boxplot('diagnosis',y='texture_se',data=df, ax=ax2, palette='cubehelix')

s.boxplot('diagnosis',y='perimeter_se',data=df, ax=ax3, palette='cubehelix')

s.boxplot('diagnosis',y='area_se',data=df, ax=ax4, palette='cubehelix')

s.boxplot('diagnosis',y='smoothness_se',data=df, ax=ax5, palette='cubehelix')

f.tight_layout()



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='compactness_se',data=df, ax=ax2, palette='cubehelix')

s.boxplot('diagnosis',y='concavity_se',data=df, ax=ax1, palette='cubehelix')

s.boxplot('diagnosis',y='concave points_se',data=df, ax=ax3, palette='cubehelix')

s.boxplot('diagnosis',y='symmetry_se',data=df, ax=ax4, palette='cubehelix')

s.boxplot('diagnosis',y='fractal_dimension_se',data=df, ax=ax5, palette='cubehelix')    

f.tight_layout()



plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='radius_worst',data=df, ax=ax1, palette='coolwarm')

s.boxplot('diagnosis',y='texture_worst',data=df, ax=ax2, palette='coolwarm')

s.boxplot('diagnosis',y='perimeter_worst',data=df, ax=ax3, palette='coolwarm')

s.boxplot('diagnosis',y='area_worst',data=df, ax=ax4, palette='coolwarm')

s.boxplot('diagnosis',y='smoothness_worst',data=df, ax=ax5, palette='coolwarm')

f.tight_layout()



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='compactness_worst',data=df, ax=ax2, palette='coolwarm')

s.boxplot('diagnosis',y='concavity_worst',data=df, ax=ax1, palette='coolwarm')

s.boxplot('diagnosis',y='concave points_worst',data=df, ax=ax3, palette='coolwarm')

s.boxplot('diagnosis',y='symmetry_worst',data=df, ax=ax4, palette='coolwarm')

s.boxplot('diagnosis',y='fractal_dimension_worst',data=df, ax=ax5, palette='coolwarm')    

f.tight_layout()
Y.tail()
X_norm = df_norm.drop(labels='diagnosis',axis=1)

Y_norm= df_norm['diagnosis']

col=X_norm.columns



le = LabelEncoder()

le.fit(Y_norm)
X
Y_norm=le.transform(Y_norm)

Y_norm=pd.DataFrame(Y_norm)

Y_norm.tail()
columns = df_norm.columns
# Functionalize model fittting



def FitModel(X,Y,algo_name,algorithm,gridSearchParams,cv):

    np.random.seed(10)

    

    x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)

    

    grid = GridSearchCV(

        estimator=algorithm,

        param_grid=gridSearchParams,

        cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

    

    

    grid_result = grid.fit(x_train, y_train)

    best_params = grid_result.best_params_

    pred = grid_result.predict(x_test)

    cm = confusion_matrix(y_test, pred)

   # metrics =grid_result.gr

    print(pred)

    pickle.dump(grid_result,open(algo_name,'wb'))

   

    print('Best Params :',best_params)

    print('Classification Report :',classification_report(y_test,pred))

    print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))

    print('Confusion Matrix : \n', cm)
param ={

            'C': [0.1, 1, 100, 1000],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        }

FitModel(X_norm,Y_norm,'SVC_norm',SVC(),param,cv=5)
param ={

            'n_estimators': [100, 500, 1000, 2000],

           

        }

FitModel(X_norm,Y_norm,'Random Forest',RandomForestClassifier(),param,cv=10)
np.random.seed(10)



x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)





forest = RandomForestClassifier(n_estimators=1000)

fit = forest.fit(x_train, y_train)

accuracy = fit.score(x_test, y_test)

predict = fit.predict(x_test)

cmatrix = confusion_matrix(y_test, predict)



#--------------------------------------------------------------------------------------#

# Perform k fold cross-validation





print ('Accuracy of Random Forest: %s' % "{0:.2%}".format(accuracy))

#%%Feature importances

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]



print("Feature ranking:")

for f in range(X.shape[1]):

    print("feature %s (%f)" % (list(X)[f], importances[indices[f]]))
feat_imp = pd.DataFrame({'Feature':list(X),

                        'Gini importance':importances[indices]})

plt.rcParams['figure.figsize']=(12,12)

s.set_style('whitegrid')

ax = s.barplot(x='Gini importance', y='Feature', data=feat_imp)

ax.set(xlabel='Gini Importance')

plt.show()
param ={

            'n_estimators': [100, 500, 1000, 2000],

           

        }

FitModel(X_norm,Y_norm,'XGBoost_norm',XGBClassifier(),param,cv=5)
df_norm.head()
from imblearn.over_sampling import SMOTE
sm =SMOTE(random_state=42)

X_res , Y_res = sm.fit_resample(X_norm,Y_norm)
sm.fit_sample(X_norm,Y_norm)
param ={

            'n_estimators': [100, 500, 1000, 2000],

           }

FitModel(X_res,Y_res,'Random Forest',RandomForestClassifier(),param,cv=10)
param ={

            'C': [0.1, 1, 100, 1000],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        }

FitModel(X_res,Y_res,'SVC',SVC(),param,cv=5)
param ={

            'n_estimators': [100, 500, 1000, 2000],

           

        }

FitModel(X_res,Y_res,'XGBoost',XGBClassifier(),param,cv=10)
feat_imp.index = feat_imp.Feature
feat_to_keep = feat_imp.iloc[1:15].index


type(feat_to_keep),feat_to_keep

X_res= pd.DataFrame(X_res)

Y_res = pd.DataFrame(Y_res)

X_res.columns = X_norm.columns



param ={

            'n_estimators': [100, 500, 1000, 2000],

           

        }

FitModel(X_res[feat_to_keep],Y_res,'Random Forest',RandomForestClassifier(),param,cv=10)
loaded_model = pickle.load(open("XGBoost_norm","rb"))


pred1 = loaded_model.predict(x_test)

loaded_model.best_params_
X_res.columns
x_train1,x_test1,y_train1,y_test1 = train_test_split(X_res,Y_res, test_size = 0.2)

    

train=pd.concat([y_train1,x_train1],axis=1)

train.to_csv('./train.csv',index=False,header=False)

    



y_train.to_csv('./Y-train.csv')



    

test=pd.concat([y_test1,x_test1],axis=1)

test.to_csv('./test.csv',index=False,header=False)

    

y_test.to_csv('./Y-test.csv')

    