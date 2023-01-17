import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")



train.head()

#test.head()
test.info()
train.isnull().head(4)

missing_count = train.isnull().sum()

missing_count[missing_count > 0]



test.isnull().head(4)

missing_count = test.isnull().sum()

missing_count[missing_count > 0]
train_dtype_nunique = pd.concat([train.dtypes, train.nunique()],axis=1)

train_dtype_nunique.columns = ["dtype","unique"]

train_dtype_nunique
train.dtypes

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
X = train.iloc[:,0:10]  #independent columns

y = train.iloc[:,-1]    #target column i.e price range
bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)
trainscores = pd.DataFrame(fit.scores_)

traincolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([traincolumns,trainscores],axis=1)
featureScores.columns = ['feature','class']

print(featureScores.nlargest(10,'class'))
# Compute the correlation matrix

corr = train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
features = ['chem_1','chem_4', 'chem_6', 'attribute', 'chem_7']

target = 'class'

X = train[features]

y = train[target]
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier().fit(X,y)
from sklearn.metrics import accuracy_score  #Find out what is accuracy_score





rf_prediction = clf.predict(test[features])

rf_prediction

#acc_rf = accuracy_score(rf_prediction,train[target])*100



#print("Accuracy score of random forest: {}".format(acc_rf))
submission = pd.DataFrame({'id': test['id'], 'class': rf_prediction })

submission.to_csv("submission10.csv", index=False)
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier().fit(X,y)
des_prediction = clf1.predict(test[features])

des_prediction
submission = pd.DataFrame({'id': test['id'], 'class': rf_prediction })

submission.to_csv("submission18.csv", index=False)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn import datasets

from sklearn import metrics

svc=SVC(probability=True, kernel='linear')



abc =AdaBoostClassifier(n_estimators=100, base_estimator=svc,learning_rate=1)

model = abc.fit(X, y)
ada_pred = model.predict(test[features])

ada_pred
submission = pd.DataFrame({'id': test['id'], 'class': rf_prediction })

submission.to_csv("submission19.csv", index=False)
from sklearn.ensemble import AdaBoostClassifier

from sklearn import datasets
abc = AdaBoostClassifier(n_estimators=400, learning_rate=1)

# Train Adaboost Classifer

model = abc.fit(X, y)



#Predict the response for test dataset

y_pred = model.predict(test[features])

y_pred
submission = pd.DataFrame({'id': test['id'], 'class': rf_prediction })

submission.to_csv("submission20.csv", index=False)


X = train.iloc[:,0:10]  #independent columns

y = train.iloc[:,-1]    #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()


X = train.iloc[:,0:20]  #independent columns

y = train.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
from sklearn.datasets import load_boston

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso



#Adding constant column of ones, mandatory for sm.OLS model

X_1 = sm.add_constant(X)

#Fitting sm.OLS model

model = sm.OLS(y,X_1).fit()

model.pvalues
#Backward Elimination

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

    p= []

    X_1 = X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)