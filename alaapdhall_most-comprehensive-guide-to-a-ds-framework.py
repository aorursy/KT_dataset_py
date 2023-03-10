# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling

import pandas as pd

import numpy as np

import random



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, make_scorer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from xgboost import XGBClassifier, plot_importance as plot_importance_xgb



# Deep Learning



import torch 

import torchvision as tv 
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]

train_df
train_df.columns
train_df.info()

print('_'*40)

test_df.info()
train_df.describe(include=['O'])
train_df.Sex.value_counts()
test_df.Sex.value_counts()
cat_cols = ['Pclass','Parch','Sex','SibSp','Cabin','Embarked']

for i in cat_cols:

    print(train_df[[i, 'Survived']].groupby([i], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print('-'*50)

    print('\n')

    

    

# Pclass seems like an important feature to predict survival as PClass 1 means more survived!!

# Female survived more!!

# SibSp and Parch These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# check Reltn btw cont and Class



for i in ['Age','Fare']:

    g = sns.FacetGrid(train_df, col='Survived')

    g.map(plt.hist, i , bins=20)
sns.pairplot(train_df)
fig, ax = plt.subplots(figsize=(10, 10))

corr = train_df.corr()

sns.heatmap(corr, linewidths=.5, cbar_kws={"shrink": .5},annot_kws={'fontsize':12 },annot=True,)
fig, ax = plt.subplots(figsize=(12, 7))



sns.barplot(x = train_df['Pclass'],y = train_df['Fare'])

plt.title('# P1 vs Fare', size=20)

plt.xlabel('Pclass', size=15)

plt.ylabel('Fare', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



# a Clear -ve correlation... but does this correlation matters? because Pclass is a category not numeric!!
fig, ax = plt.subplots(figsize=(17, 7))

sns.distplot(train_df['Age'].dropna())
fig, ax = plt.subplots(figsize=(17, 7))

sns.distplot(train_df['Fare'].dropna())
import pandas_profiling # library for automatic EDA

report = pandas_profiling.ProfileReport(train_df)
display(report)
!pip install autoviz 



from autoviz.AutoViz_Class import AutoViz_Class

from IPython.display import display # display from IPython.display



AV = AutoViz_Class()
# Let's now visualize the plots generated by AutoViz.

report_2 = AV.AutoViz("/kaggle/input/titanic/train.csv")
print("{} \nNan values found".format(train_df.isna().sum()))

print("{} \nNan values found".format(test_df.isna().sum()))



# train_df.dropna(inplace=True) # drop na
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]

train_df
train_df = train_df.drop(['PassengerId','Cabin','Ticket'], axis=1)

test_df = test_df.drop(['PassengerId','Cabin','Ticket'], axis=1)



#complete missing age with median

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Age'].fillna(test_df['Age'].median(), inplace = True)



#complete embarked with mode

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)



test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)



train_df
print("{} \nNan values found".format(train_df.isna().sum()))

print("{} \nNan values found".format(test_df.isna().sum()))



# train_df.dropna(inplace=True) # drop na
from scipy.stats import pearsonr

    



# get if a variable is continous, 

# do some sanity check so onehotencoders are not affected

def get_continous(df):

    cont_cols = []

    for col in df.columns:

        try:

            df[col] = list(map(int,df[col])) # convert to int, eg '123' --> 123 and 'abc'--> 'ValueError'. Win win!!

            if set(df[col].unique())!={0,1}: # skip onehot values

                cont_cols.append(col)

        except ValueError:

            pass  

    return cont_cols



# we can use df.corr(method='pearson') to get pearson relation but this compromise the flexibility

# and in order to remove we had to pass though data again, 

# rather we calculate corr column by column and remove if feels necessary, flexible and faster



def find_pearson_cor(df):

    cont_cols = get_continous(df) # get cont values to reduce number of col search

    if cont_cols is not None:

        cnt = 0

        should_drop=[]

        for idx,i in enumerate(cont_cols):

            for j in cont_cols[idx+1:]: # go through columns and find corr

                corr, _ = pearsonr(df[i],df[j]) # use scipy

                print(i,"has corr value =",corr,"with",j)

                if corr > 0.85: 

                    cnt+=1

                    random_drop = random.randint(0,1) # randomly select 1 column with high corr value 

                    if random_drop==1:

                        should_drop.append(j)

#                         df.drop(j,axis=1,inplace = True) # drop it

                        print(j,"Should be dropped with corr = ",corr)

                    else:

                        should_drop.append(i)

#                         df.drop(i,axis=1,inplace = True) # drop it

                        print(i,"Should be dropped with corr = ",corr)

        if cnt==0:

            print("No columns are highly Correlated!!")

    else:

        print("no continous columns")



    return should_drop # return df

find_pearson_cor(train_df)
# get if a variable is continous, 

# do some sanity check so onehotencoders are not affected

def get_continous(df):

    cont_cols = []

    for col in df.columns:

        try:

            df[col] = list(map(int,df[col])) # convert to int, eg '123' --> 123 and 'abc'--> 'ValueError'. Win win!!

            if set(df[col].unique())!={0,1}: # skip onehot values

                cont_cols.append(col)

        except ValueError:

            pass

  

    return cont_cols



# 2 common ways to detect outliers, we will use IQR method cz I feel it works better in practical scenerios and is more flexible.

def detect_outliers(df,OUT_THRES=7):

    df_len = len(df)

    cont_cols = get_continous(df) # get cont values



    print("Variable with continues values",cont_cols)

  

    if cont_cols is not None:

        for i in cont_cols:

            # find outliers for every column thatis continous, this is why IQR is better than z as ir gives more flexibility

            Q1 = df[i].quantile(0.20)

            Q3 = df[i].quantile(0.80)

            IQR = Q3 - Q1

            # get bounds

            Lower_Bound = Q1-1.5*IQR

            Upper_Bound = Q3+1.5*IQR



            # get actual outlier values

            df_u = df[i]<Upper_Bound

            df_l = df[i]>Lower_Bound



            # Sanity check

            # If number of outliers is less than the given threshold, only remove then, otherwise let it be, it may be occuring naturally

            if df_len-sum(df_u)<=OUT_THRES:

                df = df[df_u]



            if df_len-sum(df_l)<=OUT_THRES:

                df = df[df_l]



            print("For {} removed {} outliers".format(i,df_len-len(df)))

            df_len = len(df)

    else:

        print('No Continous variables found')



    return df
print("Final df\n",detect_outliers(train_df))
# Use SimpleImputer?



# from sklearn.impute import SimpleImputer



# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# imputer.fit(X[:, 3:4])

# X[:, 3:4] = imputer.transform(X[:, 3:4])
train_df['Sex'] = train_df['Sex'].map({'male':0,'female':1})

test_df['Sex'] = test_df['Sex'].map({'male':0,'female':1})



train_df
# Encoding the Dependent Variable if needed



# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# y = le.fit_transform(y)

# print(y)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

test_df['AgeBand'] = pd.cut(test_df['Age'], 5)



train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_df['FareBin'] = pd.qcut(train_df['Fare'], 4)

test_df['FareBin'] = pd.qcut(test_df['Fare'], 4)



train_df[['FareBin', 'Survived']].groupby(['FareBin'], as_index=False).mean().sort_values(by='FareBin', ascending=True)
for dataset in [train_df,test_df]:



    """

    you could divide Age into bins like this based on the above observed formula, 

    but  using LabelEncoder is easier and works better thus, I'll follow that

    """

#     dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

#     dataset.loc[ dataset['Age'] > 64, 'Age']

    

    

    # Creating a categorical variable for Family Sizes

    dataset['FamilySize'] = ''

    dataset['FamilySize'].loc[(dataset['SibSp'] <= 2)] = 0

    dataset['FamilySize'].loc[(dataset['SibSp'] > 2) & (dataset['SibSp'] <= 5 )] = 1

    dataset['FamilySize'].loc[(dataset['SibSp'] > 5)] = 2 





    # Creating a categorical variable to tell if the passenger is alone

    dataset['IsAlone'] = ''

    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) > 0)] = 1

    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) == 0)] = 0

    

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    

    # take only top 10 titles

    title_names = (dataset['Title'].value_counts() < 10) #this will create a true false series with title name as index



    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    print(dataset['Title'].value_counts())

    print("-"*10)

    



train_df.head()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label = LabelEncoder()



for dataset in [train_df,test_df]:

    dataset['AgeBand_Code'] = label.fit_transform(dataset['AgeBand'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    

    #Drop Columns

    dataset.drop(['Name','AgeBand','FareBin'], axis=1,inplace=True)



    

train_df
#define x and y variables for dummy features original

train_dummy = pd.get_dummies(train_df)

test_dummy = pd.get_dummies(test_df)



train_dummy
y = train_dummy['Survived']

train_dummy.drop(['IsAlone_1','FamilySize_2','Survived'],axis=1,inplace=True)

test_dummy.drop(['IsAlone_1','FamilySize_2'],axis=1,inplace=True)
"""

You could do OneHotEncoding with Column Transfer as Well!!

"""



# OneHotEncode



# # Encoding categorical data

# # Encoding the Independent Variable

# from sklearn.compose import ColumnTransformer

# from sklearn.preprocessing import OneHotEncoder



# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,6])], remainder='passthrough')

# X_encoded = np.array(ct.fit_transform(X_train))

# X_test_encoded =  np.array(ct.transform(X_test))

# X_encoded
train_dummy
from sklearn.feature_selection import SelectKBest, chi2



best_feature = SelectKBest(score_func= chi2, k = 'all')

best_feature = best_feature.fit(train_dummy.values , y)



col_scores = pd.DataFrame(best_feature.scores_)

col_names = pd.DataFrame(train_dummy.columns)



feature_score = pd.concat([col_names, col_scores], axis=1)

feature_score.columns = ['attribute', 'score']

feature_score
# Did you Notice Something?? 
fig, ax = plt.subplots(figsize=(17, 15))

corr = train_dummy.corr()

sns.heatmap(corr, linewidths=.5, cbar_kws={"shrink": .5},annot_kws={'fontsize':12 },annot=True,)
find_pearson_cor(train_dummy)
clean_data = pd.concat([train_dummy,pd.DataFrame({'Survived':y})],axis=1)

clean_data
# In here we keep only Original Values encoded

train_calc = ['Pclass','Sex','SibSp', 'Parch', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q','Embarked_S', 'Title_Master','Title_Misc','Title_Miss', 'Title_Mr','Title_Mrs'] 





# In this We keep all the featured values and remove correlated values as seen above in pearson corr heatmap

train_feat = ['Sex','Pclass', 'Embarked_C', 'Embarked_Q','Embarked_S','FamilySize_0','FamilySize_1','IsAlone_0', 'Title_Master','Title_Misc','Title_Miss', 'Title_Mr','Title_Mrs', 'FamilySize_0','FamilySize_1', 'AgeBand_Code', 'FareBin_Code']

#define x variables for original w/bin features to remove continuous variables



train_calc_df = clean_data[train_calc+['Survived']]

test_calc_df = test_dummy[train_calc]

train_calc_df
clean_feat_df = clean_data[train_feat+['Survived']]

clean_feat_df
clean_feat_df.info()
X_final = train_calc_df.drop(['Survived'],axis=1).values # for original features



X_final_feat = clean_feat_df.drop(['Survived'],axis=1).values # for new features



target = train_calc_df['Survived'].values
X_final.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_final = sc.fit_transform(X_final)

print(X_final.shape,X_final)



sc_feat = StandardScaler()

X_final_feat = sc_feat.fit_transform(X_final_feat)

print(X_final_feat.shape,X_final_feat)
X_train, X_test, y_train, y_test = train_test_split(X_final, target, test_size=0.10)

print(X_train.shape, X_test.shape)
X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(X_final_feat, target, test_size=0.10)

print(X_train_feat.shape, X_test_feat.shape)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_final, target)

acc_log = round(logreg.score(X_final, target) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_calc_df.columns.delete(-1))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_final, target)



acc_decision_tree = round(decision_tree.score(X_final, target) * 100, 2)

acc_decision_tree
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_final_feat, target)

acc_log = round(logreg.score(X_final_feat, target) * 100, 2)

acc_log
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_final_feat, target)



acc_decision_tree = round(decision_tree.score(X_final_feat, target) * 100, 2)

acc_decision_tree
# Let's create a clean predict method 

def predict_custom(classifier,x_test,y_test):

    

    predictions = classifier.predict(x_test)

    

    # get f1score,prec,recall,roc_auc, and accuracy

    f1 = f1_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, predictions)

    accuracy = accuracy_score(y_test, predictions)

    

    result_df = pd.DataFrame({'f1':[f1],'Precision':[precision],'Recall':[recall],'roc_auc_score':[roc_auc],'Accuracy':[accuracy]})

    

    print("\n\n#---------------- Test set results (Best Classifier) ----------------#\n")

    print("F1 score, Precision, Recall, ROC_AUC score, Accuracy:")

    return result_df
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "Random Forest": RandomForestClassifier(),

    "Gradient Boosting": GradientBoostingClassifier(),

    "DecisionTreeClassifier": DecisionTreeClassifier(),

    "XGB":XGBClassifier()   

}
# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    

    # scoring on final

    classifier.fit(X_final, target)

    

    training_score = cross_val_score(classifier, X_final, target, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

    print('-'*50)

    print('\n')

    

for key, classifier in classifiers.items(): 

    # scoring on splitted data!!1

    print("Scoring on",key)

    classifier.fit(X_train, y_train)

    print(predict_custom(classifier,X_test,y_test))

    print('-'*50)

    print('\n')

# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    

    # scoring on final

    classifier.fit(X_final_feat, target)

    

    training_score = cross_val_score(classifier, X_final_feat, target, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

    print('-'*50)

    print('\n')

    

for key, classifier in classifiers.items(): 

    # scoring on splitted data!!1

    print("Scoring on",key)

    classifier.fit(X_train_feat, y_train_feat)

    print(predict_custom(classifier,X_test_feat,y_test_feat))

    print('-'*50)

    print('\n')

# Best models are XGBoost, Logistic,RandomForest,GradientBoosting



# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'],

                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params,cv=5)

grid_log_reg.fit(X_final, target)

log_reg = grid_log_reg.best_estimator_

print(log_reg)



#XGBoost

xgboost_params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



grid_xgboost = GridSearchCV(XGBClassifier(), xgboost_params,cv=5)

grid_xgboost.fit(X_final, target)

xgboost_clf = grid_xgboost.best_estimator_

print(xgboost_clf)





# # Gradient Boosting Classifier



# grad_boost_parameters = {

#     "loss":["deviance"],

#     "learning_rate": [0.05, 0.1,  0.5],

#     "min_samples_split": np.linspace(0.5, 12),

#     "min_samples_leaf": np.linspace(0.5, 12),

#     "max_depth":[3,5,8],

#     "max_features":["log2","sqrt"],

#     "subsample":[0.5, 0.8, 1.0],

#     "n_estimators":[10,50,100]

#     }



# grid_gradBoost = GridSearchCV(GradientBoostingClassifier(), grad_boost_parameters,cv=5)

# grid_gradBoost.fit(X_final, target)

# grad_boost = grid_gradBoost.best_estimator_

# print(grad_boost)



# Random Forest Classifier

randF_params = { 

    'n_estimators': [50,100,200],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,6,8],

    'criterion' :['gini', 'entropy']

}



grid_tree = GridSearchCV(RandomForestClassifier(), randF_params,cv=5)

grid_tree.fit(X_final, target)



# tree best estimator

RandF_clf = grid_tree.best_estimator_

print(RandF_clf)
log_reg_score = cross_val_score(log_reg, X_final, target, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





xgboost_score = cross_val_score(xgboost_clf, X_final, target, cv=5)

print('XGBoost Cross Validation Score', round(xgboost_score.mean() * 100, 2).astype(str) + '%')



# grad_boost_score = cross_val_score(grad_boost, X_final, target, cv=5)

# print('GradientBoostingClassifier Cross Validation Score', round(grad_boost_score.mean() * 100, 2).astype(str) + '%')



randF_score = cross_val_score(RandF_clf,X_final, target, cv=5)

print('Random Forest Classifier Cross Validation Score', round(randF_score.mean() * 100, 2).astype(str) + '%')
Y_pred = log_reg.predict(test_calc_df.values)

Y_pred
submission = pd.DataFrame({

        "PassengerId": pd.read_csv('../input/titanic/test.csv')["PassengerId"],

        "Survived": Y_pred

    })



submission
submission.to_csv('../working/submission.csv', index=False)