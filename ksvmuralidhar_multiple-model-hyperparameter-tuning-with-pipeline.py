import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import KNNImputer

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

%matplotlib inline
train_df = pd.read_csv("../input/mobile-price-classification/train.csv")
train_df.shape
train_df.head().T
train_df.dtypes
train_df.describe(include="all").T
train_df.nunique()
for i,col in enumerate(train_df.columns):

    print("-"*10)

    print(col)

    print("-"*10)

    print(train_df[col].unique())
label = "price_range"
#storing categorical features

cat_features = train_df.columns[[1,3,5,17,18,19]]

print(cat_features)
#storing numerical features

num_features = train_df.columns[(train_df.columns.isin(cat_features)==False) & (train_df.columns!=label)]

print(num_features)
num_features_with_missing = num_features[train_df[num_features].min()==0]

print(num_features_with_missing)
num_features_with_missing = num_features_with_missing[2:]

print(num_features_with_missing)
#marking the missing values in the above columns

for col in num_features_with_missing:

    train_df.loc[train_df[col]==0,col] = np.nan
#Computing the % of missing values per column

train_df.isnull().mean()*100
len(train_df.loc[(train_df["pc"]==0) & (train_df["fc"]!=0)])
len(train_df.loc[(train_df["four_g"]==1) & (train_df["pc"]==0)])
len(train_df.loc[(train_df["touch_screen"]==1) & (train_df["pc"]==0)])
len(train_df.loc[(train_df["wifi"]==1) & (train_df["pc"]==0)])
fig,ax = plt.subplots(7,2,figsize=(13,40))

i=r=c=0

for tgt,feat in zip([label]*len(num_features),num_features):

    if (i%2==0) & (i>0):

        r+=1

        c=0

    sns.boxplot(x=tgt,y=feat,data=train_df,ax=ax[r,c])

    medians = train_df[[tgt,feat]].groupby(tgt).median().reset_index()

    sns.lineplot(x=tgt,y=feat,data=medians,ax=ax[r,c],linewidth=5,color="black")

    ax[r,c].set_title("price_range vs "+feat)

    i+=1

    c+=1



plt.show()

    
for tgt,feat in zip([label]*len(cat_features),cat_features):

    cross_tab = pd.crosstab(index=train_df[feat],columns=train_df[tgt],normalize="columns")*100

    cross_tab.T.plot(kind="barh",stacked=True,figsize=(11,4),)

    plt.title("price_range vs "+feat)

    plt.xlabel("% of mobiles")

    plt.show()

    
fig = plt.figure(figsize=(15,15))

sns.heatmap(train_df[num_features].corr(),annot=True,fmt=".2f",mask=np.triu(train_df[num_features].corr()),cbar=False);
X_train,X_test,y_train,y_test = train_test_split(train_df.iloc[:,:-1],train_df.iloc[:,-1],test_size=0.2,random_state=11)
y_train.value_counts()
classifier_pipe = Pipeline(steps=(["knn_imputer",KNNImputer()],["classifier",DecisionTreeClassifier(random_state=11)]))





classifier_param_grid = [{

                      "classifier":[DecisionTreeClassifier(random_state=11)],

                      #"knn_imputer__n_neighbors":np.arange(3,22,2), #preprocessing hyperparameter tuning can also be done

                      "classifier__criterion":["gini","entropy"],

                      "classifier__max_depth":np.arange(10,21,2),

                      #"classifier__min_samples_split":np.arange(2,21,3),

                      #"classifier__min_samples_leaf":np.arange(1,10,2)

                     },



                     {

                      "classifier":[RandomForestClassifier(random_state=11)],

                      #"knn_imputer__n_neighbors":np.arange(3,22,2),

                      "classifier__criterion":["gini","entropy"],

                      "classifier__n_estimators":np.arange(50,1200,500),

                      #"classifier__min_samples_split":np.arange(2,21,3),

                      #"classifier__min_samples_leaf":np.arange(1,10,2)

                     }]





grid_cv = GridSearchCV(estimator=classifier_pipe,param_grid=classifier_param_grid,scoring="accuracy",cv=5)
grid_cv.fit(X_train,y_train)

print(f"BEST SCORE: {grid_cv.best_score_}")

final_classifier_1 = grid_cv.best_estimator_

print(f"VALIDATION_SCORE: {final_classifier_1.score(X_test,y_test)}")

print(f"\n\nBEST CLASSIFIER: {final_classifier_1}")
grid_cv.fit(X_train.drop(columns=["fc","pc"]),y_train)

print(f"BEST SCORE: {grid_cv.best_score_}")

final_classifier_2 = grid_cv.best_estimator_

print(f'VALIDATION SCORE: {final_classifier_2.score(X_test.drop(columns=["fc","pc"]),y_test)}')

print(f"\n\nBEST CLASSIFIER: {final_classifier_2}")
grid_cv.fit(X_train.drop(columns=["pc"]),y_train)

print(f"BEST SCORE: {grid_cv.best_score_}")

final_classifier_3 = grid_cv.best_estimator_

print(f'VALIDATION SCORE: {final_classifier_3.score(X_test.drop(columns=["pc"]),y_test)}')

print(f"\n\nBEST CLASSIFIER: {final_classifier_3}")
grid_cv.fit(X_train.drop(columns=["fc"]),y_train)

print(f"BEST SCORE: {grid_cv.best_score_}")

final_classifier_4 = grid_cv.best_estimator_

print(f'VALIDATION SCORE: {final_classifier_4.score(X_test.drop(columns=["fc"]),y_test)}')

print(f"\n\nBEST CLASSIFIER: {final_classifier_4}")
FINAL_MODEL = final_classifier_2
FINAL_MODEL
X_test.drop(columns=["pc","fc"],inplace=True)
FINAL_MODEL.score(X_test,y_test)
pred = FINAL_MODEL.predict(X_test)
prediction_df = pd.DataFrame({"Actual":y_test,"Prediction":pred})
prediction_df.head()
print(classification_report(y_test,pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True,cbar=False)

plt.xlabel("Prediction")

plt.ylabel("Actual");