import pandas as pd

import numpy as np

from scipy.stats import chi2_contingency 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve

from imblearn.over_sampling import SMOTE 

from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve, plot_precision_recall_curve
src = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
src.head()
src.dtypes
src.isnull().mean()
src.shape
src["DEATH_EVENT"].value_counts()
src["DEATH_EVENT"].value_counts().plot(kind="bar")

plt.title("Class Distribution");
# Storing categorical and numerical features names in different Series

cat_features = ["anaemia","diabetes","high_blood_pressure","sex","smoking","DEATH_EVENT"]

num_features = pd.Series(src.columns)

num_features = num_features[~num_features.isin(cat_features)]
for i in cat_features:

    ct = pd.crosstab(columns=src[i],index=src["DEATH_EVENT"])

    stat, p, dof, expected = chi2_contingency(ct) 

    print(f"\n{'-'*len(f'CROSSTAB BETWEEN {i.upper()} & DEATH_EVENT')}")

    print(f"CROSSTAB BETWEEN {i.upper()} & DEATH_EVENT")

    print(f"{'-'*len(f'CROSSTAB BETWEEN {i.upper()} & DEATH_EVENT')}")

    print(ct)

    print(f"\nH0: THERE IS NO RELATIONSHIP BETWEEN DEATH_EVENT & {i.upper()}\nH1: THERE IS RELATIONSHIP BETWEEN DEATH_EVENT & {i.upper()}")

    print(f"\nP-VALUE: {np.round(p,2)}")

    print("REJECT H0" if p<0.05 else "FAILED TO REJECT H0")
r = c = 0

fig,ax = plt.subplots(3,2,figsize=(14,12))

for n,i in enumerate(cat_features[:-1]):

    ct = pd.crosstab(columns=src[i],index=src["DEATH_EVENT"],normalize="columns")

    ct.T.plot(kind="bar",stacked=True,color=["green","red"],ax=ax[r,c])

    ax[r,c].set_ylabel("% of observations")

    c+=1

    if (n+1)%2==0:

        r+=1

        c=0

ax[r,c].axis("off")

plt.show()
r = c = 0

fig,ax = plt.subplots(4,2,figsize=(14,25))

for n,i in enumerate(num_features):

    sns.boxplot(x="DEATH_EVENT",y=i,data=src,ax=ax[r,c])

    ax[r,c].set_title(i.upper()+" by "+"DEATH_EVENT")

    c+=1

    if (n+1)%2==0:

        r+=1

        c=0

ax[r,c].axis("off")

plt.show()
fig = plt.figure(figsize=(8,6))

sns.heatmap(src[num_features].corr(),annot=True,fmt=".2f",mask=np.triu(src[num_features].corr()),cbar=False)

plt.show()
src[num_features].hist(figsize=(14,14))

plt.show();
X = src.iloc[:,:-1]

y = src.iloc[:,-1]
rf = RandomForestClassifier(n_estimators=5000,random_state=11)

rf.fit(X,y)

feat_imp = pd.DataFrame(rf.feature_importances_)

feat_imp.index = pd.Series(src.iloc[:,:-1].columns)

feat_imp = (feat_imp*100).copy().sort_values(by=0,ascending=False)

feat_imp = feat_imp.reset_index()

feat_imp.columns = ["Feature","Importance_score"]



fig = plt.figure(figsize=(6,10))

sns.scatterplot(data=feat_imp,x=5,y=np.linspace(100,0,12),size="Importance_score",sizes=(200,2000),legend=False)

for i,feat,imp in zip(np.linspace(100,0,12),feat_imp["Feature"],feat_imp["Importance_score"]):

    plt.text(x=5.05,y=i-1,s=feat)

    plt.text(x=4.89,y=i-1,s=np.round(imp,2))

plt.axis("off")

plt.title("Feature Importance")

plt.show()
for var in np.arange(feat_imp.shape[0],6,-1):

    X_new = X[feat_imp.iloc[:var,0]].copy()

    X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

    final_rf = RandomForestClassifier(random_state=11)

    gscv = GridSearchCV(estimator=final_rf,param_grid={

        "n_estimators":[100,500,1000,5000],

        "criterion":["gini","entropy"]

    },cv=5,n_jobs=-1,scoring="f1_weighted")



    gscv.fit(X_train,y_train)

    print(str(var)+" variables:  "+str(gscv.best_estimator_)+"  F1 score: "+str(gscv.best_score_))
X_new = X[feat_imp.iloc[:8,0]].copy()

X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

final_rf = RandomForestClassifier(random_state=11)

gscv = GridSearchCV(estimator=final_rf,param_grid={

    "n_estimators":[100,500,1000,5000],

    "criterion":["gini","entropy"]

},cv=5,n_jobs=-1,scoring="f1_weighted")



gscv.fit(X_train,y_train)

FINAL_MODEL_NO_SMOTE = gscv.best_estimator_
FINAL_MODEL_NO_SMOTE.score(X_train,y_train)
train_pred = FINAL_MODEL_NO_SMOTE.predict(X_train)

print(classification_report(y_train,train_pred))
FINAL_MODEL_NO_SMOTE.score(X_test,y_test)
X_new = X.copy()

X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

smote = SMOTE(random_state = 11) 

X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)





X_train.insert(12,"category","NO_SMOTE")

X_train_smote.insert(12,"category","SMOTE")





final_X = X_train.append(X_train_smote).copy()

final_X.drop_duplicates(subset=list(final_X.columns[:-1]),inplace=True)

final_cat = final_X["category"]

final_X.drop(columns="category",inplace=True)

pca = PCA(n_components=2)

final_X = pd.DataFrame(pca.fit_transform(final_X))

final_X["category"] = list(final_cat)

final_X = final_X.loc[(final_X[0]<=200000) & (final_X[1]<=2000),:].copy()

sns.relplot(data=final_X,x=0,y=1,hue="category",alpha=0.6,s=100,height=6,aspect=1.5)

plt.title("Synthetic Examples Generated Using SMOTE");
for var in np.arange(feat_imp.shape[0],6,-1):

    X_new = X[feat_imp.iloc[:var,0]].copy()

    X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

    smote = SMOTE(random_state = 11) 

    X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

    final_rf = RandomForestClassifier(random_state=11)

    

    

    gscv = GridSearchCV(estimator=final_rf,param_grid={

        "n_estimators":[100,500,1000,5000],

        "criterion":["gini","entropy"]

    },cv=5,n_jobs=-1,scoring="f1_weighted")



    gscv.fit(X_train_smote,y_train_smote)

    print(str(var)+" variables:  "+str(gscv.best_estimator_)+"  F1 score: "+str(gscv.best_score_))
X_new = X[feat_imp.iloc[:8,0]].copy()

X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

smote = SMOTE(random_state = 11) 

X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

final_rf = RandomForestClassifier(random_state=11)

gscv = GridSearchCV(estimator=final_rf,param_grid={

    "n_estimators":[100,500,1000,5000],

    "criterion":["gini","entropy"]

},cv=5,n_jobs=-1,scoring="f1_weighted")



gscv.fit(X_train_smote,y_train_smote)

FINAL_MODEL = gscv.best_estimator_
FINAL_MODEL.score(X_train_smote,y_train_smote)
train_pred = FINAL_MODEL.predict(X_train_smote)

print(classification_report(y_train_smote,train_pred))
FINAL_MODEL.score(X_test,y_test) #Test set score
pred = FINAL_MODEL.predict(X_test)
print(classification_report(y_test,pred))
train_size,train_acc,test_acc = learning_curve(FINAL_MODEL, X_train_smote,y_train_smote,cv=5)

learn_df = pd.DataFrame({"Train_size":train_size,"Train_Accuracy":train_acc.mean(axis=1),"Test_Accuracy":test_acc.mean(axis=1)}).melt(id_vars="Train_size")

sns.lineplot(x="Train_size",y="value",data=learn_df,hue="variable")

plt.title("Learning Curve")

plt.ylabel("Accuracy");
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

plt.ylabel("Actual")

plt.xlabel("Prediction");
X_new = X[feat_imp.iloc[:8,0]].copy()

X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

smote = SMOTE(random_state = 11) 

X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

final_rf = RandomForestClassifier(random_state=11)

gscv = GridSearchCV(estimator=final_rf,param_grid={

    "n_estimators":[5000,7000],

    "criterion":["gini","entropy"],

    "max_depth":[3,5,7],

    "min_samples_split":[80,100],

    "min_samples_leaf":[40,50],

},cv=5,n_jobs=-1,verbose=11,scoring="f1_weighted")



gscv.fit(X_train_smote,y_train_smote)

FINAL_MODEL = gscv.best_estimator_
FINAL_MODEL
gscv.best_score_
FINAL_MODEL.score(X_train_smote,y_train_smote)
train_pred = FINAL_MODEL.predict(X_train_smote)

print(classification_report(y_train_smote,train_pred))
FINAL_MODEL.score(X_test,y_test) #Test set score
pred = FINAL_MODEL.predict(X_test)
print(classification_report(y_test,pred))
plot_roc_curve(FINAL_MODEL, X_test, y_test)

plt.show()
plot_precision_recall_curve(FINAL_MODEL, X_test, y_test)

plt.show()
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

plt.ylabel("Actual")

plt.xlabel("Prediction");