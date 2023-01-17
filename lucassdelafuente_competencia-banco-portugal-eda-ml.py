import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, confusion_matrix, classification_report
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
plt.rcParams['figure.figsize'] = (16,8)
df = pd.read_csv("../input/Banco-PF-DSCOR9/data_train.csv") 
df_test = pd.read_csv('../input/Banco-PF-DSCOR9/data_test.csv')
df.head(10)
df["y"].replace(to_replace={"yes":1,"no":0}, inplace=True)
df.info()
df_test.info()
# Data set de train
unknown_list = {}

for column in df.columns:
    if(df[column].dtype=='object'):
        unknown = df[df[column]=='unknown']
        test = unknown[column].value_counts()
        unknown_list[column]= test
unknown_list
# Data set de test
unknown_list = {}

for column in df_test.columns:
    if(df_test[column].dtype=='object'):
        unknown = df_test[df_test[column]=='unknown']
        test = unknown[column].value_counts()
        unknown_list[column]= test
unknown_list
#@title
sns.boxplot(df.age)
#@title
bins = [10, 20, 30, 40, 50, 60, 70, 100]
cats = pd.cut(df.age, bins)
#@title
ax = sns.countplot(cats, palette="Blues_r", #hue=df["y"],
                   order = cats.value_counts().index)
sns.set(font_scale=1.1)
ax.set_title('Distribución de clientes por edad')
value=round(cats.value_counts()/len(cats)*100,2)
for p, label in zip(ax.patches, value):
    ax.annotate(label, (p.get_x()+0.23, p.get_height()+0.25))
jobx = sns.countplot(df["job"], palette="Greens_r", #hue=df['y'],
                   order = df["job"].value_counts().index)
jobx.set_title('Distribución de clientes por puesto de trabajo')
sns.set(font_scale=0.78)
value=round(df["job"].value_counts()/len(df["job"])*100,2)
for p, label in zip(jobx.patches, value):
    jobx.annotate(label, (p.get_x()+0.23, p.get_height()+0.25))
maritx = sns.countplot(df["marital"], palette="GnBu_d", #hue=df['y'],
                   order = df["marital"].value_counts().index)
sns.set(font_scale=1)
maritx.set_title('Distribución de clientes por estado civil')
value=round(df["marital"].value_counts()/len(df["marital"])*100,2)
for p, label in zip(maritx.patches, value):
    maritx.annotate(label, (p.get_x()+0.33, p.get_height()+0.25))
edux = sns.countplot(df["education"], palette="Oranges_r", #hue=df["y"], 
                     #hue=(df["job"]=="blue-collar"),                 
                   order = df["education"].value_counts().index)
edux.set_title('Distribución de clientes por nivel de educación')
value=round(df["education"].value_counts()/len(df["education"])*100,2)
for p, label in zip(edux.patches, value):
    edux.annotate(label, (p.get_x()+0.28, p.get_height()+0.25))
f, axes = plt.subplots(1, 3)

sns.barplot(x=df['default'].value_counts().index, y=df['default'].value_counts(normalize=True), ax=axes[0]).set_title("Credito impago")
sns.barplot(x=df['housing'].value_counts().index, y=df['housing'].value_counts(normalize=True), ax=axes[1]).set_title("Prestamo hipotecado")
sns.barplot(x=df['loan'].value_counts().index, y=df['loan'].value_counts(normalize=True), ax=axes[2]).set_title("Prestamo personal")
f, axes = plt.subplots(1, 3)

sns.barplot(x= df['contact'].value_counts().index, y= df['contact'].value_counts(normalize=True), ax=axes[0]).set_title("Medio de comunicación")
sns.barplot(x= df['month'].value_counts().index, y= df['month'].value_counts(normalize=True), ax=axes[1]).set_title("Mes del último contacto")
sns.barplot(x= df['day_of_week'].value_counts().index, y= df['day_of_week'].value_counts(normalize=True), ax=axes[2]).set_title("Dia del último contacto")
#@title
bins2=[0,2,4,6,8,10,12,14,56]
campaign_cats = pd.cut(df.campaign, bins2)
#@title
ax = sns.countplot(campaign_cats, palette="bone_r", #hue=df["y"],
                      order = campaign_cats.value_counts().index)
ax.set_title('Distribución de contacto de última campaña por cantidad de contactos')
value=round(campaign_cats.value_counts()/len(campaign_cats)*100,2)
for p, label in zip(ax.patches, value):
    ax.annotate(label, (p.get_x()+0.33, p.get_height()+0.25))
bins3=[0,2,4,6,8,10,12,14,999]
pdays_cat = pd.cut(df.pdays, bins3)
pdx = sns.countplot(pdays_cat, palette="bone_r", #hue=df["y"],
                      order = pdays_cat.value_counts().index)
value=round(pdays_cat.value_counts()/len(pdays_cat)*100,2)
for p, label in zip(pdx.patches, value):
    pdx.annotate(label, (p.get_x()+0.33, p.get_height()+0.25))
px = sns.countplot(df['previous'], palette="bone_r", #hue=df["y"],
                      order = df['previous'].value_counts().index)
value=round(df['previous'].value_counts()/len(df['previous'])*100,2)
for p, label in zip(px.patches, value):
    px.annotate(label, (p.get_x()+0.33, p.get_height()+0.25))
poutcx = sns.countplot(df['poutcome'], palette="bone_r", #hue=df["y"],
                      order = df['poutcome'].value_counts().index)
value=round(df['poutcome'].value_counts()/len(df['poutcome'])*100,2)
for p, label in zip(poutcx.patches, value):
    poutcx.annotate(label, (p.get_x()+0.33, p.get_height()+0.25))
sns.boxplot(df['emp.var.rate'])
df_economic = df.copy()
df_economic['month'] = df_economic['month'].map({'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov': 11, 'dec':12})
sns.lineplot(x = 'month', y= 'cons.price.idx', data=df_economic)
sns.distplot(df["cons.price.idx"])
sns.lineplot(x = 'month', y= 'cons.conf.idx', data=df_economic)
ax=sns.lineplot(x = 'month', y= 'cons.price.idx', data=df_economic,color="g")
ax2 = plt.twinx()
sns.lineplot(x = 'month', y= 'cons.conf.idx', data=df_economic, ax=ax2, color="r") 
sns.lineplot(x = 'month', y= 'euribor3m', data=df_economic)
sns.boxplot(df['nr.employed'])
sns.barplot(x = df['y'].value_counts().index, y= df['y'].value_counts())
age = df[df['y']==1]
age_no= df[df['y']==0]
sns.distplot(age['age'], label='Si')
sns.distplot(age_no['age'], label='No')
#plt.legend()
#@title
edad_y={}
a=10
while True:
    if a==100:
        break
    b=a+10
    df_filt=df[(df["y"]==1)]
    df_filt=df_filt[(df_filt["age"]>a) & (df_filt["age"]<b)]
    comp=round(len(df_filt)/len(df[(df["age"]>a) & (df["age"]<b)])*100,2)
    inter=str(a)+"-"+str(b)
    print(comp, "% de los de ",inter, "compraron el plazo fijo")
    edad_y[inter]=comp
    a+=10
ax = sns.countplot(data=df, x='job', hue='y')
total = float(len(df)) 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}%'.format((height/total)*100),
            ha="center") 
plt.show()
#@title
job_y={}
for i in df["job"].unique():
    df_filt=df[(df["y"]==1) & (df["job"]==i)]
    comp=round(len(df_filt)/len(df[df["job"]==i])*100,2)
    print(comp, "% de los",i, "compraron el plazo fijo")
    job_y[i]=comp
#@title
print("Valores nulos: ",(df.job=="unknown").sum())
print("Porcentaje",(df.job=="unknown").sum()/len(df)*100,"%")
ax = sns.countplot(data=df, x='marital', hue='y')
total = float(len(df)) 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}%'.format((height/total)*100),
            ha="center") 
plt.show()
marital_y={}
for i in df["marital"].unique():
    df_filt=df[(df["y"]==1) & (df["marital"]==i)]
    ratio=round(len(df_filt)/len(df[df["marital"]==i])*100,2)
    print(ratio, "% de los",i, "compraron el plazo fijo")
    marital_y[i]=ratio
f, ax = plt.subplots(1, 3)

sns.countplot(data=df, x='default',  hue='y', ax = ax[0]).set_title("Credito impago")
sns.countplot(data=df, x='housing',  hue='y', ax = ax[1]).set_title("Prestamo hipotecado")
sns.countplot(data=df, x='loan',  hue='y', ax = ax[2]).set_title("Prestamo personal")
sns.heatmap(df.corr(), annot=True, fmt= '.2f')
df.drop(columns=['id', 'previous', 'default'], inplace=True, axis=1)
df_test.drop(columns=['id', 'previous', 'default'], inplace=True, axis=1)
plt.rcParams['figure.figsize'] = (20,12)
sns.heatmap(df.corr(), annot=True, fmt= '.2f')
# Para TRAIN
for column in df.columns.drop(["month", "day_of_week"]):
    if df[column].dtype=='object':
            df = pd.concat((df.drop(columns=[column]), pd.get_dummies(df[column], prefix=column)) , axis=1)
# Para TEST                            
for column in df_test.columns.drop(["month", "day_of_week"]):
    if df_test[column].dtype=='object':
            df_test = pd.concat((df_test.drop(columns=[column]), pd.get_dummies(df_test[column], prefix=column)) , axis=1)
X = df.drop(columns=["y", "month", "day_of_week"])
X_Test = df_test.drop(columns=["month", "day_of_week"])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
def mostrar_metricas(y_test, y_pred_test):
    #Matriz_de_confusión
    cm = confusion_matrix(y_test, y_pred_test)
    cross= pd.crosstab(y_test, y_pred_test, rownames=['True'], colnames=['Predicted'], margins=True)
    cross
    sns.heatmap(cm, annot=True, fmt='.2f')
    plt.show() 
    
    #Precision_Recall
    print(classification_report(y_test, y_pred_test))
    
    #ROC_AUC
    #fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
    #roc_auc = auc(fpr, tpr)

    #plt.figure()
    #plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve')
    #plt.legend(loc="lower right")
    plt.show()
dct = DecisionTreeClassifier(class_weight='balanced', max_depth=10)
dct.fit(X_train, y_train)

y_pred_test = dct.predict(X_test)

dct_score = f1_score(y_test, y_pred_test, average='macro')

mostrar_metricas(y_test, y_pred_test)
rdm = BalancedRandomForestClassifier(max_depth=20, max_features='auto')
rdm.fit(X_train, y_train)

y_score = rdm.predict_proba(X_test)

y_pred_train = rdm.predict(X_train)
y_pred_test = rdm.predict(X_test)
rdm_balanced_score = f1_score(y_test, y_pred_test, average='macro')

mostrar_metricas(y_test, y_pred_test)
max_depth = np.arange(1, 20, 2)
train_score = []
test_score = []

for depth in max_depth: 
    rdm = RandomForestClassifier(criterion='gini', max_depth=depth, random_state=42, class_weight="balanced")
    rdm.fit(X_train, y_train)
    
    train_score.append(rdm.score(X_train, y_train))
    test_score.append(rdm.score(X_test, y_test))
plt.figure(figsize=(12,10))

feat_imp_df = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rdm.feature_importances_), X_train.columns), reverse=True))


mapping = {feat_imp_df.columns[0]:'Importancia', feat_imp_df.columns[1]: 'Variable'}
feat_imp_df = feat_imp_df.rename(columns=mapping)
sns.barplot(x=feat_imp_df['Importancia'],y=feat_imp_df['Variable'], palette="Greens_d")
sns.lineplot(x=max_depth ,y=train_score, label='Train score')
sns.lineplot(x=max_depth ,y=test_score, label= 'Test score')
plt.ylabel('Score')
plt.xlabel('Profundiad')
max_accuracy_rdm_train =  train_score[np.argmax(test_score)]*100
max_accuracy_rdm = np.max(test_score)*100

# En la profundidad donde se obtuvo el mejor valor para TEST
print('Porcentaje de aciertos sobre el set de entrenamiento:', max_accuracy_rdm_train)
print('Porcentaje de aciertos sobre el set de evaluación:', max_accuracy_rdm)
rdm = RandomForestClassifier(max_depth=max_depth[np.argmax(test_score)], n_jobs=-1, n_estimators=42, class_weight="balanced")
rdm.fit(X_train, y_train)

y_score = rdm.predict_proba(X_test)

y_pred_train = rdm.predict(X_train)
y_pred_test = rdm.predict(X_test)

rdm_score = f1_score(y_test, y_pred_test, average='macro')

mostrar_metricas(y_test, y_pred_test)
#kf = KFold(n_splits=5, shuffle=True)
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
#cv = StratifiedKFold(n_splits=5, shuffle=True)

param_grid = {
    'max_depth' : [5, 10, 20],
    'max_features' : ['sqrt', 'log2', X_test.shape[1]//2],
    'n_estimators' :[50, 100, 200],
    'min_samples_split':[5, 11, 20],
}
rdm_balanced = RandomForestClassifier(oob_score=True, random_state=42, class_weight="balanced")

rs_rf = RandomizedSearchCV(rdm_balanced, param_distributions=param_grid, scoring='f1_macro', cv=cv, verbose=1, n_jobs=-1, return_train_score=True)
#gs2_rf = GridSearchCV(rdm_balanced, param_grid=param_grid, scoring='f1_macro', cv=cv, verbose=1, n_jobs=-1, return_train_score=True)

rs_rf.fit(X_train, y_train)

y_score = rs_rf.predict_proba(X_test)

y_pred_train = rs_rf.predict(X_train)
y_pred_test = rs_rf.predict(X_test)

rs_rf_score = f1_score(y_test, y_pred_test, average='macro')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(weights='uniform', n_neighbors=3)

knn .fit(X_train, y_train)

y_pred_test = knn .predict(X_test)

knn_score = f1_score(y_test, y_pred_test, average='macro')
knn_score
xgb = XGBClassifier(objective='binary:logitraw', gamma=0.02, min_child_weight=4, verbosity=1, random_state=42)

params = [
    { # booster gbtree
    'booster': ['dart'],
    'max_depth': [2, 4, 5],
    'learning_rate': np.linspace(1e-5, 1, 3),
    'n_estimators': [100, 200],
    'scale_pos_weight': np.arange(1, 15, 4)  # recomendado:  
    }
]

#skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 42)
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3)

gs_xgb = GridSearchCV(xgb, param_grid=params, scoring='f1_macro', n_jobs=-1, cv=cv, verbose=1)

gs_xgb.fit(X_train, y_train)

y_pred_test = gs_xgb.predict(X_test)
y_score = gs_xgb.predict_proba(X_test)

xgb_score = f1_score(y_test, y_pred_test, average='macro')

mostrar_metricas(y_test, y_pred_test)
from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(penalty='l2', solver='liblinear')

lgs.fit(X_train, y_train)

y_pred_test = lgs.predict(X_test)

lgs_score = f1_score(y_test, y_pred_test, average='macro')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred_test = gnb.predict(X_test)

gnb_score = f1_score(y_test, y_pred_test, average='macro')
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('xgboost', gs_xgb), ('random', rs_rf), ('decisionTree', dct)], voting='hard', n_jobs=-1)

voting_clf.fit(X_train, y_train)

y_pred_test = voting_clf.predict(X_test)

f1_score(y_test, y_pred_test, average='macro')
for p in np.linspace(0, 1, 10):
    predict_probs = gs_xgb.predict_proba(X_test)[:,1] * p + rs_rf.predict_proba(X_test)[:,1] * ((1-p)/2) + lgs.predict_proba(X_test)[:,1] * ((1-p)/2)
    #predict_probs = gs_rf.predict_proba(X_test)[:,1] * p + dct.predict_proba(X_test)[:,1] * (1-p) 
    
    predict = np.where(predict_probs >= 0.4, 1, 0)

    print(f'xgboost * {p:.1f} + randomF * {(1-p)/2:.1f} + decisionT * {(1-p)/2:.1f}: {f1_score(y_test, predict, average="macro"):.4f}')
    #print(f'xgboost * {p:.1f} + L * {1-p:.1f}: {f1_score(y_test, predict, average="macro"):.4f}')

mostrar_metricas(y_test, predict)
models = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest Balanceado', 'Random Forest', 'Random Forest Grid Search', 'KNN' ,'XGBoost', 'Regresión Logistica', 'Naive Bayes'],
    'f1-macro': [dct_score, rdm_balanced_score, rdm_score, rs_rf_score, knn_score, xgb_score, lgs_score, gnb_score]})
models.sort_values(by='f1-macro', ascending=False)
predictions = xxx.predict(X_Test)

##Si es con el Voting
#predict_probs = random_search.predict_proba(X_Test)[:,1] * 0.8 + gs_rf.predict_proba(X_Test)[:,1] * 0.1 + lgs.predict_proba(X_Test)[:,1] * 0.1#
#predict = np.where(predict_probs >= 0.4, 1, 0)

output = pd.Series(predictions, name='y').to_csv('sample_submit.csv', index_label='id')
import pickle
with open('NOMBRE.pkl', 'wb') as fp:
    pickle.dump(poner_el_modelo_aca, fp)