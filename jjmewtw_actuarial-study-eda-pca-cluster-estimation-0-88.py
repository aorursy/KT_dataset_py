import numpy as np

import pandas as pd 

import matplotlib

import seaborn as sns

import holoviews as hv

import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from sklearn.utils import resample,shuffle

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split

from mlxtend.preprocessing import minmax_scaling

from sklearn.decomposition import PCA,SparsePCA,KernelPCA,NMF

from holoviews import opts

from sklearn import metrics, mixture, cluster, datasets

from sklearn.mixture import GaussianMixture

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score, roc_curve, auc

hv.extension('bokeh')



print('Libraries correctly loaded')
df_path = "../input/health-insurance-cross-sell-prediction/train.csv"

df_test_path = "../input/health-insurance-cross-sell-prediction/train.csv"

df = pd.read_csv(df_path)

df_test = pd.read_csv(df_test_path)



print('Number of rows: '+ format(df.shape[0]) +', number of features: '+ format(df.shape[1]))
C = (df.dtypes == 'object')

CategoricalVariables = list(C[C].index)



Integer = (df.dtypes == 'int64') 

Float   = (df.dtypes == 'float64') 

NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)



Missing_Percentage = (df.isnull().sum()).sum()/np.product(df.shape)*100

print("The number of missing entries before cleaning: " + str(round(Missing_Percentage,5)) + " %")
CategoricalVariables
df.Vehicle_Age.unique()
Vehicle_Age_map  = {'< 1 Year':0,'1-2 Year':1,'> 2 Years':2}



df['Vehicle_Age'] = df['Vehicle_Age'].map(Vehicle_Age_map)

df=df.set_index("id")



C = (df.dtypes == 'object')

CategoricalVariables = list(C[C].index)



Integer = (df.dtypes == 'int64') 

Float   = (df.dtypes == 'float64') 

NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)



df.head()
def highlight_cols(s, coldict):

    if s.name in coldict.keys():

        return ['background-color: {}'.format(coldict[s.name])] * len(s)

    return [''] * len(s)



def ExtractColumn(lst,j): 

    return [item[j] for item in lst] 
coldict = {'Gender':'lightcoral','Age':'lightcoral', 'Driving_License':'lightsalmon', 'Region_Code':'lightsalmon', 'Previously_Insured':'lightsalmon'

           , 'Vehicle_Age':'lightsalmon', 'Vehicle_Damage':'lightsalmon', 'Annual_Premium':'lightsalmon', 'Policy_Sales_Channel':'tomato'

           ,'Vintage':'tomato','Response':'darksalmon'}

df.iloc[0:5].style.apply(highlight_cols, coldict=coldict)
df_dummy = pd.get_dummies(df[CategoricalVariables], columns=CategoricalVariables)

df_numeric = df[NumericVariables]

df_final = pd.merge(df_numeric,df_dummy,on='id')



response = ['Response']

VariablesNoTarget = [x for x in df_final.columns if x not in response]

print("Dummy transformation was successful")
coldict_dummy = {'Gender_Female':'lightcoral','Gender_Male':'lightcoral','Age':'lightcoral', 'Driving_License':'lightsalmon' 

                   ,'Region_Code':'lightsalmon', 'Previously_Insured':'lightsalmon'

                   , 'Vehicle_Age':'lightsalmon', 'Vehicle_Damage_No':'lightsalmon', 'Vehicle_Damage_Yes':'lightsalmon', 'Annual_Premium':'lightsalmon'

                   , 'Policy_Sales_Channel':'tomato','Vintage':'tomato','Response':'darksalmon'}



df_final = df_final[['Age','Gender_Female','Gender_Male','Driving_License','Previously_Insured','Vehicle_Age','Region_Code','Vehicle_Damage_No'

               ,'Vehicle_Damage_Yes','Annual_Premium','Policy_Sales_Channel','Vintage','Response']]



df_final[VariablesNoTarget] = minmax_scaling(df_final, columns=VariablesNoTarget)

df_final.iloc[0:5].style.apply(highlight_cols, coldict=coldict_dummy)
SpearmanCorr = df_final.corr(method="spearman")

matplotlib.pyplot.figure(figsize=(10,10))

sns.heatmap(SpearmanCorr, vmax=.9, square=True, annot=True, linewidths=.3, cmap="YlGnBu", fmt='.1f')
age = hv.Distribution(df['Age'],label="Age").opts(color="red")

reg = hv.Distribution(df['Region_Code'],label="Region_Code").opts(color="green")

prem = hv.Distribution(df['Annual_Premium'],label="Annual_Premium").opts(color="yellow")

chan = hv.Distribution(df['Policy_Sales_Channel'],label="Policy_Sales_Channel").opts(color="blue")

vehage = hv.Distribution(df['Vehicle_Age'],label="Vehicle_Age").opts(color="purple")

vin = hv.Distribution(df['Vintage'],label="Vintage").opts(color="pink")



(age + reg + prem + chan + vehage + vin).opts(opts.Distribution(xlabel="Values", ylabel="Density", width=400, height=300,tools=['hover'],show_grid=True)).cols(3)
pca = PCA().fit(df_final[VariablesNoTarget])



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=70, facecolor='w', edgecolor='k')

ax0, ax1 = axes.flatten()



sns.set('talk', palette='colorblind')



font = {'family' : 'normal',

        'weight' : 'normal',

        'size'   : 12}



matplotlib.rc('font', **font)



ax0.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')

ax0.set_xlabel('Number of components')

ax0.set_ylabel('Cumulative explained variance');



ax1.bar(range(df_final[VariablesNoTarget].shape[1]),pca.explained_variance_)

ax1.set_xlabel('Number of components')

ax1.set_ylabel('Explained variance');



plt.tight_layout()

plt.show()
n_PCA_90 = np.size(np.cumsum(pca.explained_variance_ratio_)>0.9) - np.count_nonzero(np.cumsum(pca.explained_variance_ratio_)>0.9)

print("Already: " + format(n_PCA_90) + " components cover 90% of variance.")
#KPCA = KernelPCA(n_components = df_final[VariablesNoTarget].shape[1], kernel="rbf", fit_inverse_transform=True, gamma=10)

#KPCA_fit = KPCA.fit(df_final[VariablesNoTarget])

#X_KPCA = KPCA.fit_transform(df_final[VariablesNoTarget])

#X_KPCA_back = KPCA.inverse_transform(X_KPCA)
pca = PCA(4).fit((df_final[VariablesNoTarget]))



X_pca=pca.transform((df_final[VariablesNoTarget])) 



plt.matshow(pca.components_,cmap='viridis')

plt.yticks([0,1,2,3,4],['1st Comp','2nd Comp','3rd Comp','4th Comp'],fontsize=10)

plt.colorbar()

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],VariablesNoTarget,fontsize=10,rotation=30)

plt.tight_layout()

plt.show()
PCA_vars = [0]*len(VariablesNoTarget)



for i, feature in zip(range(len(VariablesNoTarget)),VariablesNoTarget):

    x = ExtractColumn(pca.components_,i)

    if ((max(x) > 0.4) | (min(x) < -0.4)):

        if abs(max(x)) > abs(min(x)):

            PCA_vars[i] = max(x)

        else:

            PCA_vars[i] = min(x)                 

    else:

        PCA_vars[i] = 0



PCA_vars = pd.DataFrame(list(zip(VariablesNoTarget,PCA_vars)),columns=('Name','Max absolute contribution'),index=range(1,13,1))      

PCA_vars = PCA_vars[(PCA_vars['Max absolute contribution']!=0)]

PCA_vars
df_business = df_final[['Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage']]

X = df_business.values



GM_n_components = np.arange(1, 8)

GM_models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in GM_n_components]



plt.figure(num=None, figsize=(8, 6), dpi=60, facecolor='w', edgecolor='r')

plt.plot(GM_n_components, [m.aic(X) for m in GM_models], label='AIC')

plt.tight_layout()

plt.legend(loc='best')

plt.xlabel('n_components');
GM_n_classes = 2



GMcluster = mixture.GaussianMixture(n_components=GM_n_classes, covariance_type='full',random_state = 0)

GMcluster_fit = GMcluster.fit(df_business)

GMlabels = GMcluster_fit.predict(df_business)



print('Number of clusters: ' + format(len(np.unique(GMlabels))))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5), facecolor='w', edgecolor='k')

ax = p3.Axes3D(fig)

ax.view_init(10, 40)

for l in np.unique(GMlabels):

    ax.scatter(X[GMlabels == l, 0], X[GMlabels == l, 1], X[GMlabels == l, 2],color=plt.cm.jet(float(l) / np.max(GMlabels + 1)),s=20, edgecolor='k')

plt.title('Expectation-maximization algorithm for business features clustering' )



plt.show()
df_final[['Business_Cluster']] = list(GMlabels)



coldict_cluster_1 = {'Gender_Female':'lightcoral','Gender_Male':'lightcoral','Age':'lightcoral', 'Driving_License':'lightsalmon' 

                   ,'Region_Code':'lightsalmon', 'Previously_Insured':'lightsalmon'

                   , 'Vehicle_Age':'lightsalmon', 'Vehicle_Damage_No':'lightsalmon', 'Vehicle_Damage_Yes':'lightsalmon', 'Annual_Premium':'lightsalmon'

                   , 'Policy_Sales_Channel':'tomato','Vintage':'tomato','Response':'darksalmon','Business_Cluster':'pink'}



df_final.iloc[0:5].style.apply(highlight_cols, coldict=coldict_cluster_1)
VariablesClient = [x for x in df_final[VariablesNoTarget] if x not in df_business.columns]



df_client = df_final[VariablesClient]

X = df_client.values



GMcluster_fit = GMcluster.fit(df_client)

GMlabels = GMcluster_fit.predict(df_client)



df_final[['Client_Cluster']] = list(GMlabels)



coldict_cluster_2 = {'Gender_Female':'lightcoral','Gender_Male':'lightcoral','Age':'lightcoral', 'Driving_License':'lightsalmon' 

                   ,'Region_Code':'lightsalmon', 'Previously_Insured':'lightsalmon'

                   , 'Vehicle_Age':'lightsalmon', 'Vehicle_Damage_No':'lightsalmon', 'Vehicle_Damage_Yes':'lightsalmon', 'Annual_Premium':'lightsalmon'

                   , 'Policy_Sales_Channel':'tomato','Vintage':'tomato','Response':'darksalmon','Business_Cluster':'pink','Client_Cluster':'pink'}



df_final = df_final[['Age','Gender_Female','Gender_Male','Driving_License','Previously_Insured','Vehicle_Age','Region_Code','Vehicle_Damage_No'

               ,'Vehicle_Damage_Yes','Annual_Premium','Policy_Sales_Channel','Vintage','Business_Cluster','Client_Cluster','Response']]



df_final.iloc[0:5].style.apply(highlight_cols, coldict=coldict_cluster_2)
Target= df_final['Response']

df_final_ = df_final.drop(['Response'],axis=1)



x_train,x_test,y_train,y_test = train_test_split(df_final_,Target,test_size=0.2,random_state=0)
ModelAverage = y_train.mean()

print(str(round(ModelAverage,5)))
GLM = LogisticRegression(solver='liblinear', random_state=0)

GLM_fit = GLM.fit(x_train, y_train)

GLM_probability = pd.DataFrame(GLM_fit.predict_proba(x_test))

GLM_probability.mean()
print("We expect: " +format(round((float(GLM_probability[1].mean() * x_test.shape[0]))))+ " 1's.")
GLM_clas = pd.DataFrame(GLM_fit.predict(x_test))

print("The rate is very low: "+ format(float(round(GLM_clas.mean(),5))) + " and translates to just: " + format(float(GLM_clas.mean() * x_test.shape[0])) + " records with 1's.")
GLM_Ret = hv.Distribution(GLM_probability[1],label="Probability of retention").opts(color="blue")

(GLM_Ret).opts(opts.Distribution(xlabel="Values", ylabel="Density", width=600, height=400,tools=['hover'],show_grid=True))
df_majority = df_final[df_final['Response']==0]

df_minority = df_final[df_final['Response']==1]

df_minority_upsampled = resample(df_minority,replace=True,n_samples=334399,random_state = 0)

balanced_df = pd.concat([df_minority_upsampled,df_majority])

balanced_df = shuffle(balanced_df)

balanced_df.Response.value_counts()
Target= balanced_df['Response']

df_final_ = balanced_df.drop(['Response'],axis=1)



x_train,x_test,y_train,y_test = train_test_split(df_final_,Target,test_size=0.2,random_state=0)



GLM = LogisticRegression(solver='liblinear', random_state=0)

GLM_fit = GLM.fit(x_train, y_train)

GLM_clas = pd.DataFrame(GLM_fit.predict(x_test))

GLM_probability = pd.DataFrame(GLM_fit.predict_proba(x_test))
fpr, tpr, _ = roc_curve(y_test, GLM_fit.predict_proba(x_test)[:,1])



plt.title('Logistic regression ROC curve')

plt.xlabel('FPR (Precision)')

plt.ylabel('TPR (Recall)')



plt.plot(fpr,tpr)

plt.plot((0,1), ls='dashed',color='black')

plt.show()

print ('Area under curve (AUC): ' ,format(round(auc(fpr,tpr),5)))
df_evaluation = pd.DataFrame(y_test)

df_evaluation[['GLM']] = list(GLM_fit.predict(x_test))
fpr, tpr, _ = roc_curve(y_test, x_test['Client_Cluster'])



plt.title('Client Cluster ROC curve')

plt.xlabel('FPR (Precision)')

plt.ylabel('TPR (Recall)')



plt.plot(fpr,tpr)

plt.plot((0,1), ls='dashed',color='black')

plt.show()

print ('Area under curve (AUC): ', format(round(auc(fpr,tpr),5)))
InvertClientCluster = x_test['Client_Cluster'].replace(0,-1)

InvertClientCluster = InvertClientCluster.replace(1,0)

InvertClientCluster = InvertClientCluster.replace(-1,1)

InvertClientCluster



fpr, tpr, _ = roc_curve(y_test, x_test['Business_Cluster'])



plt.title('Business Cluster ROC curve')

plt.xlabel('FPR (Precision)')

plt.ylabel('TPR (Recall)')



plt.plot(fpr,tpr)

plt.plot((0,1), ls='dashed',color='black')

plt.show()

print ('Area under curve (AUC): ', format(round(auc(fpr,tpr),5)))
space={ 'max_depth': hp.quniform("max_depth", 3,18,1),

        'gamma': hp.uniform ('gamma', 1,9),

        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),

        'reg_lambda' : hp.uniform('reg_lambda', 0,1),

        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),

        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),

        'n_estimators': 300,

        'seed': 0}



def objective(space):

    clf=XGBClassifier(n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],

                      reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),

                      colsample_bytree=int(space['colsample_bytree']))

    

    evaluation = [( x_train, y_train), ( x_test, y_test)]

    

    clf.fit(x_train, y_train,

            eval_set=evaluation, eval_metric="auc",

            early_stopping_rounds=10,verbose=False)

    

    pred = clf.predict(x_test)

    y_score = clf.predict_proba(x_test)[:,1]

    accuracy = accuracy_score(y_test, pred>0.5)

    Roc_Auc_Score = roc_auc_score(y_test, y_score)

    print ("ROC-AUC Score: ",Roc_Auc_Score)

    print ("SCORE: ", accuracy)

    return {'loss': -Roc_Auc_Score, 'status': STATUS_OK }



#trials = Trials()



#best_hyperparams = fmin(fn = objective,space = space,algo = tpe.suggest,max_evals = 100,trials = trials)



#print("The best hyperparameters are : ","\n")

#print(best_hyperparams)
XGB_=XGBClassifier(n_estimators = 300, max_depth = 13, gamma = 2.3408807913619945, reg_lambda = 0.3770436657913232,

                            reg_alpha = 40.0, min_child_weight=7.0,colsample_bytree = 0.5786479102658189 ,random_state = 0)

XGB_fit = XGB_.fit(x_train, y_train)

XGB_probability = XGB_fit.predict_proba(x_test)[:,1]

XGB_class = pd.DataFrame(XGB_fit.predict(x_test))
df_evaluation[['XGB']] = list(XGB_fit.predict(x_test))
fpr, tpr, _ = roc_curve(y_test, XGB_probability)



plt.title('XGBoost ROC curve')

plt.xlabel('FPR (Precision)')

plt.ylabel('TPR (Recall)')



plt.plot(fpr,tpr)

plt.plot((0,1), ls='dashed',color='black')

plt.show()

print ('Area under curve (AUC): ', format(round(auc(fpr,tpr),5)))