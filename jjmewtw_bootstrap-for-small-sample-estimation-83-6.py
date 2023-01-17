# Basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.preprocessing import minmax_scaling

import seaborn as sns

import matplotlib.pyplot as plt



# Libraries for bootstrap

from sklearn.utils import resample



# Libraries for estimation

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Pre-defined random state        

RandState = 100
df_path = '/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv'

df = pd.read_csv(df_path)

NoRecords = df.shape[0]
#df.head()

df.describe()
C = (df.dtypes == 'object')

CategoricalVariables = list(C[C].index)



print(CategoricalVariables)



Integer = (df.dtypes == 'int64') 

Float   = (df.dtypes == 'float64') 

NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)



print(NumericVariables)
Missing_Percentage = (df.isnull().sum()).sum()/np.product(df.shape)*100



print("The number of missing entries: " + str(round(Missing_Percentage,10)) + " %")
AllVars = list(df.columns)  

ContVars = ['age','ejection_fraction','creatinine_phosphokinase','platelets','serum_creatinine','serum_sodium','time']

BinVars = [x for x in AllVars if x not in ContVars]

df.index.name = 'Id'



sc_df = df

sc_df[ContVars] = minmax_scaling(df, columns=ContVars)
boot = resample(sc_df, replace=True, n_samples=round(NoRecords*0.80), random_state=RandState) # 240 ~= 299 * 80%

# boot.describe()



boot_indices_list = list(boot.index)  

boot_indices = pd.DataFrame(boot.index)  



# out of bag observations

oob = sc_df[~sc_df.index.isin(boot_indices_list)]

print('The number of records in bootstrap sample is: ' + format(boot.shape[0]) + '. The number of records in out-of-bag sample is: ' + format(oob.shape[0]))
boot_indices_agg = boot_indices.groupby('Id').Id.count()



(unique, counts) = np.unique(boot_indices_agg, return_counts=True)



boot_indices_agg = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['Value','Frequency'])



boot_indices_agg.plot(kind='bar',x='Value',y='Frequency')
def BootstrapRatio(Data, F): # Data for our data set, F for the size of sample

    BootSet = resample(Data, replace=True, n_samples=round(NoRecords*F))

    DeadBoot = BootSet.loc[BootSet.DEATH_EVENT == 1].shape[0] 

    DeadBootRatio = DeadBoot/BootSet.shape[0]

    OobSet = Data[~Data.index.isin(list(BootSet.index))]

    DeadOob = OobSet.loc[OobSet.DEATH_EVENT == 1].shape[0]

    DeadOobRatio = DeadOob/OobSet.shape[0]

    

    return DeadBootRatio, DeadOobRatio
B = 101 # How many runs - 1, so 100 runs



B_results = pd.DataFrame(index=range(1,B), columns=['DeathsBoot','DeathsOob'])



for i in range(1,B):

    B_results.DeathsBoot[i],B_results.DeathsOob[i] = BootstrapRatio(sc_df,0.8)

    

B_results
print('The diff is: '+format(round(B_results.DeathsBoot.mean() - B_results.DeathsOob.mean(),3)))
B_1 = 501 # How many runs - 1, so 500 runs



B_1_results = pd.DataFrame(index=range(1,B_1), columns=['DeathsBoot','DeathsOob','DeathsBootAvg','DeathsOobAvg'])



for i in range(1,B_1):

    B_1_results.DeathsBoot[i],B_1_results.DeathsOob[i] = BootstrapRatio(sc_df,0.8)

    B_1_results.DeathsBootAvg[i]=B_1_results.DeathsBoot.mean()

    B_1_results.DeathsOobAvg[i]=B_1_results.DeathsOob.mean()

    

B_1_results
sns.distplot(a=B_1_results['DeathsBoot'], hist=False, rug=True, label="Deaths Boot")

sns.distplot(a=B_1_results['DeathsOob'], hist=False, rug=True, label="Deaths Oob")

plt.legend();
# sns.regplot(data = B_1_results.reset_index(), x = 'index', y = 'DeathsBoot', fit_reg=False, label="Deaths Boot")

# sns.regplot(data = B_1_results.reset_index(), x = 'index', y = 'DeathsOob', fit_reg=False, label="Deaths Oob")



sns.regplot(data = B_1_results.reset_index(), x = 'index', y = 'DeathsBootAvg', fit_reg=False, label="Deaths Boot Avg")

sns.regplot(data = B_1_results.reset_index(), x = 'index', y = 'DeathsOobAvg', fit_reg=False, label="Deaths Oob Avg")
RF = DecisionTreeClassifier()



def BootstrapRandomForest(Data, F, Target): # Data for our data set, F for the size of sample

    BootSet = resample(Data, replace=True, n_samples=round(NoRecords*F))

    OobSet = Data[~Data.index.isin(list(BootSet.index))]

    RF.fit(BootSet.drop(columns=[Target]), BootSet[Target])

    predictions = RF.predict(OobSet.drop(columns=[Target]))

    score = accuracy_score(OobSet[Target], predictions)

    

    return score
B_2 = 1001 # How many runs - 1, so 1000 runs



B_2_results = pd.DataFrame(index=range(1,B_2), columns=['ScoreRF','ScoreXgb'])



for i in range(1,B_2):

    B_2_results.ScoreRF[i] = BootstrapRandomForest(sc_df,0.8,'DEATH_EVENT')

    

B_2_results
sns.distplot(B_2_results['ScoreRF'], color="g").set_title("Accuracy scores for random forest (bootstrapping method)", color="g")
alpha = 0.95

p = ((1.0-alpha)/2.0) * 100

lower = max(0.0, np.percentile(B_2_results['ScoreRF'], p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(B_2_results['ScoreRF'], p))

print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
Xgb =XGBClassifier( booster='gbtree',

             importance_type='gain', learning_rate=0.01,

             max_depth=4, min_child_weight=1.5, n_estimators=500, objective='binary:logistic')



def BootstrapXgb(Data, F, Target): # Data for our data set, F for the size of sample

    BootSet = resample(Data, replace=True, n_samples=round(NoRecords*F))

    OobSet = Data[~Data.index.isin(list(BootSet.index))]

    Xgb.fit(BootSet.drop(columns=[Target]), BootSet[Target])

    predictions = Xgb.predict(OobSet.drop(columns=[Target]))

    score = accuracy_score(OobSet[Target], predictions)

    

    return score



for i in range(1,B_2):

    B_2_results.ScoreXgb[i] = BootstrapXgb(sc_df,0.8,'DEATH_EVENT')

    

B_2_results
sns.distplot(B_2_results['ScoreXgb'], color="g").set_title("Accuracy scores for extreme boosting (bootstrapping method)", color="g")
alpha = 0.95

p = ((1.0-alpha)/2.0) * 100

lower = max(0.0, np.percentile(B_2_results['ScoreXgb'], p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(B_2_results['ScoreXgb'], p))

print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
sns.distplot(a=B_2_results['ScoreRF'], hist=False, rug=True, label="Accuracy Random Forest")

sns.distplot(a=B_2_results['ScoreXgb'], hist=False, rug=True, label="Accuracy Extreme boosting")

plt.legend();