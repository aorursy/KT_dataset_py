# import important libraries here
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import random

random.seed(0)
!ls /kaggle/input/minor-project-2020
# read csv file
input_csv = '/kaggle/input/minor-project-2020/train.csv'
df = pd.read_csv(input_csv)
df.head()
df.info()
# All columns are either int or float, no string
df.describe()
# drop ID since it is of no use to us (it's a primary key)
df.drop('id',axis=1,inplace=True)
# remove duplicates
"""
df_ones = df[df['target']==1]
df_zeros = df[df['target']==0]
print(df_ones.shape)
print(df_zeros.shape)
# discard duplicate rows---only for target 0
print(df_zeros.shape)
subset_to_consider = ['col_'+str(i) for i in range(88)]
subset_to_consider.append('target')
df_zeros = df_zeros.drop_duplicates(subset=subset_to_consider,keep='first')
print(df_zeros.shape)

df = pd.concat([df_zeros,df_ones])
print(df.shape)
"""
# removing duplicates affects the results adverserly. Counter-intuitive, but yeah
# correlation matrix
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style('whitegrid')
plt.subplots(figsize = (100,100))
sns.heatmap(df.corr(),
            annot=True,
            mask = mask,
            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='white',
            fmt='.2g',
            center = 0,
            square=True)
# no pair of columns correlated enough to be discarded
predictors = list(df.columns)
predictors.remove('target')
len(predictors)
X = df[predictors]
y = df['target']
print("X's SHAPE: " + str(X.shape))
print("y's SHAPE: " + str(y.shape))
# check if the dataset is skewed
print("Number of 0's: " + str((y==0).sum()))
print("Number of 1's: " + str((y==1).sum()))

# very skewed :(
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
# normalise
#scalar = MinMaxScaler()
#X_train = scalar.fit_transform(X_train)
#X_test = scalar.transform(X_test)
print("0's in test: " + str((y_test==0).sum()))
print("1's in test: " + str((y_test==1).sum()))
# resample X and y
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
#overs = RandomOverSampler(sampling_strategy=0.1,random_state=0)
#unders = RandomUnderSampler(sampling_strategy=0.8,random_state=0)
#X_train, y_train = overs.fit_resample(X_train,y_train)

#X_train, y_train = unders.fit_resample(X_train,y_train)
print("Number of 0's: " + str((y_train==0).sum()))
print("Number of 1's: " + str((y_train==1).sum()))

os = SMOTE(random_state=0)
X_train,y_train = os.fit_sample(X_train, y_train)
print("Number of 0's: " + str((y_train==0).sum()))
print("Number of 1's: " + str((y_train==1).sum()))

print(X_train.shape)
print(y_train.shape)
# grid search using stratified K-fold validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score


param_grid={"C":[0.1,1000], "penalty":["l2"],"solver":["newton-cg"]}

scorer = make_scorer(roc_auc_score)
model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring=scorer, iid=True,verbose=10,n_jobs=-1)
model.fit(X_train, y_train)
# print(model.cv_results_)
print(model.best_score_)
print(model.best_params_)
pred=model.predict_proba(X_test)
print(roc_auc_score(y_test,pred[:,1]))
pred=model.predict_proba(X_test)
print(roc_auc_score(y_test,pred[:,1]))
# read csv file
test_csv = '/kaggle/input/minor-project-2020/test.csv'
df = pd.read_csv(test_csv)
cols = list(df.columns)
cols.remove('id')
print(len(cols))
predictors = list(df.columns)
predictors.remove('id')
X_inf = df[predictors]
#scaled_X_inf = scalar.transform(X_inf)
y_inf = model.predict_proba(X_inf.values)
ids = list(df['id'])
y_inf_lst = list(y_inf[:,1])
sub_dict = {'id':ids,'target':y_inf_lst}
sub = pd.DataFrame(sub_dict)
sub.head()
sub.to_csv('submission_LR_with_prob_SMOTE.csv',index=False)