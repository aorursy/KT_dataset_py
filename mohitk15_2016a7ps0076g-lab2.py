import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier



from tqdm import tqdm_notebook
train_data = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

test_data = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
train_data.head()
train_data.info()
train_data['class'].value_counts()
train_data=train_data.drop('id',axis=1)
train_data.drop_duplicates(inplace=True)

train_data.shape
corr = train_data.corr()



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
df = train_data

features = ['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

features_all = ['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']
#for knowing which feature is best for predicting

X_be = df[features_all]  #independent columns

Y_be = df['class']    #target column i.e price range

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=7)

fit = bestfeatures.fit(X_be,Y_be)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_be.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(11,'Score'))  #print 10 best features
X = df[features].copy()

Y = df['class'].copy()

X_test = test_data[features].copy()

X_id = test_data['id'].copy()
X_test.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X[features])

X_test_scaled = scaler.fit_transform(X_test[features])

X_test = X_test_scaled

X_new = X_scaled


X_train,X_val,Y_train,Y_val = train_test_split(X_new,Y,test_size=0.2,random_state=0,stratify=Y)
Y_train.value_counts(),Y_val.value_counts()
# m = 0.0

# ran = 0

# est = 0

# for i in  tqdm_notebook(range(1,10)):

#     for j in tqdm_notebook(range(1,50)):

#         clf2 = RandomForestClassifier(random_state=j).fit(X_train,Y_train)

#         Y_pred_2 = clf2.predict(X_val).round()

#         t = accuracy_score(Y_val, Y_pred_2)

#         if t > m:

#             m=t

#             ran=j

#             est=i

# print(m,ran,est)
clf2 = RandomForestClassifier(n_estimators=5,random_state=65).fit(X_new,Y)

Y_pred_2 = clf2.predict(X_test)
Y_pred_2.shape
Y_out = pd.DataFrame(Y_pred_2,columns=['class'])

Y_out = pd.concat([X_id,Y_out],axis=1)
Y_out['class'].value_counts()
Y_out.to_csv('output1.csv',index=False)
estimators = [('rf', RandomForestClassifier(random_state=65, n_estimators=6)), ('et', ExtraTreesClassifier(random_state=3, n_estimators=6)), ('xgb', XGBClassifier())]



soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_new,Y)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_new,Y)

soft_acc = soft_voter.predict(X_test)

hard_acc = hard_voter.predict(X_test)

Y_out = pd.DataFrame(soft_acc,columns=['class'])

Y_out = pd.concat([X_id,Y_out],axis=1)

Y_out['class'].value_counts()
Y_out = pd.DataFrame(hard_acc,columns=['class']) #best 

Y_out = pd.concat([X_id,Y_out],axis=1)

Y_out['class'].value_counts()
Y_out.to_csv('output1_2.csv',index=False)