# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn


# Any results you write to the current directory are saved as output.
building_own = pd.read_csv('../input/building-dataset-hackerearth-ml-6/Building_Ownership_Use.csv')
building_own.head()
train = pd.read_csv('../input/building-dataset-hackerearth-ml-6/train.csv')
train.shape
building_str = pd.read_csv('../input/building-dataset-hackerearth-ml-6/Building_Structure.csv')
test = pd.read_csv('../input/building-dataset-hackerearth-ml-6/test.csv')
test.shape

#Merging the files
combine = pd.merge(building_own,building_str, on='building_id')
res_train = pd.merge(combine,train, on = 'building_id')
res_test = pd.merge(combine,test, on = 'building_id')
res_train.head()
res_train.describe()



res_train = res_train.drop((['vdcmun_id_y','district_id_y','ward_id_y','vdcmun_id','district_id']) , axis = 1)

res_test = res_test.drop((['vdcmun_id_y','district_id_y','ward_id_y','vdcmun_id','district_id']) , axis = 1)
len(res_train)
columns = list(res_train.columns.values)
#columns
res_test.shape
res_train.shape
#res_train.dtypes
res_train.columns[res_train.isnull().any()]
miss = res_train.isnull().sum() / len(res_train)
miss = miss[miss > 0]
miss
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#Plotting the values

sns.set(style = "whitegrid", color_codes = True)
sns.barplot(x = 'Name', y = 'count',data = miss )
plt.xticks(rotation = 90)
target = res_train['damage_grade'].value_counts()
print(target)
target = target.to_frame()
target.columns = ['Count']
target.index.names = ['Damage_Grade']
target['Damage_Grade'] = target.index 

#Plotting the Target variable
sns.set(style = 'whitegrid', color_codes = True)
sns.barplot(x = 'Damage_Grade', y = 'Count', data = target)
plt.xticks(rotation = 90)
res_train['legal_ownership_status'].value_counts().plot.bar()
#Age of Building
sns.distplot(res_train['age_building'])

#Skewness
print("The skewness of Age of building is {}".format(res_train['age_building'].skew()))
print(res_train['count_families'].value_counts())
res_train['count_families'].value_counts().sort_index().plot.line()
print("The skewness of Count of families is {}".format(res_train['count_families'].skew()))
print(res_train['condition_post_eq'].unique())
res_train['condition_post_eq'].value_counts().plot.bar()
# fig=plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.hist(res_train['plinth_area_sq_ft'], bins = 100)
# plt.title('Plint Area Distribution')
# plt.xlabel('Plint Area in Square feet')
# plt.ylabel('Frequency')
# plt.show()

sns.distplot(res_train['plinth_area_sq_ft'],hist = True, color = 'darkgreen', bins = 100, hist_kws={'edgecolor':'black'})
#len(res_train['height_ft_post_eq'].unique())
#sns.distplot(res_train['height_ft_pre_eq'].value_counts(), hist = True, color = 'darkblue', bins = 79)
res_train['height_ft_pre_eq'].value_counts().plot.hist()
res_train['height_ft_post_eq'].value_counts().plot.hist()
x1 = list(res_train['height_ft_pre_eq'])
x2 = list(res_train['height_ft_post_eq'])

colors = ['#E69F00', '#56B4E9']
names = ['Height Before Earthquake', 'Height After Earthquake']



plt.hist(x1, alpha = 0.5, label = names[0])
plt.hist(x2, alpha = 0.5, label = names[1])

plt.legend()
res_train['area_assesed'].unique()
sns.set(font_scale=0.7)
sns.countplot(res_train['area_assesed'])
import scipy.stats as ss
from scipy.stats import chi2_contingency
from scipy.stats import chi2
clean_up = {'damage_grade' : {"Grade 1" : 1, "Grade 2" : 2, "Grade 3" : 3,"Grade 4" : 4,"Grade 5" : 5}}
train_dg = pd.DataFrame()
train_dg['damage_grade'] = res_train['damage_grade']

train_dg.replace(clean_up, inplace = True)
print(train_dg['damage_grade'].head())

res_train['damage_grade'] = train_dg['damage_grade']
res_train['damage_grade'].head()
cat = [c for c in res_train if res_train[c].dtypes == "object"]
cat.remove('building_id')
def ChiSquareTest(cat,res_train):
  
  for c in cat:
    print(c)
    tab = pd.crosstab(res_train['damage_grade'], res_train[c])
    stat, p, dof, expected = chi2_contingency(tab)
    print('dof=%d' % dof)
    #print(expected)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    if abs(stat) >= critical:
      print('Dependent (reject H0)')
    else:
      print('Independent (fail to reject H0)')
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
      print('Dependent (reject H0)')
    else:
      print('Independent (fail to reject H0)')
     
    print(" ")

  
  
    
ChiSquareTest(cat,res_train)
cat_binary = [c for c in res_train if len(res_train[c].unique()) == 2]
cat_binary

def cramers_v(x, y):
  correlation_coeff = []
  for c in cat_binary:
    confusion_matrix = pd.crosstab(res_train[c],y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    
    correlation_coeff.append([c,np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))])
    
  return correlation_coeff
cramers_v(cat_binary, res_train['damage_grade'])
ChiSquareTest(cat_binary,res_train)
res_train.drop(['has_secondary_use_use_police','building_id'], axis = 1)
res_test.drop(['has_secondary_use_use_police','building_id'], axis = 1)
cont = [c for c in res_train if len(res_train[c].unique()) > 15]
cont
indices = 0,1,2,3
cont = [i for j, i in enumerate(cont) if j not in indices]
cont
def hist_cont(cont, res_train):
  nd = pd.melt(res_train, value_vars = cont)
  n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
  n1 = n1.map(sns.distplot, 'value')
  
  return n1
hist_cont(cont,res_train)

from scipy.stats import skew
skewed = res_train[cont].apply(lambda x: skew(x.dropna().astype(float)))
print(skewed)
skewed = skewed[skewed > 0.30]
skewed = skewed.index
res_train[skewed] = np.log1p(res_train[skewed])
res_test[skewed] = np.log1p(res_test[skewed])


hist_cont(cont,res_train)

def calculateDistribution(cat, res_train):
  for c in cat:
    print(c)
    print((res_train[c].value_counts())/ len(res_train[c]))
    print(" ")
calculateDistribution(cat,res_train)
res_train_copy = res_train
res_train_copy.head()
res_test_copy = res_test
res_test_copy.head()
#Legal Ownership Status
# Private          0.962939
# Public           0.021294
# Institutional    0.009940
# Other            0.005827

res_train_copy['IsPrivate'] = (res_train_copy["legal_ownership_status"] == "Private") * 1 
res_test_copy['IsPrivate'] = (res_test_copy["legal_ownership_status"] == "Private") * 1 

#land_surface_condition
# Flat              0.830164
# Moderate slope    0.137870
# Steep slope       0.031966
res_train_copy['IsFlat'] = (res_train_copy["land_surface_condition"] == "Flat") * 1
res_test_copy['IsFlat'] = (res_test_copy["land_surface_condition"] == "Flat") * 1
# foundation_type
# Mud mortar-Stone/Brick    0.841331
# Bamboo/Timber             0.057590
# Cement-Stone/Brick        0.054574
# RC                        0.040962
# Other                     0.005543

res_train_copy['IsMudFoundation'] = (res_train_copy["foundation_type"] == "Mud mortar-Stone/Brick") * 1
res_test_copy['IsMudFoundation'] = (res_test_copy["foundation_type"] == "Mud mortar-Stone/Brick") * 1
# roof_type
# Bamboo/Timber-Light roof    0.701973
# Bamboo/Timber-Heavy roof    0.235493
# RCC/RB/RBC                  0.062535
res_train_copy['IsBambooRoofLight'] = (res_train_copy["roof_type"] == "Bamboo/Timber-Light roof") * 1
res_test_copy['IsBambooRoofLight'] = (res_test_copy["roof_type"] == "Bamboo/Timber-Light roof") * 1
# ground_floor_type
# Mud            0.804121
# Brick/Stone    0.095558
# RC             0.094446
# Timber         0.003902
# Other          0.001974
res_train_copy['IsFloorTypeMud'] = (res_train_copy["ground_floor_type"] == "Mud") * 1
res_test_copy['IsFloorTypeMud'] = (res_test_copy["ground_floor_type"] == "Mud") * 1

# other_floor_type
# TImber/Bamboo-Mud    0.632353
# Timber-Planck        0.168374
# Not applicable       0.152426
# RCC/RB/RBC           0.046847
res_train_copy['OtherFloorTypeMud'] = (res_train_copy["other_floor_type"] == "TImber/Bamboo-Mud") * 1
res_test_copy['OtherFloorTypeMud'] = (res_test_copy["other_floor_type"] == "TImber/Bamboo-Mud") * 1
# position
# Not attached       0.774600
# Attached-1 side    0.165042
# Attached-2 side    0.051155
# Attached-3 side    0.009203

res_train_copy['IsNotAttached'] = (res_train_copy["position"] == "Not attached") * 1
res_test_copy['IsNotAttached'] = (res_test_copy["position"] == "Not attached") * 1
# plan_configuration
# Rectangular                        0.959290
# Square                             0.021969
# L-shape                            0.014263
# Multi-projected                    0.001309
# T-shape                            0.001208
# Others                             0.000966
# U-shape                            0.000562
# Building with Central Courtyard    0.000180
# E-shape                            0.000158
# H-shape                            0.000095
res_train_copy['IsPlanConfigRectangular'] = (res_train_copy["plan_configuration"] == "Rectangular") * 1
res_test_copy['IsPlanConfigRectangular'] = (res_test_copy["plan_configuration"] == "Rectangular") * 1
res_train_copy['count_floors_change'] = (res_train_copy['count_floors_post_eq'] - res_train_copy['count_floors_pre_eq'])
res_train_copy['height_ft_change'] = (res_train_copy['height_ft_post_eq'] - res_train_copy['height_ft_pre_eq'])
res_test_copy['count_floors_change'] = (res_test_copy['count_floors_post_eq'] - res_test_copy['count_floors_pre_eq'])
res_test_copy['height_ft_change'] = (res_test_copy['height_ft_post_eq'] - res_test_copy['height_ft_pre_eq'])

res_train_copy.head()
res_train_copy.drop(['count_floors_pre_eq', 'height_ft_pre_eq'], axis=1, inplace=True)
res_test_copy.drop(['count_floors_pre_eq', 'height_ft_pre_eq'], axis=1, inplace=True)

remove_columns = ["legal_ownership_status","land_surface_condition","foundation_type","roof_type","ground_floor_type","other_floor_type","position","plan_configuration","count_floors_post_eq","height_ft_post_eq"]
def dropColumns(res_train_copy,res_test_copy,remove_columns):
  for i in remove_columns:
    res_train_copy.drop([i],axis = 1, inplace = True)
    res_test_copy.drop([i],axis = 1, inplace = True)
    
  return res_train_copy,res_test_copy
res_train_copy, res_test_copy = dropColumns(res_train_copy,res_test_copy,remove_columns)
res_train_copy.shape
res_test_copy.shape
res_train_copy.drop(['building_id'], axis = 1, inplace = True)
res_test_copy.drop(['building_id'], axis = 1, inplace = True)

#res_train_copy.dtypes
miss
res_train_copy['count_families'].fillna(res_train_copy['count_families'].mode()[0],inplace=True)
res_test_copy['count_families'].fillna(res_test_copy['count_families'].mode()[0],inplace=True)

print(res_train_copy['has_repair_started'].value_counts())
print(res_test_copy['has_repair_started'].value_counts())
res_train_copy['has_repair_started'].fillna(False,inplace=True)
res_test_copy['has_repair_started'].fillna(False,inplace=True)
res_train_copy['has_repair_started'] = res_train_copy['has_repair_started'].astype('int64')
res_test_copy['has_repair_started'] = res_test_copy['has_repair_started'].astype('int64')
res_train_copy['area_assesed'].unique()

res_train_copy['condition_post_eq'].unique()
y_train = res_train_copy['damage_grade']
res_train_copy.drop(['damage_grade'], axis = 1, inplace = True)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import tree
res_train_one_hot = pd.get_dummies(res_train_copy)

res_test_one_hot = pd.get_dummies(res_test_copy)
res_train_one_hot.drop(["district_id_x","vdcmun_id_x","ward_id_x"],axis = 1, inplace = True)
res_test_one_hot.drop(["district_id_x","vdcmun_id_x","ward_id_x"],axis = 1, inplace = True)
res_train_one_hot.head()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(res_train_one_hot, y_train)
y_preds_tree = clf.predict(res_test_one_hot)
prediction=pd.DataFrame({'building_id': test['building_id'], 'damage_grade':y_preds_tree})
target = {1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3', 4: 'Grade 4', 5: 'Grade 5'}
prediction.damage_grade.replace(target, inplace=True)
y_true = pd.read_csv('../input/original-submission/original-submission.csv')
sklearn.metrics.accuracy_score(y_true['damage_grade'],prediction['damage_grade']) * 100
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf = 2)
rf.fit(res_train_one_hot, y_train)
rf.score(res_train_one_hot, y_train)*100
feature_imp = pd.Series(rf.feature_importances_,index=res_train_one_hot.columns).sort_values(ascending=False)
feature_imp.head()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
ypreds = rf.predict(res_test_one_hot)
prediction=pd.DataFrame({'building_id': test['building_id'], 'damage_grade':ypreds})
target = {1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3', 4: 'Grade 4', 5: 'Grade 5'}
prediction.damage_grade.replace(target, inplace=True)
sklearn.metrics.accuracy_score(y_true['damage_grade'],prediction['damage_grade']) * 100
prediction.to_csv('submission.csv', index=False)
prediction.head()