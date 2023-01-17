# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict,Counter
from functools import partial
from scipy.stats import spearmanr,pearsonr
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import auc,make_scorer,roc_auc_score,f1_score,roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import Trials
from hyperopt import tpe
from hyperopt import fmin
import lightgbm as gbm 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)
pd.options.display.max_columns = 150
plt.style.use('fivethirtyeight')
unc_data = pd.read_csv('../input/train.csv',index_col='Id')
unc_data_test = pd.read_csv('../input/test.csv',index_col='Id')
unc_data.head()
#shape of data
print('training_data:',unc_data.shape)
print('test_data:',unc_data_test.shape)
#for checking null values
def check_null(data):
    for col in data.columns:
        na_values = data[col].isnull().sum()
        rows = data.shape[0]
        print(col,na_values,(na_values/rows)*100)
#for recalculating the dependency column
def calc_dependency(data):
    #dropping the old dependency column
    data = data.drop('dependency',axis=1)
    #recalculating the dependency
    data['dependency'] = (data.hogar_nin + data.hogar_mayor) / (data.hogar_adul - data.hogar_mayor)
    #new dependeny column calculated as per the formula with consistent entries
    data['dependency'] = data['dependency'].replace([np.inf,-np.inf],np.nan).fillna(0).round(2)
    return data
#for performing sanity checks on the categorical columns to check if they have consistent values
def sanity_check(data,full_col_list):
    for col_name,col_list in full_col_list.items():
        if data[col_list].apply(lambda x:x.sum(),axis=1).sum() == data.shape[0]:
            print('the {} columns look fine,no cleaning necessary'.format(col_name))
        else:
            print('there is a discrepancy in {} columns,requires investigation'.format(col_name))
#checking for null values--training data
print('training_data')
check_null(unc_data)
#checking for null values--test data
print('test_data')
check_null(unc_data_test)
#rez_esc column has 9139 rows where the value is either null or zero and escolari is non zero
print(unc_data[['escolari','rez_esc']][(unc_data['rez_esc'].isna()) | (unc_data['rez_esc'] == 0)])
#rez_esc column has 22916 rows where the value is either null or zero and escolari is non zero
print('test_data',unc_data_test[['escolari','rez_esc']][(unc_data_test['rez_esc'].isna()) | (unc_data_test['rez_esc'] == 0)])
#dropping the rez_esc column in training and test set
unc_data = unc_data.drop('rez_esc',axis=1)
unc_data_test = unc_data_test.drop('rez_esc',axis=1)
#dependency column needs to be cleaned as it has inconsistent entries as per the formula
unc_data['dependency'].unique()
unc_data_test['dependency'].unique()
#dependency column needs to be recalculated as per the given formula in the description.
#have checked the required columns consistency by adding up hogar_nin and hogar_adul and comparing with hogar_total
print((unc_data.hogar_nin + unc_data.hogar_adul == unc_data.hogar_total).sum())
#the three columns are consistent in test set as well,hence recalculating the dependency column
print((unc_data_test.hogar_nin + unc_data_test.hogar_adul == unc_data_test.hogar_total).sum())
#recalculating the dependency column in training set
unc_data = calc_dependency(unc_data)
#recalculating the dependency column in test set
unc_data_test = calc_dependency(unc_data_test)
#v18q1-no. of tablets owned has nan values for rows where the house does not own a tablet
#replacing the nan values with 0
unc_data['v18q1'] = unc_data['v18q1'].fillna(0)
unc_data_test['v18q1'] = unc_data_test['v18q1'].fillna(0)
#investigating the edjefe and edjefa columns which have numerical and categorical entries.
#need to reconcile these columns into one entry type,either numerical or continuous
unc_data[['edjefe','edjefa','parentesco1','male','female','escolari']]
unc_data['edjefe'].unique()
#As per the observation of these entries,the calculation of edjefe and edjefa are inconsistent
#they need to be recalculated correctly or dropped altogether to avoid inducing false/incorrect information in the dataset
unc_data[['edjefe','edjefa','parentesco1','male','female','escolari']][unc_data['edjefa']=='yes']
#dropping edjefe and edjefa columns
unc_data = unc_data.drop(['edjefe','edjefa'],axis=1)
unc_data_test = unc_data_test.drop(['edjefe','edjefa'],axis=1)
#tamhog and hhsize have the same values accross the dataset,hence can drop one of the two
print((unc_data['tamhog'] == unc_data['hhsize']).sum())
print((unc_data_test['tamhog'] == unc_data_test['hhsize']).sum())
#dropping tamhog column
unc_data = unc_data.drop('tamhog',axis=1)
unc_data_test = unc_data_test.drop('tamhog',axis=1)
#investigating the mobilephone columns to check if they are consistent with each other
print('training_data',unc_data[['mobilephone','qmobilephone']].query('mobilephone == 0 & qmobilephone != 0').sum())
print('test_data',unc_data_test[['mobilephone','qmobilephone']].query('mobilephone == 0 & qmobilephone != 0').sum())
#grouping all the columns according to the attributes they represent about the household/individual
categorical_columns = {'wall_attrib' :['paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc','paredfibras','paredother'],
                       'floor_attrib':['pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera'],
                       'roof_attrib' :['techozinc','techoentrepiso','techocane','techootro'],
                       'water_attrib':['abastaguadentro','abastaguafuera','abastaguano'],
                       'electric_attrib' : ['public','planpri','noelec','coopele'],
                       'sanitary_attrib' : ['sanitario1','sanitario2','sanitario3','sanitario5','sanitario6'],
                       'energy_attrib' : ['energcocinar1','energcocinar2','energcocinar3','energcocinar4'],
                       'disposal_attrib' : ['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'],
                       'wall_qual' : ['epared1','epared2','epared3'],
                       'roof_qual' : ['etecho1','etecho2','etecho3'],
                       'floor_qual' : ['eviv1','eviv2','eviv3'],
                       'status' : ['estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7'],
                       'edu_level' : ['instlevel1','instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9'],
                       'ownership_status' : ['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5'],
                       'region' : ['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6']
                      }
#performing sanity checks on the columns to check if they have consistent entries
sanity_check(unc_data,categorical_columns)
#we need to investigate the roof,electric and education level columns
#the same columns need to be investigated in test set as well
sanity_check(unc_data_test,categorical_columns)
#there are certain rows where all the roof columns have 0
#as per the competition host it is a small glitch which implies the roof is made out of waste material
#we need to introduce a new column which captures this information
roof_attrib = categorical_columns['roof_attrib']
unc_data[roof_attrib][unc_data.techozinc+unc_data.techoentrepiso+unc_data.techocane+unc_data.techootro != 1]
unc_data_test[roof_attrib][unc_data_test.techozinc+unc_data_test.techoentrepiso+unc_data_test.techocane+unc_data_test.techootro != 1]
#techowaste would be the new column that would be 1 for roof made out of waste material
unc_data['techowaste'] = 0
unc_data['techowaste'][unc_data.techozinc+unc_data.techoentrepiso+unc_data.techocane+unc_data.techootro == 0] = 1
unc_data_test['techowaste'] = 0
unc_data_test['techowaste'][unc_data_test.techozinc+unc_data_test.techoentrepiso+unc_data_test.techocane+unc_data_test.techootro == 0] = 1
#as per the competition host,when all the columns have zeroes,then the value should be other
#need to create a new column that would capture this information
electric_attrib = categorical_columns['electric_attrib']
unc_data[electric_attrib][unc_data.public+unc_data.planpri+unc_data.noelec+unc_data.coopele != 1]
#creating a new column elecother
unc_data['elecother'] = 0
unc_data['elecother'][unc_data.public+unc_data.planpri+unc_data.noelec+unc_data.coopele == 0] = 1
unc_data_test['elecother'] = 0
unc_data_test['elecother'][unc_data_test.public+unc_data_test.planpri+unc_data_test.noelec+unc_data_test.coopele == 0] = 1
#these are the three ids where the education level is unknown
#we do not know if they never had an education,so we cannot substitute 0
#we can introduce another level to capture this information or assume they did not have an education
edu_level = categorical_columns['edu_level']
unc_data[edu_level][unc_data[edu_level].apply(lambda x:x.sum() != 1,axis=1)]
#creating a new column instlevel0
unc_data['instlevel0'] = 0
unc_data['instlevel0'][unc_data[edu_level].apply(lambda x:x.sum() != 1,axis=1)] = 1
#these are the ids where the education level is unknownin the test set
unc_data_test['escolari'][unc_data_test[edu_level].apply(lambda x:x.sum() != 1,axis=1)]
unc_data_test['instlevel0'] = 0
unc_data_test['instlevel0'][unc_data_test[edu_level].apply(lambda x:x.sum() != 1,axis=1)] = 1
#dropping the squared_attributes.would be constructing features later during feature engineering
squared_attrib = ['SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']
unc_data = unc_data.drop(squared_attrib,axis=1)
unc_data_test = unc_data_test.drop(squared_attrib,axis=1)
#cleaning the v2a1 column which contains null values
#monthly rent payment would only be applicable to households which pay rent
#we need to check the ownership status and decide what values to substitute in this column
ownership_status = categorical_columns['ownership_status']
unc_data[ownership_status][(unc_data.v2a1.isnull()) & (unc_data.tipovivi3 != 1)].sum().plot(kind='bar')
plt.show()
unc_data_test[ownership_status][(unc_data_test.v2a1.isnull()) & (unc_data_test.tipovivi3 != 1)].sum().plot(kind='bar')
plt.show()
#substituting 0 for rows where the ownership status is 1,i.e the house is owned
unc_data.loc[(unc_data.tipovivi1 == 1),'v2a1'] = 0
unc_data_test.loc[(unc_data_test.tipovivi1 == 1),'v2a1'] = 0
#for the rest of the rows,where the rent is null,we would be imputing the values 
#and also adding a column indicating the same.
mean_rent_train = unc_data['v2a1'].mean()

unc_data['rent-missing'] = 0

unc_data['rent-missing'][unc_data.v2a1.isnull()] = 1

unc_data['v2a1']=unc_data['v2a1'].fillna(mean_rent_train)

mean_rent_test = unc_data_test['v2a1'].mean()
unc_data_test['rent-missing'] = 0
unc_data_test['rent-missing'][unc_data_test.v2a1.isnull()] = 1
unc_data_test['v2a1']=unc_data_test['v2a1'].fillna(mean_rent_test)
#the next step is to correct the labels where the members of the household,
#have a different poverty label to the head of the household
same_labels = unc_data.groupby('idhogar')['Target'].apply(lambda x:x.nunique()==1)
diff_labels = same_labels[same_labels != True]
#correcting the target labels
for idx in diff_labels.index:
    correct_label = unc_data['Target'][(unc_data.idhogar == idx) & (unc_data.parentesco1 == 1)].values[0]
    unc_data['Target'][(unc_data.idhogar == idx) & (unc_data.parentesco1 != 1)] = correct_label
#imputing the mean_educ column
edu_median = unc_data['meaneduc'].median()
unc_data['meaneduc'] = unc_data['meaneduc'].fillna(edu_median)
edu_median = unc_data_test['meaneduc'].median()
unc_data_test['meaneduc'] = unc_data_test['meaneduc'].fillna(edu_median)
#lets see the number of samples present in the dataset for each poverty level
train = unc_data
train.Target.value_counts().plot.bar()
#for plotting distributions with respect to each poverty level
def plot_distribution_cat(data,column_list):
    plt.figure(figsize = (20, 16))
    colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
    poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})
    for i,col in enumerate(column_list):
        ax = plt.subplot(4,2,i+1)
        for level,color in colors.items():
            sns.kdeplot(train.loc[train['Target'] == level, col],ax=ax,color=color,label=poverty_mapping[level])
        plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')
    plt.subplots_adjust(top = 2)

def plot_categorical(column,desc,label,kind):
    grouped = train.groupby(['Target'])[column].value_counts(normalize=True)
    grouped = grouped.rename('count')
    grouped = grouped.reset_index(['Target',column])
    if kind == 'bar':
        sns.barplot(x=column, y="count", hue="Target", data=grouped,palette=OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'}))
        if isinstance(label[0],str):
            rotation=60
        else:
            rotation=0
    elif kind == 'point':
        sns.pointplot(x=column, y="count", hue="Target", data=grouped,palette=OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'}),dodge=0.18)
        rotation=0
    plt.xlabel(desc)
    plt.ylabel('normalized_count')
    locs,labels = plt.xticks()
    plt.xticks(locs,label,rotation=rotation)
    plt.legend(loc='best')
    plt.show()
    
def create_ordinal_columns(df,feat_dict):
    for col_name,col_list in feat_dict.items():
        df[col_name] = df[col_list].apply(lambda x:np.argmax(x),axis=1,raw=True)
        print('created ordinal column {}'.format(col_name))

def corr_heatmap(column_list,method='pearson'):
    plt.figure(figsize = (8, 8))
    corr = train[column_list].corr(method)
    sns.heatmap(corr,annot=True,cmap='binary_r')
    plt.show()
#calculate spearman and pearson correlation of columns with the Target column    
def calc_sp_pr(columns):
    scorr = []
    S_p_value = []
    pcorr = []
    P_p_value = []
    for col in columns:
        scorr.append(spearmanr(train[col],train['Target']).correlation)
        pcorr.append(pearsonr(train[col],train['Target'])[0])
        S_p_value.append(spearmanr(train[col],train['Target']).pvalue)
        P_p_value.append(pearsonr(train[col],train['Target'])[1])
    return pd.DataFrame({'spearman_r':scorr,'S_p_value':S_p_value,'pearson_r':pcorr,'P_p_value':P_p_value},index=columns).sort_values('spearman_r',ascending=False)

def boxplot_distribution(column_list):
    plt.figure(figsize = (20, 16))
    colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
    poverty_mapping = ['extreme','moderate','vulnerable','non-vulnerable']
    for i,col in enumerate(column_list):
        ax = plt.subplot(4,2,i+1)
        sns.boxplot(x='Target',y=col,data=train,ax=ax,palette=colors)
        plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')
        locs,labels = plt.xticks()
        plt.xticks(locs,poverty_mapping)
    plt.subplots_adjust(top = 2)
    

#let us check the distribution of our continuous value columns
float_columns = train.select_dtypes('float')
plot_distribution_cat(train,float_columns)
#let us check if there is a difference in the number of rooms in the households
plot_categorical('rooms','No.of rooms',kind='bar',label=[i for i in range(0,13)])
plot_categorical('hogar_total','Total persons in the household',label=[i for i in range(0,20)],kind='bar')
#let us check the no. of children,adults and elderly people for each poverty level.
plot_categorical('hogar_nin','No. of children',kind='bar',label=[i for i in range(0,15)])
plot_categorical('hogar_adul','No. of adults',kind='bar',label=[i for i in range(0,15)])
plot_categorical('hogar_mayor','No. of 65+ individuals',kind='bar',label=[i for i in range(0,15)])
regions = [reg for reg in train.columns if reg.startswith('lugar')]
train['region'] = train[regions].apply(lambda x:np.argmax(x),axis=1,raw=True)
region_names=['Central','Chorotega','Pacafico central','Brunca','Huetar Atlantica','Huetar Norte']
plot_categorical('region','region',region_names,kind='bar')
train.drop('region',axis=1,inplace=True)
plot_categorical('escolari','Years of Education',label=[i for i in range(0,22)],kind='bar')
plot_categorical('v18q','Tablet',['no','yes'],kind='point')
plot_categorical('refrig','Refrigerator',['no','yes'],kind='point')
plot_categorical('mobilephone','Mobile Phone',['no','yes'],kind='point')
plot_categorical('television','TV',['no','yes'],kind='point')
plot_categorical('computer','Computer',['no','yes'],kind='point')
plot_categorical('v14a','Bathroom',['no','yes'],kind='point')

ordinal_attributes = {'wall_attrib' :['paredother','pareddes','paredfibras','paredmad','paredzinc','paredzocalo','paredblolad','paredpreb'],
                       'floor_attrib':['pisonotiene','pisoother','pisonatur','pisomadera','pisocemento','pisomoscer'],
                       'roof_attrib' :['techootro','techowaste','techocane','techozinc','techoentrepiso'],
                       'water_attrib':['abastaguano','abastaguafuera','abastaguadentro'],
                       'electric_attrib' : ['noelec','elecother','coopele','public','planpri'],
                       'sanitary_attrib' : ['sanitario1','sanitario6','sanitario5','sanitario2','sanitario3'],
                       'energy_attrib' : ['energcocinar1','energcocinar4','energcocinar3','energcocinar2'],
                       'wall_qual' : ['epared1','epared2','epared3'],
                       'roof_qual' : ['etecho1','etecho2','etecho3'],
                       'floor_qual' : ['eviv1','eviv2','eviv3'],
                       'edu_level' : ['instlevel1','instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9'],
                       'ownership_status' : ['tipovivi5','tipovivi4','tipovivi3','tipovivi2','tipovivi1'],
                       'area':['area2','area1']
                  }
create_ordinal_columns(train,ordinal_attributes)
# we need to drop the redundant attributes from the dataset
for col_list in ordinal_attributes.values():
    train.drop(col_list,axis=1,inplace=True)
#we need to drop one of the columns from [male,female] and [area1,area2] and rename area1 to  area
#male column dropped
#area2 dropped,area1 renamed to area indicating 1-->area1,0-->area2
train.drop('male',axis=1,inplace=True)
train = train.rename({'area1':'area'},axis='columns')
#### checking the correlation between household attributes
# household_level attributes
household_attributes = ['hacdor','rooms','hacapo','r4t3','tamviv','hhsize','hogar_total','bedrooms','overcrowding']
corr_heatmap(household_attributes)
#we can drop hhsize and r4t3 and keep hogar_total in our dataset
train.drop(['hhsize','r4t3'],axis=1,inplace=True)
house_attributes = ['wall_attrib','floor_attrib','roof_attrib','water_attrib','electric_attrib','sanitary_attrib','energy_attrib','wall_qual','roof_qual','floor_qual',\
'edu_level','ownership_status']
results = calc_sp_pr(house_attributes)
results[['spearman_r','pearson_r']].plot.barh()
# let us check the household attributes as well.
household_attributes = ['hacdor','rooms','hacapo','tamviv','hogar_total','bedrooms','overcrowding']
results = calc_sp_pr(household_attributes)
results[['spearman_r','pearson_r']].plot.barh()
amenities = ['v18q','refrig','mobilephone','television','computer','v14a']
results = calc_sp_pr(amenities)
results[['spearman_r','pearson_r']].plot.barh()
#diff between persons living and size of the household
#a positive score for having amenities like refrigerator,TV etc
#a pos score for living in an urban area
#score for good house condition
#households with female heads,more number of young children and elderly,living in rural areas are indicators of poverty
#school age children not attending school
# standard of living feature to assess if households have amenities like refrigerator,TV and other assets etc
#mean age of head of household
#mean edu of household head
#mean of no. of children age 12 and under,individuals age 65+ and over,mean of children under 18
#persons per room
train['person/room'] = train['rooms'] / train['hogar_total']
#proportion of males
train['prop_male'] = train['r4h3'] / train['hogar_total']
#proportion of females
train['prop_female'] = train['r4m3'] / train['hogar_total']
#rent per person
train['rent/person'] = train['v2a1'] / train['hogar_total']
#rent per room
train['rent/room'] = train['v2a1'] / train['rooms']
#single,divorced,widowed,separated
train['without_spouse'] = 0
train['without_spouse'][(train.estadocivil4 == 1) | (train.estadocivil5 == 1) | (train.estadocivil6 == 1) | (train.estadocivil7 == 1)] = 1
## if household head is female as a feature
train['female_head'] = 0
train['female_head'][(train.parentesco1 == 1) & (train.female == 1)] = 1

## proportion of children under 12
train['prop_under_12'] = train['r4t1']/train['hogar_total']
#proportion of adults
train['prop_adults'] = train['hogar_adul']/train['hogar_total']
#proportion of elderly
train['prop_elderly'] = train['hogar_mayor']/train['hogar_total']
#education of children 0-12
idx_edu_chld = train[train.age <= 12].groupby('idhogar')['escolari'].mean()
idx_edu_chld = idx_edu_chld.reset_index()
idx_edu_chld.rename(columns={"escolari": "mean_edu_child"},inplace=True)
train = train.merge(idx_edu_chld,on='idhogar',how='left')
#there would be some rows where mean_edu_child would be Nan as there would be some households that do not have children aged 0-12
#because we are doing a left join on merge,some rows would be Nan as the idhogar values on our training set would not be in idx_edu_chld 
#as the households' do not have children aged 0-12
train['mean_edu_child'].fillna(0,inplace=True)
#education of children 12-18
idx_edu_teen = train[(train.age >= 12) & ( train.age < 19)].groupby('idhogar')['escolari'].mean()
idx_edu_teen = idx_edu_teen.reset_index()
idx_edu_teen.rename(columns={"escolari": "mean_edu_teen"},inplace=True)
train = train.merge(idx_edu_teen,on='idhogar',how='left')
#there would be some rows where mean_edu_child would be Nan as there would be some households that do not have children aged 0-12
#because we are doing a left join on merge,some rows would be Nan as the idhogar values on our training set would not be in idx_edu_chld 
#as the households' do not have children aged 0-12
train['mean_edu_teen'].fillna(0,inplace=True)
#standard of living score
#it would consider electricity,water,energy used for cooking,sanitation,amenities like refrigerator,TV,mobile phone etc.
# it would have a low score 0 and a max score of 1
#since we have ordered these features,it would be easy for us to normalize and add them for a final score.
train['standard_of_living'] = 1/5 * (1/4 * train['electric_attrib'] + 1/2 * train['water_attrib'] + 1/3 * train['energy_attrib'] + 1/4 * train['sanitary_attrib']\
+ 1/5 * (train['refrig'] + train['television'] + train['v18q'] + train['mobilephone'] + train['computer']))
train['house_quality'] = 1/4 * ((1/2 * train['wall_qual']) * (1/6 * train['wall_attrib']) + (1/2 * train['floor_qual']) * (1/6 * train['floor_attrib']) + (1/2 * train['roof_qual']) * (1/6 * train['roof_attrib']) + train['cielorazo'])
## variety of gadgets
train['gadgets'] = 1/5 * (train['refrig'] + train['television'] + train['v18q'] + train['mobilephone'] + train['computer'])
## number of gadgets
train['n_gadgets'] = train['qmobilephone'] + train['v18q1']
##gadgets per person
train['gadget/person'] = train['n_gadgets'] / train['hogar_total']
train['safe_waste_disposal'] = 0
train['safe_waste_disposal'][(train.elimbasu1 == 1) | (train.elimbasu2 == 1)] = 1
#aggregating age,escolari features
def agg_features(df,col_list):
    # Define custom function
    range_ = lambda x: x.max() - x.min()
    range_.__name__ = 'range_'
    
    col_list.append('idhogar')
    
    ind = df[col_list]
    
    # Group and aggregate
    ind_agg = ind.groupby('idhogar').agg(['min', 'max', 'sum', 'mean', 'std', range_])
    ind_agg.head()
    
    new_col = []
    for c in ind_agg.columns.levels[0]:
        for stat in ind_agg.columns.levels[1]:
            new_col.append(f'{c}-{stat}')

    ind_agg.columns = new_col
    
    # Create correlation matrix
    corr_matrix = ind_agg.corr()

# Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
    
    print('no. of features to drop having correlation > 0.95:',len(to_drop))
    
    if len(to_drop) > 0:
        ind_agg = ind_agg.drop(columns = to_drop)
        
    print('dataset features shape before aggregation: ', df.shape)

    ind_agg = ind_agg.fillna(0)
    # Merge on the household id
    final = df.merge(ind_agg, on = 'idhogar', how = 'left')

    print('dataset features shape after aggregation: ', final.shape)
    
    return final
    
    
    
agg_col_list = ['age','escolari','v2a1','house_quality','standard_of_living','dependency','gadgets','gadget/person','n_gadgets']
train = agg_features(train,agg_col_list)
#we would be dropping the waste disposal columns as we have already created a new column indicating the same.
columns = ['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6']
train.drop(columns,axis=1,inplace=True)

#dropping columns other than parentesco1
columns = ['parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12']
train.drop(columns,axis=1,inplace=True)
columns = ['estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7']
train.drop(columns,axis=1,inplace=True)

#No. of columns after feature engineering
print(train.shape)
train.index = unc_data.index
cont_variables = ['prop_adults','prop_under_12','prop_elderly','mean_edu_child','mean_edu_teen','standard_of_living','house_quality']
plot_distribution_cat(train,cont_variables)
plot_categorical('safe_waste_disposal','waste_disposal',['unsafe','safe'],kind='point')
plot_categorical('female_head','female_head',['no','yes'],kind='point')
new_features = ['female_head','prop_under_12','prop_adults','prop_elderly','mean_edu_child','mean_edu_teen','standard_of_living','house_quality','safe_waste_disposal']
results = calc_sp_pr(new_features)
results[['spearman_r','pearson_r']].plot.barh()
col_list = ['prop_under_12','prop_adults','mean_edu_child','mean_edu_teen','standard_of_living','house_quality']
boxplot_distribution(col_list)
test = unc_data_test.copy()
test.head()
test.drop(['hhsize','r4t3'],axis=1,inplace=True)
create_ordinal_columns(test,ordinal_attributes)
for col_list in ordinal_attributes.values():
    test.drop(col_list,axis=1,inplace=True)
test.drop('male',axis=1,inplace=True)
test = test.rename({'area1':'area'},axis='columns')
#persons per room
test['person/room'] = test['rooms'] / test['hogar_total']
#proportion of males
test['prop_male'] = test['r4h3'] / test['hogar_total']
#proportion of females
test['prop_female'] = test['r4m3'] / test['hogar_total']
#rent per person
test['rent/person'] = test['v2a1'] / test['hogar_total']
#rent per room
test['rent/room'] = test['v2a1'] / test['rooms']
#single,divorced,widowed,separated
test['without_spouse'] = 0
test['without_spouse'][(test.estadocivil4 == 1) | (test.estadocivil5 == 1) | (test.estadocivil6 == 1) | (test.estadocivil7 == 1)] = 1
#if household head is female
test['female_head'] = 0
test['female_head'][(test.parentesco1 == 1) & (test.female == 1)] = 1
## proportion of children under 12
test['prop_under_12'] = test['r4t1']/test['hogar_total']
#proportion of adults
test['prop_adults'] = test['hogar_adul']/test['hogar_total']
#proportion of elderly
test['prop_elderly'] = test['hogar_mayor']/test['hogar_total']
#education of children 0-12
idx_edu_chld = test[test.age <= 12].groupby('idhogar')['escolari'].mean()
idx_edu_chld = idx_edu_chld.reset_index()
idx_edu_chld.rename(columns={"escolari": "mean_edu_child"},inplace=True)
test = test.merge(idx_edu_chld,on='idhogar',how='left')
test['mean_edu_child'].fillna(0,inplace=True)
#education of children 12-18
idx_edu_teen = test[(test.age >= 12) & (test.age <= 18)].groupby('idhogar')['escolari'].mean()
idx_edu_teen = idx_edu_teen.reset_index()
idx_edu_teen.rename(columns={"escolari": "mean_edu_teen"},inplace=True)
test = test.merge(idx_edu_teen,on='idhogar',how='left')
test['mean_edu_teen'].fillna(0,inplace=True)
#standard of living score
test['standard_of_living'] = 1/5 * (1/4 * test['electric_attrib'] + 1/2 * test['water_attrib'] + 1/3 * test['energy_attrib'] + 1/4 * test['sanitary_attrib']\
+ 1/5 * (test['refrig'] + test['television'] + test['v18q'] + test['mobilephone'] + test['computer']))
#house quality score
test['house_quality'] = 1/4 * ((1/2 * test['wall_qual']) * (1/6 * test['wall_attrib']) + (1/2 * test['floor_qual']) * (1/6 * test['floor_attrib']) + (1/2 * test['roof_qual']) * (1/6 * test['roof_attrib']) + test['cielorazo'])
## variety of gadgets
test['gadgets'] = 1/5 * (test['refrig'] + test['television'] + test['v18q'] + test['mobilephone'] + test['computer'])
## number of gadgets
test['n_gadgets'] = test['qmobilephone'] + test['v18q1']
##gadgets per person
test['gadget/person'] = test['n_gadgets'] / test['hogar_total']
#safe waste disposal
test['safe_waste_disposal'] = 0
test['safe_waste_disposal'][(test.elimbasu1 == 1) | (test.elimbasu2 == 1)] = 1
agg_col_list = ['age','escolari','v2a1','house_quality','standard_of_living','dependency','gadgets','gadget/person','n_gadgets']
test = agg_features(test,agg_col_list)
#we would be dropping the waste disposal columns as we have already created a new column indicating the same.
columns = ['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6']
test.drop(columns,axis=1,inplace=True)
#dropping columns other than parentesco1
columns = ['parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12']
test.drop(columns,axis=1,inplace=True)
#dropping relationship columns
columns = ['estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7']
test.drop(columns,axis=1,inplace=True)
test.index = unc_data_test.index
#We would only be using samples or rows of data where parentesco1 is 1
#i.e we would only be using data pertaining to head of households
train_head = train[train.parentesco1 == 1].reset_index()
test = test.reset_index()
print(train_head.shape)
print(test.shape)
#preparing the submission dataframe
submission_base = test[['Id', 'idhogar']].copy()
test_ids = test['idhogar']

train_head.drop(['Id','idhogar'],axis=1,inplace=True)
test.drop(['Id','idhogar'],axis=1,inplace=True)
feat_cols = train_head.columns.difference(['Target'])
train_feat = train_head[feat_cols].values
#scaling the features to have 0 mean and unit variance
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_feat)
test_scaled = scaler.fit_transform(test)
features = train_scaled
labels = train_head['Target']
#As we have seen,we have class imbalance in our dataset,which needs to be taken care of,
#hence,we would be oversampling our minority classes using SMOTE algorithm
#This algorithm creates synthetic samples from our dataset and upsamples the minority classes.
#creating the dataset to be used in LightGBM
feature_names = list(train_head.columns.difference(['Target']))
train_set = gbm.Dataset(features, label = labels,feature_name=feature_names)
def avg_roc(multi_class_scores):
    result = list()
    for i in range(1,5):
        roc_class = np.ravel(multi_class_scores[i])
        result.append(np.mean(np.ravel(roc_class)))
    avg_roc = np.mean(result)
    return avg_roc
    #print('average roc-auc value for each class')
    #print('-' * 50)
    #poverty_level = {1:'extreme',2:'moderate',3:'vulnerable',4:'non-vulnerable'}
    #for i in range(0,4):
        #print(poverty_level[i+1],' class--',result[i])
    #print('-' * 50)
    #print('overall average roc-auc value-{}'.format(np.round(avg_roc,3)))
    
def plot_ROC(fpr,tpr,n_classes):
    
    colors = ['red', 'yellow', 'blue','green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.0,
                 label='ROC of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=0.25)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="best")
    plt.show()

def compare_distributions(param):
    uniform_C = list()
    for i in range(0,200):
        x = sample(space)
        uniform_C.append(x[param])

    sns.kdeplot(uniform_C, label = 'uniform-dist')
    sns.kdeplot(results[param], label = 'Bayes Optimization')
    plt.legend(loc = 'best')
    plt.title('{} Distribution'.format(param))
    plt.xlabel('{}'.format(param)); plt.ylabel('Density');
    plt.show();

def roc_auc_score_proba(y_true, proba):
    return roc_auc_score(y_true, proba[:, 1])

# define your scorer
roc_auc_weighted = make_scorer(roc_auc_score, average='weighted')

def macro_f1_score(labels,predictions):
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True

def plot_feature_importances(df, n = 10, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    plt.style.use('fivethirtyeight')
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    return df

#domain space
'''space = {
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}'''
'''N_FOLDS=5
def gbm_objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    #y_bin = label_binarize(Y_resampled, classes=[1,2,3,4])
    #n_classes = y_bin.shape[1]
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
        
    strkfold = StratifiedKFold(n_splits = n_folds, shuffle = True)
    
    clf = gbm.LGBMClassifier(**params,n_estimators=1000,objective='multiclass',importance_type='gain',n_jobs=-1,metric=None)
    
    valid_scores = []
    best_estimators = []
    run_times = []
    
    # Perform n_folds cross validation
    for train_indices, valid_indices in strkfold.split(features, labels):
        
        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        
        class_dist = Counter(y_train)
        ratio_ = {4:class_dist[4],3:class_dist[3]*2,2:class_dist[2]*2,1:class_dist[1]*2}
        
        sm = SMOTE(ratio=ratio_,random_state=50)
        X_train_res,Y_train_res = sm.fit_sample(X_train,y_train)
    
        start = timer()
        
        clf.fit(X_train_res, Y_train_res, early_stopping_rounds = 400, 
                  eval_metric = macro_f1_score, 
                  eval_set = [(X_train_res, Y_train_res), (X_valid, y_valid)],
                  eval_names = ['train', 'valid'],
                  verbose = 400)
        
        end = timer()
        # Record the validation fold score
        valid_scores.append(clf.best_score_['valid']['macro_f1'])
        best_estimators.append(clf.best_iteration_)
        
        run_times.append(end - start)
    
    score = np.mean(valid_scores)
    score_std = np.std(valid_scores)
    loss = 1 - score
    
    run_time = np.mean(run_times)
    run_time_std = np.std(run_times)
    
    estimators = int(np.mean(best_estimators))
    params['n_estimators'] = estimators
    
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'time': run_time, 'time_std': run_time_std, 'status': STATUS_OK, 
            'score': score, 'score_std': score_std}'''

#tpe_algorithm = tpe.suggest
#gbm_bayes_trials = Trials()
'''%%capture

# Global variable
global  ITERATION

ITERATION = 0
MAX_EVALS = 100

# Run optimization
best = fmin(fn = gbm_objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = gbm_bayes_trials, rstate = np.random.RandomState(50))'''
'''gbm_results = gbm_bayes_trials.results
#converting the results to dataframe
dt = sorted([{'loss':trial['loss'],'boosting_type':trial['params']['boosting_type'],'colsample':trial['params']['colsample_bytree'],\
             'min_child_samples':trial['params']['min_child_samples'],'num_leaves':trial['params']['num_leaves'],'reg_alpha':trial['params']['reg_alpha'],'reg_lambda':trial['params']['reg_lambda'],\
             'subsample_for_bin':trial['params']['subsample_for_bin'],'subsample':trial['params']['subsample'],'train_time':trial['train_time']} for trial in gbm_results],key=lambda x:(x['loss'],x['train_time']))
results = pd.DataFrame(dt)
results.head()'''
hyper =  {'bagging_fraction': 0.9410068419634143,
 'boosting_type': 'dart',
 'colsample_bytree': 0.7568976937851579,
 'min_child_samples': 35,
 'min_child_weight': 26.709811008563438,
 'min_split_gain': 0.05365377666160257,
 'num_leaves': 24,
 'reg_alpha': 1.0462496845733886,
 'reg_lambda': 0.6874474257041001,
 'subsample': 0.6621345483522493,
 'subsample_for_bin': 240000}
feature_names = list(train_head.columns.difference(['Target']))
from IPython.display import display

def model_gbm(features, labels, test_features, test_ids, 
              nfolds = 5, return_preds = False, hyp = None):
    """Model using the GBM and cross validation.
       Trains with early stopping on each fold.
       Hyperparameters probably need to be tuned."""
    
    feat_names = feature_names

    # Option for user specified hyperparameters
    if hyp is not None:
        # Using early stopping so do not need number of esimators
        if 'n_estimators' in hyp:
            del hyp['n_estimators']
        params = hyp
    
    else:
        # Model hyperparameters
        params = hyper
    
    # Build the model
    model = gbm.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)
    
    # Using stratified kfold cross validation
    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)
    
    # Hold all the predictions from each fold
    predictions = pd.DataFrame()
    importances = np.zeros(len(feat_names))
    
    # Convert to arrays for indexing
    #features = np.array(features)
    #test_features = np.array(test_features)
    #labels = np.array(labels).reshape((-1 ))
    
    valid_scores = []
    
    # Iterate through the folds
    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):
        
        # Dataframe for fold predictions
        fold_predictions = pd.DataFrame()
        
        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        
        class_dist = Counter(y_train)
        ratio_ = {4:class_dist[4],3:class_dist[3]*2,2:class_dist[2]*2,1:class_dist[1]*2}
        
        sm = SMOTETomek(random_state=50)
        X_train_res,Y_train_res = sm.fit_sample(X_train,y_train)
        
        # Train with early stopping
        model.fit(X_train_res, Y_train_res, early_stopping_rounds = 400, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train_res, Y_train_res), (X_valid, y_valid)],
                  eval_names = ['train', 'valid'],
                  verbose = 200)
        
        # Record the validation fold score
        valid_scores.append(model.best_score_['valid']['macro_f1'])
        
        # Make predictions from the fold as probabilities
        fold_probabilitites = model.predict_proba(test_features)
        
        # Record each prediction for each class as a separate column
        for j in range(4):
            fold_predictions[(j + 1)] = fold_probabilitites[:, j]
            
        # Add needed information for predictions 
        fold_predictions['idhogar'] = test_ids
        fold_predictions['fold'] = (i+1)
        
        # Add the predictions as new rows to the existing predictions
        predictions = predictions.append(fold_predictions)
        
        # Feature importances
        importances += model.feature_importances_ / nfolds   
        
        # Display fold information
        display(f'Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}')

    # Feature importances dataframe
    feature_importances = pd.DataFrame({'feature': feat_names,
                                        'importance': importances})
    
    valid_scores = np.array(valid_scores)
    display(f'{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}.')
    
    # If we want to examine predictions don't average over folds
    if return_preds:
        predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
        predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
        return predictions, feature_importances
    
    # Average the predictions over folds
    predictions = predictions.groupby('idhogar', as_index = False).mean()
    
    # Find the class and associated probability
    predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
    predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
    predictions = predictions.drop(columns = ['fold'])
    
    # Merge with the base to have one prediction for each individual
    submission = submission_base.merge(predictions[['idhogar', 'Target']], on = 'idhogar', how = 'left').drop(columns = ['idhogar'])
        
    # Fill in the individuals that do not have a head of household with 4 since these will not be scored
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
    
    # return the submission and feature importances along with validation scores
    return submission, feature_importances, valid_scores
%%capture
submission, gbm_fi, valid_scores = model_gbm(features, labels, 
                                             test_scaled, test_ids, return_preds=False)
np.mean(valid_scores)

submission.index = submission.Id
submission.drop('Id',axis=1,inplace=True)
submission.to_csv('gbm_baseline_13.csv')
_ = plot_feature_importances(gbm_fi, threshold=0.95)
