# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
sns.set()
# load dataset

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.shape
test_df.shape
train_df.columns.values
train_df.head()
train_df.isnull().sum()[train_df.isnull().sum() > 0]
import missingno as msno
msno.matrix(train_df[['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']], color = (0.211, 0.215, 0.274))
plt.show()
sns.countplot(train_df.loc[(pd.isnull(train_df['v2a1'])), 'tipovivi1'])
plt.title("Ownership")
plt.show()
train_df.loc[(pd.isnull(train_df['v2a1']) & train_df['tipovivi1'] == 1), 'v2a1'] = 0
train_df = train_df.loc[pd.notnull(train_df['v2a1'])]
train_df['v18q1'].dropna().value_counts()
train_df.loc[(pd.isnull(train_df['v18q1'])), 'v18q1'] = 0
train_df['rez_esc'].dropna().value_counts()
# statistical measures of those with rez_esc

train_df.loc[pd.notnull(train_df['rez_esc']),('age')].describe()
# statistical measures of those with missing rez_esc

train_df.loc[pd.isnull(train_df['rez_esc']),('age')].describe()
train_df.drop(columns='rez_esc', inplace = True)
train_df.loc[pd.isnull(train_df['meaneduc']), ('edjefa', 'edjefe', 'escolari', 'meaneduc')]
train_df.loc[pd.isnull(train_df['meaneduc']), 'meaneduc'] = train_df.loc[pd.isnull(train_df['meaneduc']), 'escolari']
train_df.loc[pd.isnull(train_df['SQBmeaned']), 'SQBmeaned'] = train_df.loc[pd.isnull(train_df['SQBmeaned']), 'meaneduc']**2
len(train_df.loc[train_df['age'] == 0].index)
train_df = train_df.loc[train_df['age']!=0]
msno.matrix(train_df[['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']], color = (0.211, 0.215, 0.274))
plt.show()
train_df.groupby('dependency').size()
mode = train_df.loc[(train_df['dependency'] != 'yes') & (train_df['dependency'] != 'no'), 'dependency'].astype(float).mode()
mode
def mutate_columns(df):
    df.loc[df['dependency']=='no', 'dependency'] = 0
    df.loc[df['dependency']=='yes', 'dependency'] = 0.5 # TODO computation of dependency
    df['dependency'] = df['dependency'].astype('float16')
    
    #The same applies with edjefe and edjefa EXCEPT that the 'yes' value doesn't make any sense? Does it to you? 
    #Anyway, let's just impute meaneduc for 'yes' values
    
    df.loc[df['edjefe']=='no', 'edjefe'] = 0
    df.loc[df['edjefe']=='yes', 'edjefe'] = df['meaneduc']
    df['edjefe'] = df['edjefe'].astype('uint8')
    
    df.loc[df['edjefa']=='no', 'edjefa'] = 0
    df.loc[df['edjefa']=='yes', 'edjefa'] = df['meaneduc']
    df['edjefa'] = df['edjefa'].astype('uint8')
mutate_columns(train_df)
mutate_columns(test_df)

plt.figure(figsize = (10,5))
sns.countplot(x='Target', data=train_df, palette="OrRd_r")
plt.xticks([0,1,2,3],['extreme poverty','moderate poverty','vulnerable households','non vulnerable households'])
plt.xlabel('')
plt.ylabel('')
plt.title("Poverty Levels", fontsize = 14)

plt.show()
tdf = train_df[['Target']]
n_train_df = train_df
for col in ['v18q', 'refrig', 'computer', 'television', 'mobilephone', 'v14a']:
    n_train_df[col] = n_train_df[col].astype('category')
dfcat = pd.get_dummies(n_train_df[[ 'v18q', 'refrig', 'computer', 'television', 'mobilephone', 'v14a']])
df_ = pd.concat([dfcat, tdf], axis=1)

df_ = df_.groupby(['Target']).sum()
df_.reset_index(inplace = True)
plt.figure(figsize=(12,4))
groups = ['extreme','moderate','vulnerable','non-vulnerable']

ordered_df = df_.sort_values(by='Target')
my_range=range(1,len(df_.index)+1)


plt.scatter(ordered_df['v18q_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['v18q_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['v18q_1'], xmax=ordered_df['v18q_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Tablet ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['refrig_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['refrig_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['refrig_1'], xmax=ordered_df['refrig_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Refrigerator ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['computer_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['computer_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['computer_1'], xmax=ordered_df['computer_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Computer ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['television_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['television_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['television_1'], xmax=ordered_df['television_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Television ownership", fontsize = 14)
plt.show()

plt.figure(figsize=(12, 4))

plt.scatter(ordered_df['mobilephone_1'], my_range, color='#0055a4', label='Present', s=200)
plt.scatter(ordered_df['mobilephone_0'], my_range, color='#9b1d20' , label='Not present', s=200)
plt.legend(loc = 4, prop={'size': 10})
plt.hlines(y=my_range, xmin=ordered_df['mobilephone_1'], xmax=ordered_df['mobilephone_0'], alpha=0.5)
plt.yticks(np.arange(1,5),groups)
plt.xlabel("Number of household")
plt.title("Mobile phone ownership", fontsize = 14)
plt.show()

df_ = train_df[['Target', 'male', 'female', 'age']]
df_.loc[(train_df['male'] == 1), 'sex'] = 'male'
df_.loc[(train_df['female'] == 1), 'sex'] = 'female'

plt.figure(figsize = (10,8))
sns.violinplot(x='Target',y='age', data=df_, hue='sex', split=True)
plt.xticks(np.arange(0,5),groups)
plt.show()

plt.figure(figsize = (15,15))
gs = gridspec.GridSpec(4, 2, hspace=0.3)

plt.subplot(gs[0,0])
g = sns.countplot(train_df['r4h3'], hue=train_df['Target'], color="#3274a1")
plt.title("Total males in a household", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
legend = g.get_legend()
legend.set_title("Income group")
new_labels = ['extreme', 'moderate', 'vulnerable', 'non-vulnerable']
for t, l in zip(legend.texts, new_labels): t.set_text(l)

plt.subplot(gs[0,1])
g = sns.countplot(train_df['r4m3'], hue=train_df['Target'], color="#d32d41")
plt.title("Total females in a household", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
legend = g.get_legend()
legend.set_title("Income group")
new_labels = ['extreme', 'moderate', 'vulnerable', 'non-vulnerable']
for t, l in zip(legend.texts, new_labels): t.set_text(l)

plt.subplot(gs[1,0]) 
sns.countplot(train_df['r4h1'], hue=train_df['Target'], color="#3274a1")
plt.title("Males < 12 years old", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
plt.legend('')


plt.subplot(gs[1,1]) 
sns.countplot(train_df['r4m1'], hue=train_df['Target'], color="#d32d41")
plt.title("Females < 12 years old", fontsize = 14)
plt.xlabel('')
plt.legend('')

plt.subplot(gs[2,0]) 
sns.countplot(train_df['r4h2'], hue=train_df['Target'], color="#3274a1")
plt.title("Males >= 12 years old", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
plt.legend('')

plt.subplot(gs[2,1]) 
sns.countplot(train_df['r4m2'], hue=train_df['Target'], color="#d32d41")
plt.title("Females >= 12 years old", fontsize = 14)
plt.xlabel('')
plt.ylabel('')
plt.legend('')

plt.show()
df_q = train_df[['Target', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']]
df_q.loc[df_q['epared1'] == 1, 'wall'] = 'Bad'
df_q.loc[df_q['epared2'] == 1, 'wall'] = 'Regular'
df_q.loc[df_q['epared3'] == 1, 'wall'] = 'Good'

df_q.loc[df_q['etecho1'] == 1, 'roof'] = 'Bad'
df_q.loc[df_q['etecho2'] == 1, 'roof'] = 'Regular'
df_q.loc[df_q['etecho3'] == 1, 'roof'] = 'Good'

df_q.loc[df_q['eviv1'] == 1, 'floor'] = 'Bad'
df_q.loc[df_q['eviv2'] == 1, 'floor'] = 'Regular'
df_q.loc[df_q['eviv3'] == 1, 'floor'] = 'Good'

df_q = df_q[['Target', 'wall', 'roof', 'floor']]
print("Roof quality")
print("==============================================================================================================================")
df_q.loc[df_q['Target'] == 1, 'Target'] = 'Extreme'
df_q.loc[df_q['Target'] == 2,'Target'] = 'Moderate'
df_q.loc[df_q['Target'] == 3,'Target'] = 'Vulnerable'
df_q.loc[df_q['Target'] == 4,'Target'] = 'Non-Vulnerable'
ax = sns.catplot(x = 'roof', col = 'Target', data = df_q, kind="count", col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable']).set_titles("{col_name}")
ax.fig.set_size_inches(15,4)
ax.set(ylabel = '')
plt.show()

print("Wall quality")
print("==============================================================================================================================")

ax = sns.catplot(x = 'wall', col = 'Target', data = df_q, kind="count" ,col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable'], order = ['Bad', 'Regular', 'Good']).set_titles("{col_name}")
ax.fig.set_size_inches(15,4)
ax.set(ylabel = '')
plt.show()

print("Floor quality")
print("==============================================================================================================================")

ax = sns.catplot(x = 'floor', col = 'Target', data = df_q, kind="count", col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable']).set_titles("{col_name}")
ax.fig.set_size_inches(15,4)
ax.set(ylabel = '')
plt.show()
# bars1 = [12, 28, 1, 8, 22]
# bars2 = [28, 7, 16, 4, 10]
# bars3 = [25, 3, 23, 25, 17]
 
# # Heights of bars1 + bars2 (TO DO better)
# bars = [40, 35, 17, 12, 32]
 
# # The position of the bars on the x-axis
# r = [0,1,2,3,4]
 
# # Names of group and bar width
# names = ['A','B','C','D','E']
# barWidth = 1
 
# # Create brown bars
# plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# # Create green bars (middle), on top of the firs ones
# plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
# # Create green bars (top)
# plt.bar(r, bars3, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)
 
# # Custom X axis
# plt.xticks(r, names, fontweight='bold')
# plt.xlabel("group")


plt.figure(figsize = (10,6))
plt.subplot(111)
sns.boxplot(x = 'Target', y = 'overcrowding', data = train_df)
plt.title('Person per room')
plt.xlabel('')
plt.ylabel('')
plt.xticks(np.arange(0,4), ['extreme','moderate','vulnerable','non-vulnerable'])
plt.show()
temp_train = train_df
temp_test = test_df
train_df = temp_train
test_df = temp_test
def feature_engineer(x):
    
    x['refrig'] = x['refrig'].astype(int)
    x['computer'] = x['computer'].astype(int)
    x['television'] = x['television'].astype(int)
    x['mobilephone'] = x['mobilephone'].astype(int)
    x['v18q'] = x['v18q'].astype(int)
    x['epared1'] = x['epared1'].astype(int)
    x['epared2'] = x['epared2'].astype(int)
    x['epared3'] = x['epared3'].astype(int)
    x['etecho1'] = x['etecho1'].astype(int)
    x['etecho2'] = x['etecho2'].astype(int)
    x['etecho3'] = x['etecho3'].astype(int)
    x['eviv1'] = x['eviv1'].astype(int)
    x['eviv2'] = x['eviv2'].astype(int)
    x['eviv3'] = x['eviv3'].astype(int)
    x['abastaguadentro'] = x['abastaguadentro'].astype(int)
    x['abastaguafuera'] = x['abastaguafuera'].astype(int)
    x['abastaguano'] = x['abastaguano'].astype(int)
    x['abastaguano'] = x['abastaguano'].astype(int)
    x[['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']] = x[['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']].apply(pd.to_numeric)
    
    x['appliances'] = (x['refrig'] + x['computer'] + x['television'])

    x['rent_by_hhsize'] = x['v2a1'] / x['hhsize'] # rent by household size
    x['rent_by_people'] = x['v2a1'] / x['r4t3'] # rent by people in household
    x['rent_by_rooms'] = x['v2a1'] / x['rooms'] # rent by number of rooms
    x['rent_by_living'] = x['v2a1'] / x['tamviv'] # rent by number of persons living in the household
    x['rent_by_minor'] = x['v2a1'] / x['hogar_nin']
    x['rent_by_adult'] = x['v2a1'] / x['hogar_adul']
    x['rent_by_dep'] = x['v2a1'] / x['dependency']
    x['rent_by_head_educ'] = x['v2a1'] / (x['edjefe'] + x['edjefa'])
    x['rent_by_educ'] = x['v2a1'] / x['meaneduc']
    x['rent_by_numPhone'] = x['v2a1'] / x['qmobilephone']
    x['rent_by_gadgets'] = x['v2a1'] / (x['computer'] + x['mobilephone'] + x['v18q'])
    x['rent_by_num_gadgets'] = x['v2a1'] / (x['v18q1'] + x['qmobilephone'])
    x['rent_by_appliances'] = x['v2a1'] / x['appliances']
    
    x['tablet_density'] = x['v18q1'] / x['r4t3']
    x['phone_density'] = x['qmobilephone'] / x['r4t3']
    
    x['wall_qual'] = x['epared3'] - x['epared1']
    x['roof_qual'] = x['etecho3'] - x['etecho1']
    x['floor_qual'] = x['eviv3'] - x['eviv1']
    x['water_qual'] = x['abastaguadentro'] - x['abastaguano']
    
    x['house_qual'] = x['wall_qual'] + x['roof_qual'] + x['floor_qual']
    
    x['person_per_room'] = x['hhsize'] / x['rooms']
    x['person_per_appliances'] = x['hhsize'] / x['appliances']
    
    x['educ_qual'] = (1 * x['instlevel1']) + (2 * x['instlevel2']) + (3 * x['instlevel3']) + (4 * x['instlevel4']) + (5 * x['instlevel5']) + (6 * x['instlevel6']) + ( 7 * x['instlevel7']) + (8 * x['instlevel8']) + (9 * x['instlevel9'])
    
    def reverse_label_encoding(row, df):
        for c in df.columns:
            if row[c] == 1:
                return int(c[-1])
            
    def rate_sanitary(row, df):
        c = df.columns.tolist()[0]
        
        if row[c] == 'sanitario2':
            return 3
        elif row[c] == 'sanitario3':
            return 2
        elif row[c] == 'sanitario5':
            return 1
        else:
            return 0
        
    def rate_cooking(row, df):
        c = df.columns.tolist()[0]
        
        if row[c] == 'energcocinar2':
            return 3
        elif row[c] == 'energcocinar3':
            return 2
        elif row[c] == 'energcocinar4':
            return 1
        else:
            return 0
        
    def rate_rubbish(row, df):
        c = df.columns.tolist()[0]
        
        if row[c] == 'elimbasu1':
            return 1
        elif row[c] == 'elimbasu2':
            return 2
        else:
            return 0
            
    x['sanitary'] = x.apply(lambda q: reverse_label_encoding(q, x[['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']]), axis=1)
    x['cooking'] =  x.apply(lambda q: reverse_label_encoding(q, x[['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']]), axis=1)
    x['rubbish'] = x.apply(lambda q: reverse_label_encoding(q, x[['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']]), axis=1)
    x['region'] = x.apply(lambda q: reverse_label_encoding(q, x[['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']]), axis=1)
    
    x['sanitary_i'] = x.apply(lambda q: rate_sanitary(q, x[['sanitary']]), axis = 1)
    x['cooking_i'] = x.apply(lambda q: rate_cooking(q, x[['cooking']]), axis = 1)
    x['rubbish_i'] = x.apply(lambda q: rate_rubbish(q, x[['rubbish']]), axis = 1)
    
    x['zone'] = x['area1'] - x['area2']

    x.replace([np.inf, -np.inf], 0, inplace = True)
    x.fillna(0, inplace = True)
feature_engineer(train_df)
feature_engineer(test_df)
# def agg_features(y):
#     agg_feat = ['hacdor', 'v18q1', 'dis', 'r4h3', 'r4m3', 'age', 'hogar_nin', 'hogar_adul', 'hogar_total', 'dependency',
#                 'appliances', 'phone_density', 'tablet_density', 'house_qual', 'person_per_appliances', 'educ_qual'
#                ]
#     # https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
#     for group in ['idhogar', 'zone', 'region']:
#         for feat in agg_feat:
#             for agg_m in ['mean','sum']:
#                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()[[feat, group]]
# #                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()
#                 new_col = feat + '_' + agg_m + '_' + group 
#                 id_agg.rename(columns = {feat : new_col} , inplace = True)
#                 y = y.merge(id_agg, how = 'left', on = group)

    
#     drop_ = ['sanitary', 'cooking', 'rubbish', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
#             'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',
#             'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
    
#     y.drop((drop_), inplace = True, axis = 1)
#     y.replace([np.inf, -np.inf], 0, inplace = True)
#     y.fillna(0, inplace = True)
#     return y
def agg_features(y):
    mean_list = ['rez_esc', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco2',
             'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
    o_list = ['escolari', 'age', 'escolari_age']
        
    for group in ['idhogar', 'zone', 'region']:
        for feat in mean_list:
            for agg_m in ['mean']:
                id_agg = y[feat].groupby(y[group]).agg(agg_m)#.reset_index()[[feat, group]]
#                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()
                new_col = feat + '_' + agg_m + '_' + group 
                #id_agg.rename(columns = {feat : new_col} , inplace = True)
                #y = y.merge(id_agg, how = 'left', on = group)
                y[new_col] = id_agg
                
    for item in o_list:
        for agg_m in ['mean','std','min','max','sum']:
            id_agg = y[feat].groupby(y[group]).agg(agg_m)#.reset_index()[[feat, group]]
#                 id_agg = y[feat].groupby(y[group]).agg(agg_m).reset_index()
            new_col = feat + '_' + agg_m + '_' + group 
                #id_agg.rename(columns = {feat : new_col} , inplace = True)
                #y = y.merge(id_agg, how = 'left', on = group)
            y[new_col] = id_agg
                
                
    return y
train_df = agg_features(train_df)
test_df = agg_features(test_df)
train_df.fillna(value=0, inplace=True)
train_df = train_df.loc[train_df['parentesco1'] == 1]
y = train_df[['Target']]
x = train_df.drop(['Target','Id','idhogar'], axis = 1)

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
clf = RandomForestClassifier()
clf.fit(x, y)

imp = clf.feature_importances_
name = np.array(x.columns.values.tolist())


df_imp = pd.DataFrame({'feature':name, 'importance':imp})
df_imp = df_imp.sort_values(by='importance', ascending=False)
plt.figure(figsize=(8,20))
sns.barplot(df_imp.loc[(df_imp['importance'] > 0.01),'importance'], y = df_imp.loc[(df_imp['importance'] > 0.01),'feature'])
plt.title('Important features')
plt.show()
important_cols = df_imp['feature']
x_ = x[important_cols]
# plt.figure(figsize = (20,16))
# sns.heatmap(x_.corr(), cmap='YlOrRd')
# plt.show()
# x_ = x[['meaneduc', 'dependency', 'person_per_room', 'qmobilephone', 'overcrowding', 'hogar_nin', 'age', 'r4t2', 'rooms', 'cielorazo', 'r4h3', 'r4h2', 'r4m3', 'v2a1', 'rent_by_hhsize', 'r4t1', 'escolari', 'v18q', 'r4m1', 'bedrooms', 'edjefe', 'eviv3', 'epared3', 'hogar_adul', 'etecho3', 'r4m2', 'tamviv']]
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_ = scaler.fit_transform(x_)
from sklearn.model_selection import train_test_split

features = [c for c in x_.columns if c not in ['Target']]
target = train_df[['Target']]
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_sample(x_, target)
sns.countplot(y_resampled)
plt.show()
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled,  y_resampled, test_size=0.1, random_state=1)
print("foo")
import xgboost as xgb
model = xgb.XGBClassifier( objective='multiclass', n_jobs=4, n_estimators=5000, class_weight='balanced', learning_rate=0.1, scale_pos_weight = 1).fit(X_resampled, y_resampled.ravel())
y_trainpred_xg = model.predict(X_train)
y_testpred_xg = model.predict(X_valid)
def evaluate(y_train, y_valid, y_trainpred, y_testpred):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score

    print("MAE :: ", mean_absolute_error(y_train, y_trainpred))
    print("Train Accuracy :: ", accuracy_score(y_train, y_trainpred)*100) 
    print("Test Accuracy  :: ", accuracy_score(y_valid, y_testpred) * 100)
    print("Confusion matrix :: \n", confusion_matrix(y_valid, y_testpred))
evaluate(y_train, y_valid, y_trainpred_xg, y_testpred_xg)
predict = model.predict(test_df[features].values)
predict.shape
submission = pd.DataFrame()
submission['Id'] = test_df['Id']
submission['Target'] = predict
submission.to_csv('submissions.csv', index=False)
