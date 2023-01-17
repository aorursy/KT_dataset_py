import numpy as np 
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('../input/train.csv')
print(train_data.info())
train_data.describe()
train_data.columns[train_data.dtypes==object]
print(train_data.dependency.unique())
print(train_data.edjefe.unique())
print(train_data.edjefa.unique())

households = train_data.groupby('idhogar').apply(lambda x: len(x))
print(households.describe())
plt.hist(households, bins=range(1, 13), align='left')
plt.xlabel("Number of household's members")
plt.ylabel('Number of households')
plt.grid(True)
plt.xlim([1, 13])
plt.xticks(range(1, 14))
plt.show()
train_data_na = (train_data.isnull().sum() / len(train_data)) * 100
train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_data_na})
f, ax = plt.subplots(figsize=(14, 6))
plt.xticks(rotation='90')
sns.barplot(x=train_data_na.index, y=train_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
missing_data
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].hist(train_data['Target'], bins=[0.5, 1.5, 2.5, 3.5, 4.5]);
ax[1].hist(train_data['Target'], bins=[0.5, 1.5, 2.5, 3.5, 4.5], normed=True);
ax[0].set_xlabel('Target variable')
ax[0].set_ylabel('number of people')
ax[0].set_xlim([0.5, 4.5])
ax[0].set_xticks(range(1, 5))
ax[0].grid(True)
ax[1].set_xlabel('Target variable')
ax[1].set_ylabel('percentage of people')
ax[1].set_yticks(np.arange(0.0, 0.7, 0.1))
ax[1].set_yticklabels(range(0, 70, 10))
ax[1].set_xlim([0.5, 4.5])
ax[1].set_xticks(range(1, 5))
ax[1].grid(True)
plt.show()
corrmat = train_data.dropna().corr().abs()['Target'].sort_values(ascending=False).drop('Target')
f, ax = plt.subplots(figsize=(20, 6))
plt.xticks(rotation='90')
sns.barplot(x=corrmat.head(50).index, y=corrmat.head(50))
plt.xlabel('Features', fontsize=15)
plt.ylabel('Abs correlation with Target variable', fontsize=15)
plt.show()
lables = ['younger than 12 years of age', '12 years of age and older', 'Total individuals in the household', 'gender']
men_cors = ['r4h1', 'r4h2', 'r4h3', 'male']
women_cors = ['r4m1', 'r4m2', 'r4m3', 'female']

fig, ax = plt.subplots(figsize=(12,6))
ind = np.arange(len(men_cors))
width = 0.35

p1 = ax.bar(ind, corrmat[men_cors].values, width, color='r', bottom=0)
p2 = ax.bar(ind + width, corrmat[women_cors].values, width,
            color='y', bottom=0)

ax.set_title('Correlation of variables by gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(lables)
ax.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.ylabel('correlation with target variable')
ax.autoscale_view()
plt.show()
train_data['v18q1'].fillna(0, inplace=True)
print('number of data rows with (NOT have tablet) & (number of tablets > 0) = %0.f' % len(train_data[(train_data['v18q'] == 0.0) & (train_data['v18q1'] > 0)]))
print('number of data rows with (NOT have mobile phone) & (number of phones > 0) = %0.f' % len(train_data[(train_data['mobilephone'] == 0.0) & (train_data['qmobilephone'] > 0)]))
print('sum(Total females in the household) - sum(Females younger than 12 years of age + Females 12 years of age and older) = {}'.format(train_data['r4m3'].sum() - train_data['r4m2'].sum() - train_data['r4m1'].sum()))
print('sum(Total males in the household) - sum(Males younger than 12 years of age + Males 12 years of age and older) = {}'.format(train_data['r4h3'].sum() - train_data['r4h2'].sum() - train_data['r4h1'].sum()))
print('sum(Total persons in the household) - sum(persons younger than 12 years of age + persons 12 years of age and older) = {}'.format(train_data['r4t3'].sum() - train_data['r4t2'].sum() - train_data['r4t1'].sum()))
print('number of rows for which gender is not specified = {}'.format(train_data['male'].sum() + train_data['female'].sum() - len(train_data)))

wall_material = ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']
print('number of rows for which wall materials are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in wall_material])))
floor_material = ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']
print('number of rows for which floor materials are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in floor_material])))
roof_type = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']
print('number of rows for which roof types are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in roof_type])))
toilet_status = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
print('number of rows for which toilet status are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in toilet_status])))
cooking_energy_source = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']
print('number of rows for which source of cooking energies are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in cooking_energy_source])))
rubbish_disposal_type = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']
print('number of rows for which rubbish disposal types are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in rubbish_disposal_type])))
marital_status = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']
print('number of rows for which marital status are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in marital_status])))
role_in_family = ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
print('number of rows for which roles in family are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in role_in_family])))
level_of_education = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']
print('number of rows for which level of educations are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in level_of_education])))
house_ownership = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
print('number of rows for which house owenerships are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in house_ownership])))
region = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
print('number of rows for which regions are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in region])))
area = ['area1', 'area2']
print('number of rows for which areas are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in area])))

print('number of rows with undefined roof type = {}'.format(len(train_data[(train_data['techozinc'] == 0) & (train_data['techoentrepiso'] == 0) & (train_data['techocane'] == 0) & (train_data['techootro'] == 0)])))
print('number of rows with undefined education level = {}'.format(len(train_data) - sum(train_data[level_of_education].sum(axis=1))))
f, ax = plt.subplots(12, 1, figsize=(14, 168))
types = [wall_material, floor_material, roof_type, toilet_status, cooking_energy_source, rubbish_disposal_type, marital_status, role_in_family, level_of_education, house_ownership, region, area]
types_titles = ['wall materials', 'floor materials', 'roof type', 'toilet status', 'source of cooking energy', 'rubbish disposal type', 'marital status', 'role in family', 'level of education', 'house ownership', 'region', 'area']
for i, t in enumerate(types):
    sns.barplot(x=t, y=train_data[t].sum().values, ax=ax[i])
    ax[i].set_title('distribution of ' + types_titles[i])

plt.show()
sns.kdeplot(train_data.age, legend=False)
plt.title('overall age distribution')
plt.xlabel("Age");
p = sns.FacetGrid(data = train_data, hue = 'Target', size = 5, legend_out=True)
p = p.map(sns.kdeplot, 'age')
plt.legend()
plt.title("Age distribution by household condition")
p;