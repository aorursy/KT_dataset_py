train_FileName = "../input/adult-pmr3508/train_data.csv"

test_FileName = "../input/adult-pmr3508/test_data.csv"
import pandas



adult_train = pandas.read_csv(train_FileName,

                                names=[

                                "ID","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                                "Hours per week", "Country", "Target"], # names of columns

                                skiprows=[0], # skip first line (column names in csv), 0-indexed

                                sep=r'\s*,\s*',

                                engine='python',

                                na_values="?") # missing data identified by '?'

adult_train
missing_data_columns = adult_train.columns[adult_train.isnull().any()]

adult_train[adult_train.isnull().any(axis=1)][missing_data_columns]
total_size = adult_train.shape

missing_size = (adult_train[adult_train.isnull().any(axis=1)][missing_data_columns]).shape

print('A conclusao é que existem {} linhas com dados faltantes, presentes em {} colunas.'.format(missing_size[0],missing_size[1]))

print('Isso representa {:.2f}% do total de dados amostrados'.format(100-100*(total_size[0]-missing_size[0])/total_size[0]))
import matplotlib.pyplot as plt



print('Workclass:')

print(adult_train['Workclass'].describe(include='all'))

print('\n')



plt.figure()

adult_train['Workclass'].value_counts().plot(kind="bar")

plt.title('Distribution of Workclass in Train Data')

plt.grid('minor')

print('Elementos Faltantes (Total): {}'.format(adult_train['Workclass'].isnull().sum()))

print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['Workclass'].isnull().sum()/adult_train['Workclass'].size))
print('Occupation:')

print(adult_train['Occupation'].describe(include='all'))

print('\n')



plt.figure()

adult_train['Occupation'].value_counts().plot(kind="bar")

plt.title('Distribution of Occupation in Train Data')

plt.grid('minor')

print('Elementos Faltantes (Total): {}'.format(adult_train['Occupation'].isnull().sum()))

print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['Occupation'].isnull().sum()/adult_train['Occupation'].size))
print('Country:')

print(adult_train['Country'].describe(include='all'))

print('\n')



plt.figure()

adult_train['Country'].value_counts().plot(kind="bar")

plt.title('Distribution of Country in Train Data')

plt.grid('minor')

print('Elementos Faltantes (Total): {}'.format(adult_train['Country'].isnull().sum()))

print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['Country'].isnull().sum()/adult_train['Country'].size))
value = adult_train['Workclass'].describe().top

adult_train['Workclass'] = adult_train['Workclass'].fillna(value)



value = adult_train['Country'].describe().top

adult_train['Country'] = adult_train['Country'].fillna(value)



value = adult_train['Occupation'].describe().top

adult_train['Occupation'] = adult_train['Occupation'].fillna(value)



adult_train
numerical_entries = ["Age","fnlwgt","Education-Num","Capital Gain","Capital Loss","Hours per week"] # only numerical data
# instalando seaborn versao 0.11.0 para ter acesso a função histplot

!pip install --upgrade seaborn==0.11.0



import seaborn

print(seaborn.__version__)

seaborn.set()



seaborn.pairplot(adult_train, vars=numerical_entries, hue='Target')
adult_train[numerical_entries].describe(include='all')
compare_entries = numerical_entries.copy()

compare_entries.append('Target')

comparison = adult_train[compare_entries].copy()



(comparison.loc[comparison['Target']=='<=50K']).describe()
(comparison.loc[comparison['Target']=='>50K']).describe()
capital_analysis_gain = adult_train.loc[(adult_train['Capital Gain']>0)]

capital_analysis_loss = adult_train.loc[(adult_train['Capital Loss']>0)]



plt.figure(figsize=(15, 7))

seaborn.histplot(capital_analysis_gain,x="Capital Gain",hue="Target",multiple="stack",hue_order=["<=50K",">50K"])



plt.figure(figsize=(15, 7))

seaborn.histplot(capital_analysis_loss,x="Capital Loss",hue="Target",multiple="stack",hue_order=["<=50K",">50K"])
capital_analysis_gain[compare_entries].describe(include='all')
capital_analysis_loss[compare_entries].describe(include='all')
capital_analysis_gain = adult_train.loc[(adult_train['Capital Gain']>0) & (adult_train['Capital Gain']<20000)]



plt.figure(figsize=(15, 7))

seaborn.histplot(capital_analysis_gain,x="Capital Gain",hue="Target",multiple="stack",hue_order=["<=50K",">50K"])
categoric_entries = ["Workclass","Education","Martial Status","Occupation","Relationship","Race","Sex","Country"] # only categorical data
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Workclass",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_workclass_total = adult_train["Workclass"].value_counts()

types_workclass_over50K = ((adult_train.loc[adult_train['Target']=='>50K'])['Workclass']).value_counts()



types_entries = list(adult_train["Workclass"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Workclass"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Workclass"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros da Workclass {:16s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Education",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Martial Status",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(adult_train["Martial Status"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Martial Status"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Martial Status"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros da Martial Status {:21s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Occupation",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(adult_train["Occupation"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Occupation"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Occupation"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros da Occupation {:17s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
name_entry = []

number_entry = []



types_entries = list(adult_train["Occupation"].unique())

for i in types_entries:

    name_entry.append(i)

    

    total_entries = adult_train.loc[adult_train["Occupation"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Occupation"]==i) & (adult_train['Target']=='>50K')].shape[0]

    percentage_entries = 100*over50k_entries/total_entries

    

    number_entry.append(percentage_entries)



occupation_percentages_df = pandas.DataFrame({'Occupation':name_entry,'Probability':number_entry})

occupation_percentages_df.sort_values(by='Probability')
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Relationship",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(adult_train["Relationship"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Relationship"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Relationship"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros da Relationship {:14s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Race",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(adult_train["Race"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Race"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Race"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros da Race {:18s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Sex",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(adult_train["Sex"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Sex"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Sex"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros do Sex {:6s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
plt.figure(figsize=(15, 7))

seaborn.histplot(adult_train,x="Country",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(adult_train["Country"].unique())

for i in types_entries:

    total_entries = adult_train.loc[adult_train["Country"]==i].shape[0]

    over50k_entries = adult_train.loc[(adult_train["Country"]==i) & (adult_train['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros do Country {:26s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
country_analysis_df = adult_train.copy()

country_analysis_df.loc[country_analysis_df["Country"]!="United-States", "Country"] = "Not-USA"



plt.figure(figsize=(15, 7))

seaborn.histplot(country_analysis_df,x="Country",hue="Target",multiple="stack",hue_order=["<=50K",">50K"]);
types_entries = list(country_analysis_df["Country"].unique())

for i in types_entries:

    total_entries = country_analysis_df.loc[country_analysis_df["Country"]==i].shape[0]

    over50k_entries = country_analysis_df.loc[(country_analysis_df["Country"]==i) & (country_analysis_df['Target']=='>50K')].shape[0]

    

    print('De {:5d} membros do Country {:13s}, {:4d} tem salário maior do que 50K, representando {:.2f}% do total'.format(total_entries,i,over50k_entries,100*over50k_entries/total_entries))
# Show table while dropping unimportant columns

adult_train.drop(['ID','Education','Relationship','Country','fnlwgt'],axis=1)
adult_train["Workclass"].unique()
workclass_dict = {'Private':1, 'Local-gov':1, 'Self-emp-inc':3, 'State-gov':1, 

                  'Self-emp-not-inc':1, 'Federal-gov':2,'Without-pay':0, 'Never-worked':0}



workclass_newcolumn = pandas.DataFrame({'Workclass':(adult_train['Workclass'].map(workclass_dict))})
adult_train["Martial Status"].unique()
martialstatus_dict = {'Divorced':0, 'Married-civ-spouse':1, 'Never-married':0, 

                      'Widowed':0, 'Married-AF-spouse':1, 'Married-spouse-absent':0,'Separated':0}



martialstatus_newcolumn = pandas.DataFrame({'Martial Status':(adult_train['Martial Status'].map(martialstatus_dict))})
occupation_percentages_df
occupation_dict = occupation_percentages_df.set_index('Occupation')['Probability'].to_dict()



occupation_newcolumn = pandas.DataFrame({'Occupation':(adult_train['Occupation'].map(occupation_dict))})
adult_train["Race"].unique()
race_dict = {'White':1, 'Black':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':0, 'Other':0}



race_newcolumn = pandas.DataFrame({'Race':(adult_train['Race'].map(race_dict))})
adult_train["Sex"].unique()
sex_dict = {'Female':0, 'Male':1}



sex_newcolumn = pandas.DataFrame({'Sex':(adult_train['Sex'].map(sex_dict))})
cluster_capitalgain = pandas.cut(adult_train['Capital Gain'], bins = [-1, 2500, 7000, 100001], labels = [0, 1, 2])



capitalgain_newcolumn = pandas.DataFrame({'Capital Gain':cluster_capitalgain})
cluster_capitalloss = pandas.cut(adult_train['Capital Loss'], bins = [-1, 1800, 5001], labels = [0, 1])



capitalloss_newcolumn = pandas.DataFrame({'Capital Loss':cluster_capitalloss})
age_newcolumn = adult_train['Age']

education_newcolumn = adult_train['Education-Num']

hoursperweek_newcolumn = adult_train['Hours per week']



adult_train_X = pandas.concat([age_newcolumn,workclass_newcolumn,education_newcolumn,martialstatus_newcolumn,

                               occupation_newcolumn,race_newcolumn,sex_newcolumn,capitalgain_newcolumn,

                               capitalloss_newcolumn,hoursperweek_newcolumn],axis=1)

adult_train_X
import numpy



target_dict = {'<=50K':0, '>50K':1}



adult_train_Y = pandas.DataFrame({'Target':(adult_train['Target'].map(target_dict))})



adult_train_XY = pandas.concat([adult_train_X,adult_train_Y],axis=1)

correlation_matrix = (adult_train_XY.astype(numpy.int)).corr()



seaborn.set()

plt.figure(figsize=(15,7))

seaborn.heatmap(correlation_matrix, annot=True)
import sklearn

import sklearn.preprocessing



scaler = sklearn.preprocessing.MinMaxScaler()



adult_train_X_s = pandas.DataFrame(scaler.fit_transform(adult_train_X), columns = adult_train_X.columns)

adult_train_X_s
model_X = adult_train_X_s.values

model_Y = adult_train['Target'].values



import sklearn.neighbors

import sklearn.model_selection



accuracy_mean = []

accuracy_std = []



k_min = 1

k_max = 60



print('Starting analysis of k...')

for k in range(k_min, k_max+1):

    kNN_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)

    accuracy = sklearn.model_selection.cross_val_score(kNN_class, model_X, model_Y, cv=5)

    

    accuracy_mean.append(100*accuracy.mean())

    accuracy_std.append(100*accuracy.std())  

    

    if(k%2==0 and k%4!=0):

        print('k = {:2d} - Accuracy = {:5.2f}% | '.format(k,accuracy_mean[-1]),end='')

    elif(k%2==0):

        print('k = {:2d} - Accuracy = {:5.2f}%'.format(k,accuracy_mean[-1]))
plt.figure(figsize=(15, 7))

plt.errorbar(numpy.arange(k_min,k_max+1), accuracy_mean, accuracy_std, marker='o')

plt.plot([k_min-1,k_max+1],[max(accuracy_mean),max(accuracy_mean)],'--')

plt.xlim([k_min-1,k_max+1])

plt.xlabel('k')

plt.ylabel('Accuracy [%]')

plt.title('Analysis of k (for kNN)')
plt.figure(figsize=(15, 7))

plt.plot(numpy.arange(k_min,k_max+1), numpy.insert(numpy.diff(accuracy_mean),0,0), marker='o')

plt.xlim([k_min-1,k_max+1])

plt.xlabel('k')

plt.ylabel('Accuracy Variation [%]')

plt.title('Analysis of k (for kNN)')
k_opt = 34



kNN_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k_opt)

kNN_class.fit(model_X,model_Y)
adult_test = pandas.read_csv(test_FileName,

                             names=[

                             "ID","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                             "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                             "Hours per week", "Country"], # names of columns

                             skiprows=[0], # skip first line (column names in csv), 0-indexed

                             sep=r'\s*,\s*',

                             engine='python',

                             na_values="?") # missing data identified by '?'

adult_test
missing_data_columns = adult_test.columns[adult_test.isnull().any()]

adult_test[adult_test.isnull().any(axis=1)][missing_data_columns]
total_size = adult_test.shape

missing_size = (adult_test[adult_test.isnull().any(axis=1)][missing_data_columns]).shape

print('A conclusao é que existem {} linhas com dados faltantes, presentes em {} colunas.'.format(missing_size[0],missing_size[1]))

print('Isso representa {:.2f}% do total de dados amostrados'.format(100-100*(total_size[0]-missing_size[0])/total_size[0]))
print('Workclass:')

print(adult_test['Workclass'].describe(include='all'))

print('\n')



plt.figure()

adult_test['Workclass'].value_counts().plot(kind="bar")

plt.title('Distribution of Workclass in Train Data')

plt.grid('minor')

print('Elementos Faltantes (Total): {}'.format(adult_test['Workclass'].isnull().sum()))

print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_test['Workclass'].isnull().sum()/adult_test['Workclass'].size))
print('Occupation:')

print(adult_test['Occupation'].describe(include='all'))

print('\n')



plt.figure()

adult_test['Occupation'].value_counts().plot(kind="bar")

plt.title('Distribution of Occupation in Train Data')

plt.grid('minor')

print('Elementos Faltantes (Total): {}'.format(adult_test['Occupation'].isnull().sum()))

print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_test['Occupation'].isnull().sum()/adult_test['Occupation'].size))
print('Country:')

print(adult_test['Country'].describe(include='all'))

print('\n')



plt.figure()

adult_test['Country'].value_counts().plot(kind="bar")

plt.title('Distribution of Country in Train Data')

plt.grid('minor')

print('Elementos Faltantes (Total): {}'.format(adult_test['Country'].isnull().sum()))

print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_test['Country'].isnull().sum()/adult_test['Country'].size))
# Eliminate nan

value = adult_test['Workclass'].describe().top

adult_test['Workclass'] = adult_test['Workclass'].fillna(value)



value = adult_test['Country'].describe().top

adult_test['Country'] = adult_test['Country'].fillna(value)



value = adult_test['Occupation'].describe().top

adult_test['Occupation'] = adult_test['Occupation'].fillna(value)
workclass_newcolumn = pandas.DataFrame({'Workclass':(adult_test['Workclass'].map(workclass_dict))})

martialstatus_newcolumn = pandas.DataFrame({'Martial Status':(adult_test['Martial Status'].map(martialstatus_dict))})

occupation_newcolumn = pandas.DataFrame({'Occupation':(adult_test['Occupation'].map(occupation_dict))})

race_newcolumn = pandas.DataFrame({'Race':(adult_test['Race'].map(race_dict))})

sex_newcolumn = pandas.DataFrame({'Sex':(adult_test['Sex'].map(sex_dict))})



cluster_capitalgain = pandas.cut(adult_test['Capital Gain'], bins = [-1, 2500, 7000, 100000], labels = [0, 1, 2])

capitalgain_newcolumn = pandas.DataFrame({'Capital Gain':cluster_capitalgain})



cluster_capitalloss = pandas.cut(adult_test['Capital Loss'], bins = [-1, 1800, 5001], labels = [0, 1])

capitalloss_newcolumn = pandas.DataFrame({'Capital Loss':cluster_capitalloss})



age_newcolumn = adult_test['Age']

education_newcolumn = adult_test['Education-Num']

hoursperweek_newcolumn = adult_test['Hours per week']



adult_test_X = pandas.concat([age_newcolumn,workclass_newcolumn,education_newcolumn,martialstatus_newcolumn,

                               occupation_newcolumn,race_newcolumn,sex_newcolumn,capitalgain_newcolumn,

                               capitalloss_newcolumn,hoursperweek_newcolumn],axis=1)



adult_test_X_s = pandas.DataFrame(scaler.fit_transform(adult_test_X), columns=adult_test_X.columns)

adult_test_X_s
model_X = adult_test_X_s.values

predictions = kNN_class.predict(model_X)
submission = pandas.DataFrame()

submission[0] = adult_test.index

submission[1] = predictions

submission.columns = ['Id','income']

submission.to_csv('submission.csv',index=False)