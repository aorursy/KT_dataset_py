import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
dfs = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
dfs.shape
pd.set_option('display.max_rows', 122)
pd.set_option('display.max_columns', 111)
colunas = dfs.columns
for item in colunas:
    print(item)
df_param = dfs[['Patient age quantile','SARS-Cov-2 exam result','Platelets','Leukocytes','Eosinophils','Monocytes','Hematocrit','Hemoglobin','Red blood Cells','Mean corpuscular volume (MCV)','Red blood cell distribution width (RDW)','Total Bilirubin','Indirect Bilirubin','Direct Bilirubin']]
df_param.isnull().sum()

df_pnull = df_param.dropna()
df_pnull['SARS-Cov-2 exam result']=df_pnull['SARS-Cov-2 exam result'].str.replace('negative', '0')
df_pnull['SARS-Cov-2 exam result']=df_pnull['SARS-Cov-2 exam result'].str.replace('positive', '1')
df_pnull['SARS-Cov-2 exam result'] = df_pnull['SARS-Cov-2 exam result'].values.astype(np.float64)
df_pnull['SARS-Cov-2 exam result'].describe()
df_pnull.isnull().sum()
#Check cols with null values
dfs.isnull().sum()

#Sub dataset with age, COVID test and hemograma
df_hemograma = dfs[colunas[1:20]]
df_hemograma.head()
#Drop rows with incomplete exams for hemograma
df_hemo_filter = df_hemograma.dropna()
# Replace string in COVID test to int64
df_hemo_filter['SARS-Cov-2 exam result'].replace('negative', '0', inplace = True)
df_hemo_filter['SARS-Cov-2 exam result'].replace('positive', '1', inplace = True)
df_hemo_filter['SARS-Cov-2 exam result'] = df_hemo_filter['SARS-Cov-2 exam result'].values.astype(np.int64)
#Subset with Age,COVID test and Income status
df_check = df_hemo_filter[colunas[1:6]]
df_check.head()
#Check number of rows group by COVID status
positivo=df_hemo_filter[(df_hemo_filter["SARS-Cov-2 exam result"] ==1)]
negativo=df_hemo_filter[(df_hemo_filter["SARS-Cov-2 exam result"] ==0)]
print(positivo.shape,negativo.shape)
df_hemo_filter.head()
# Exploratory Analysis of check Dataset
plt.rc("font", size = 10)

count = 0

for i in list(df_check.columns):
    print('>----------<')
    print()
    print(count)
    print()
    print(df_check.groupby(i).size()) # Resumindo o conjunto de dados pela função 'groupby()'
    print()
    f, ax = plt.subplots(figsize=(20, 4))
    sns.countplot(x = i, data = df_check, palette = 'hls')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
    count = count + 1
df_pnull.describe()
# Exploratory Analysis of df_pnull
plt.rc("font", size = 10)

count = 0

for i in list(df_pnull.columns):
    print('>----------<')
    print()
    print(count)
    print()
    print(df_pnull.groupby(i).size()) # Resumindo o conjunto de dados pela função 'groupby()'
    print()
    f, ax = plt.subplots(figsize=(20, 4))
    sns.countplot(x = i, data = df_pnull, palette = 'hls')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
    count = count + 1
plt.rc("font", size = 10)
plt.rcParams["figure.figsize"] = (20,4)

for i in list(df_check.columns):
    table = pd.crosstab(df_check[i], dfs["SARS-Cov-2 exam result"])
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title(i + ' X COVID-19_pos (True)')
    plt.xlabel(i + ' Status')
    plt.ylabel('Proportion')
plt.rc("font", size = 10)
plt.rcParams["figure.figsize"] = (20,4)

for i in list(df_pnull.columns):
    table = pd.crosstab(df_pnull[i], df_pnull["SARS-Cov-2 exam result"])
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title(i + ' X COVID-19_pos (True)')
    plt.xlabel(i + ' Status')
    plt.ylabel('Proportion')
plt.rc("font", size = 10)
plt.rcParams["figure.figsize"] = (20,4)

for i in list(df_check.columns):
    table = pd.crosstab(df_check[i], dfs["Patient age quantile"])
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title(i + ' X Idade (True)')
    plt.xlabel(i + ' Status')
    plt.ylabel('Proportion')
plt.rc("font", size = 10)
plt.rcParams["figure.figsize"] = (20,4)

for i in list(df_hemo_filter.columns):
    table = pd.crosstab(df_hemo_filter[i], df_hemo_filter["SARS-Cov-2 exam result"])
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title(i + ' X COVID-19_pos (True)')
    plt.xlabel(i + ' Status')
    plt.ylabel('Proportion')
df_pnull.columns
#Rename hemograma Dataset
df_pnull.columns=['Idade','COVID','Plaquetas','Leucocitos','Eosinofilos','Monocitos','Ht','Hemoglobina','Hemacias','MCV','RDW','Bili_total','Bili_ind','Bili_dir']
df_hemo_filter.columns = ['Idade','COVID','Interna','Semi','UTI', 'Hematrocito','Hemoglobina','Plaquetas','Vol_Plaquetas','Hemacias','Linfocitos','MCHC','Leucocitos','Basofilos','MCH','Eosinofilos','MCV','Monocitos','RDW']
df_new = df_hemo_filter.copy()
df_pnull.head()
#Scaling col['IDADE']
colunas_1=df_new.columns
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_new), columns=colunas_1)
scaler2 = MinMaxScaler()
colunas_2 = df_pnull.columns
df_scaled2 = pd.DataFrame(scaler2.fit_transform(df_pnull), columns=colunas_2)
#Split Features and Label
Xs = df_scaled.drop(["COVID"],axis=1)
Ys = df_scaled["COVID"]
X2=df_scaled2.drop(["COVID"],axis=1)
Y2=df_scaled2['COVID']
df_scaled2.describe()
#Test Feature importance: unbalanced Dataset using correction
brf = BalancedRandomForestClassifier(n_estimators=1000, random_state=0)
brf.fit(Xs, Ys)

  
for num,item in enumerate(brf.feature_importances_):
    print(Xs.columns[num],item)
    

brf2 = BalancedRandomForestClassifier(n_estimators=1000, random_state=0)
brf2.fit(X2, Y2) 
for num,item in enumerate(brf2.feature_importances_):
    print(X2.columns[num],item)
#Testing RandomForestClassifier at unbalanced dataset
X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, train_size=0.8,test_size=0.2, random_state=101)
brf = BalancedRandomForestClassifier(n_estimators=1000, random_state=0)
brf.fit(X_train, y_train) 
y_pred = brf.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
X2.head()


#Testing RandomForestClassifier at unbalanced dataset Bilirubinas
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, train_size=0.7,test_size=0.3, random_state=101)
brf2 = BalancedRandomForestClassifier(n_estimators=1000)
brf2.fit(X_train2, y_train2) 
y_pred2 = brf2.predict(X_test2)
result_1 = confusion_matrix(y_test2, y_pred2)
print("Confusion Matrix:")
print(result_1)
result1_1 = classification_report(y_test2, y_pred2)
print("Classification Report:",)
print (result1_1)
result2_1 = accuracy_score(y_test2,y_pred2)
print("Accuracy:",result2_1)
#Oversampling the minority class

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(Xs, Ys)
from collections import Counter
print(sorted(Counter(y_resampled).items()))
ros2 = RandomOverSampler(random_state=0)
X_resampled2, y_resampled2 = ros2.fit_resample(X2, Y2)
from collections import Counter
print(sorted(Counter(y_resampled2).items()))
#Reconstructing hemograma dataset with resampled data
X_revamp = X_resampled.copy()
X_revamp.insert(1, "COVID", y_resampled, True) 
df_resample = X_revamp.copy()
df_sample = df_new.copy()
#comparing correlation before and after oversampling
corr = df_sample.corr()
corr2 = df_resample.corr()
#Reconstructing bilirubina dataset with resampled data
X_revamp2 = X_resampled2.copy()
X_revamp2.insert(1, "COVID", y_resampled2, True) 
df_resample2 = X_revamp2.copy()
df_sample2 = df_pnull.copy()
#comparing correlation before and after oversampling
corr_2 = df_sample2.corr()
corr2_2 = df_resample2.corr()
#Show correlation matrix heatmap BEFORE oversampling
sns.heatmap(corr, annot=True)
plt.show()
#Show correlation matrix heatmap AFTER oversampling
sns.heatmap(corr2, annot=True)
plt.show()
sns.heatmap(corr_2, annot=True)
plt.show()
sns.heatmap(corr2_2, annot=True)
plt.show()
#Split train and test datasets. 

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.8,test_size=0.2, random_state=101)
#Exploratory Feature Analysis 1


model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)
importance = model.feature_importances_
for i,v in enumerate(importance):
    
    print('Feature: %0d, Score: %.5f,Coluna: ' % (i,v))
    print(X_resampled.columns[i])
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#Exploratory Feature Analysis 2

model = XGBClassifier()

model.fit(X_train, y_train)

importance = model.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    print(X_resampled.columns[i])
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#Exploratory Feature Analysis 3
model = DecisionTreeClassifier()
model.fit(X_resampled, y_resampled)
importance = model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    print(X_resampled.columns[i])
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#evaluate metrics Hemogram subset

model1 = KNeighborsClassifier()

model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
#evaluate metrics Hemogram Dataset
model1 = XGBClassifier()

model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
#evaluate metrics Hemogram Dataset
model1 = DecisionTreeClassifier()

model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
#Evaluate metrics Hemogram Dataset

model1 = RandomForestClassifier()
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
#Create a new subset with Age,Platelets,Leucocytes,Eosinophyles Monocytes
Novo_Xs = X_resampled[['Idade','Plaquetas','Leucocitos','Eosinofilos','Monocitos']]
Novo_X_train, Novo_X_test, Novo_y_train, Novo_y_test = train_test_split(Novo_Xs, y_resampled, train_size=0.8,test_size=0.2, random_state=101)
#evaluate metrics with the latest subset 3

model2 = KNeighborsClassifier()

model2.fit(Novo_X_train, Novo_y_train)

y_pred = model2.predict(Novo_X_test)
result = confusion_matrix(Novo_y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Novo_y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Novo_y_test,y_pred)
print("Accuracy:",result2)
#evaluate metrics with the latest subset 4
model2 = DecisionTreeClassifier()

model2.fit(Novo_X_train, Novo_y_train)

y_pred = model2.predict(Novo_X_test)
result = confusion_matrix(Novo_y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Novo_y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Novo_y_test,y_pred)
print("Accuracy:",result2)
#evaluate metrics with the simplified subset 2




model2 = RandomForestClassifier()
model2.fit(Novo_X_train, Novo_y_train)

y_pred = model2.predict(Novo_X_test)
result = confusion_matrix(Novo_y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Novo_y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Novo_y_test,y_pred)
print("Accuracy:",result2)
#evaluate metrics with the latest subset 2
model2 = XGBClassifier()

model2.fit(Novo_X_train, Novo_y_train)

y_pred = model2.predict(Novo_X_test)
result = confusion_matrix(Novo_y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Novo_y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Novo_y_test,y_pred)
print("Accuracy:",result2)
#evaluate metrics with bilirubina dataset
Novo_X_train3, Novo_X_test3, Novo_y_train3, Novo_y_test3 = train_test_split(X_resampled2, y_resampled2, train_size=0.7,test_size=0.3, random_state=101)

model3 = RandomForestClassifier(n_estimators=1000,random_state=101)
model3.fit(Novo_X_train3, Novo_y_train3)

y_pred3 = model3.predict(Novo_X_test3)
result_3 = confusion_matrix(Novo_y_test3, y_pred3)
print("Confusion Matrix:")
print(result_3)
result1_3 = classification_report(Novo_y_test3, y_pred3)
print("Classification Report:",)
print (result1_3)
result2_3 = accuracy_score(Novo_y_test3,y_pred3)
print("Accuracy:",result2_3)

#evaluate metrics with bilirubina simplified without idade dataset
#X_resampled3 = X_resampled2[['Ht','Hemoglobina','Hemacias','MCV','RDW','Bili_ind']]
X_resampled3 = X_resampled2[['Hemacias','Ht','Hemoglobina','Leucocitos','Bili_ind']]
Novo_X_train3, Novo_X_test3, Novo_y_train3, Novo_y_test3 = train_test_split(X_resampled3, y_resampled2, train_size=0.7,test_size=0.3, random_state=101)

model3 = RandomForestClassifier(n_estimators=1000,random_state=101)
model3.fit(Novo_X_train3, Novo_y_train3)

y_pred3 = model3.predict(Novo_X_test3)
result_3 = confusion_matrix(Novo_y_test3, y_pred3)
print("Confusion Matrix:")
print(result_3)
result1_3 = classification_report(Novo_y_test3, y_pred3)
print("Classification Report:",)
print (result1_3)
result2_3 = accuracy_score(Novo_y_test3,y_pred3)
print("Accuracy:",result2_3)
colunas = dfs.columns
#Initiate EDA by Age x Other respiratory diseases
teste_novo =  dfs[colunas[21:39]]
idade = dfs["Patient age quantile"]
COVID = dfs["SARS-Cov-2 exam result"]
teste_novo["Idade"]=idade 
teste_novo["COVID"]=COVID
teste_novo = teste_novo.drop(["Mycoplasma pneumoniae"],axis=1)
teste_novo =teste_novo.dropna()
teste_novo.head()
teste_filtrado = teste_novo.copy()
colunas = teste_filtrado.columns
print (colunas)
#EDA -transforming all features in binary or numeric
colunas = teste_filtrado.columns
for coluna in colunas[0:18]:
    teste_filtrado[coluna].replace('not_detected', '0', inplace = True)
    teste_filtrado[coluna].replace('detected', '1', inplace = True)
    teste_filtrado[coluna] = teste_filtrado[coluna].values.astype(np.int64)
teste_filtrado["COVID"].replace('negative', '0', inplace = True)
teste_filtrado["COVID"].replace('positive', '1', inplace = True)
teste_filtrado["COVID"] = teste_filtrado["COVID"].values.astype(np.int64)
teste_filtrado=teste_filtrado.drop(['Parainfluenza 2'],axis=1)

scaler1 = MinMaxScaler()
teste_filtrado[['Idade']] = scaler1.fit_transform(teste_filtrado[['Idade']])
teste_filtrado.tail(10)
# Exploratory Analysis of Other Respiratory Diseases Dataset
plt.rc("font", size = 10)

count = 0

for i in list(teste_filtrado.columns):
    print('>----------<')
    print()
    print(count)
    print()
    print(teste_filtrado.groupby(i).size()) # Resumindo o conjunto de dados pela função 'groupby()'
    print()
    f, ax = plt.subplots(figsize=(20, 4))
    sns.countplot(x = i, data = teste_filtrado, palette = 'hls')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
    count = count + 1
plt.rc("font", size = 10)
plt.rcParams["figure.figsize"] = (20,4)

for i in list(teste_filtrado.columns):
    table = pd.crosstab(teste_filtrado[i], teste_filtrado['COVID'])
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title(i + ' X COVID-19_pos (True)')
    plt.xlabel(i + ' Status')
    plt.ylabel('Proportion')
# Oversampling of Other Respiratory Disease Dataset
ros = RandomOverSampler(random_state=0)
Xs_Unb = teste_filtrado.drop(["COVID"],axis=1)
Ys_Unb = teste_filtrado["COVID"]
X_filter, y_filter = ros.fit_resample(Xs_Unb, Ys_Unb)
X_refilter = X_filter.copy()
X_refilter.insert(0, "COVID", y_filter, True) 
df_resample = X_refilter.copy()

corr3=df_resample.corr()
sns.heatmap(corr3, annot=True)
plt.show()

#Testing RandomForestClassifier at unbalanced dataset
X_train, X_test, y_train, y_test = train_test_split(Xs_Unb, Ys_Unb, train_size=0.8,test_size=0.2, random_state=101)
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
brf.fit(X_train, y_train) 
y_pred = brf.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
#Evaluate metrics Other Respiratory Diseases Oversampling Dataset

X_train, X_test, y_train, y_test = train_test_split(X_filter, y_filter, train_size=0.8,test_size=0.2, random_state=101)
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)






