# importanto as bibliotecas

# Manipulação dos dados
import pandas as pd  

# Para uso de matrizes e arrays
import numpy as np  

# Visualização 
import matplotlib.pyplot as plt
import seaborn as sns

# Estatística
import statsmodels as sm

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
# SeniorCitizen

df['SeniorCitizen'] = df.SeniorCitizen.astype('object')
df.info()
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors = 'coerce')

df.TotalCharges.describe()
df.info()
df.TotalCharges.isnull().sum()
df.TotalCharges.fillna(value = df.tenure *  df.MonthlyCharges, inplace = True)
df.TotalCharges.isnull().sum()
df.describe().round()
numerics = df[['tenure','MonthlyCharges', 'TotalCharges', 'Churn']]

plt.figure(figsize = (8,8))

sns.regplot(x = 'tenure', y = 'TotalCharges', data = numerics)

plt.title('Relationship between loyalty months and total revenue')
plt.figure(figsize = (8,8))
plt.title('Relationship between monthly fee and total revenue')
ax = sns.regplot(x = 'MonthlyCharges', y = 'TotalCharges', data = numerics)
plt.figure(figsize = (15,10))
sns.pairplot(numerics)
plt.figure(figsize = (15,10))

sns.boxplot(x = 'tenure', y = 'TotalCharges', data = df)

plt.title('Box Plot of Total Payments X Months of Loyalty')
plt.figure(figsize = (15,10))
sns.countplot(df['tenure'])

df.describe(include = 'object')
pd.crosstab(df.Churn, df.SeniorCitizen,
            margins = True)
# Should make a function for that..
print('The percentage of elderly people who left the company:{}%'.format(476/1142*100))
print('The non-elderly population is:{}%'.format(1393/5901*100)) 
plt.figure(figsize = (8,8))
sns.set(style = 'whitegrid')

sns.countplot(df.SeniorCitizen, hue = df.Churn )
mens_media_idoso = df[df['SeniorCitizen'] == 1]
mens_media_idoso = mens_media_idoso.MonthlyCharges.mean()
mens_media_idoso

n_idoso_media_mes = df[df['SeniorCitizen'] == 0]
n_idoso_media_mes = n_idoso_media_mes.MonthlyCharges.mean()

print('The average monthly expenditure for the elderly is :{}'.format(mens_media_idoso))
print('The average monthly expenditure for non-elderly persons is :{}'.format(n_idoso_media_mes))
# Checking

media_mes_idade = df.groupby('SeniorCitizen').mean() 
media_mes_idade.round()
plt.figure(figsize = (10,8))

sns.set(style = 'whitegrid')
sns.boxplot(x = df.SeniorCitizen, y = df.TotalCharges, hue = df.Churn)

plt.title('Total Revenue by Seniors and Non-Seniors')
df.SeniorCitizen.value_counts(normalize = True)
plt.figure(figsize = (8,8))
sns.set(style = 'whitegrid')
sns.countplot(df.gender, hue = df.Churn)
receita_gender = df.groupby(by = 'gender')['TotalCharges', 'MonthlyCharges'].mean().round()
receita_gender
df.groupby(by = 'gender')['tenure'].mean().round()
plt.figure(figsize = (8,8))
sns.set(style = 'whitegrid')
sns.countplot(df.Partner, hue = df.Churn)
df.groupby('Partner')['TotalCharges', 'MonthlyCharges', 'tenure'].mean().plot(kind = 'bar', stacked = True, 
                                                                             figsize = (8,8))
pd.crosstab(df.Partner, df.Dependents).plot(kind = 'bar', stacked = True, figsize = (8,8))
plt.figure(figsize = (15,10))
sns.countplot(df['tenure'], hue = df.Partner)
df.InternetService.value_counts(normalize = True)
pd.crosstab(df.InternetService, df.PhoneService, margins = True)
plt.figure(figsize = (15,5))
sns.countplot(df.InternetService, hue = df.Churn)
df.groupby('InternetService')['TotalCharges'].mean()
df.head()
df.drop(['customerID', 'gender'], axis = 1, inplace = True)
df_model = df

df_model.head()
df_model.columns
## Here i forked some code from another Kernel

columns_to_convert = ['Partner', 'Dependents','PhoneService','OnlineSecurity' ,
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling',
                      'Churn']
                      
                      
    
    
for item in columns_to_convert:
    df_model[item].replace(to_replace=['Yes', 'No'], value= [1,0], inplace = True)
df_model.head()
# adjusting the column Multiple Lines

df_model.MultipleLines = df_model.MultipleLines.replace(to_replace= 'No phone service', value = 'No')
df_model.MultipleLines = df_model.MultipleLines.replace(to_replace= ['Yes', 'No'], value = [1,0])
df_model.MultipleLines.value_counts()
pd.get_dummies(df_model, columns = ['InternetService', 'Contract', 'PaymentMethod'], drop_first = True)
df_model.OnlineSecurity = df_model.OnlineSecurity.replace(to_replace= 'No internet service', value = 0)
df_model.OnlineBackup = df_model.OnlineBackup.replace(to_replace= 'No internet service', value = 0)
df_model.DeviceProtection = df_model.DeviceProtection.replace(to_replace= 'No internet service', value = 0)
df_model.TechSupport = df_model.TechSupport.replace(to_replace= 'No internet service', value = 0)
df_model.StreamingTV = df_model.StreamingTV.replace(to_replace= 'No internet service', value = 0)
df_model.StreamingMovies = df_model.StreamingMovies.replace(to_replace= 'No internet service', value = 0)

df_model.head(10)
df_model2 = pd.get_dummies(df_model, columns = ['InternetService', 'Contract', 'PaymentMethod'], drop_first = True)
df_model2.head(20)
from sklearn.model_selection import train_test_split

X = df_model2.drop('Churn',axis=1)
y = df_model2['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
# Predictions

predictions = clf.predict(X_test)
predictions
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators= 10)
rfc.fit(X_train,y_train)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test,rfc_predictions))