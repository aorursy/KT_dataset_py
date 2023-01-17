import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data_train.head()
data_train.describe(include='all')
full_data = data_train.append(data_test)
full_data.count()
full_data.isnull().sum()
full_data['Embarked'].describe()
full_data_preprocessed = full_data.copy()
full_data_preprocessed['Embarked'] = full_data['Embarked'].fillna('S')
full_data_preprocessed['Embarked'].isnull().sum()
full_data['Fare'].median()
full_data_preprocessed['Fare'] = full_data['Fare'].fillna(full_data['Fare'].median())
full_data_preprocessed['Fare'].isnull().sum()
women_count = data_train.loc[data_train.Sex == 'female']['Survived'].count()
women_survived = data_train.loc[data_train.Sex == 'female']["Survived"].sum()
women_survived_perc = data_train.loc[data_train.Sex == 'female']["Survived"].sum() / data_train.loc[data_train.Sex == 'female']["Survived"].count()
women_count, women_survived, women_survived_perc
man_count = data_train.loc[data_train.Sex == 'male']['Survived'].count()
man_survived = data_train.loc[data_train.Sex == 'male']['Survived'].sum()
man_survived_perc = data_train.loc[data_train.Sex == 'male']['Survived'].sum() / data_train.loc[data_train.Sex == 'male']['Survived'].count()
man_count, man_survived, man_survived_perc
# Creating new DataFrame to store preprocessed values
data_preprocessed = data_train[['Survived']]

#Mapping Sex
data_preprocessed['IsFemale'] = data_train['Sex'].map({'male':0, 'female':1})
data_preprocessed
Pclass_count = data_train['Pclass'].count()
Pclass_1_count = data_train.loc[data_train['Pclass'] == 1]['Pclass'].count()
Pclass_2_count = data_train.loc[data_train['Pclass'] == 2]['Pclass'].count()
Pclass_3_count = data_train.loc[data_train['Pclass'] == 3]['Pclass'].count()
Pclass_1_perc = Pclass_1_count / Pclass_count
Pclass_2_perc = Pclass_2_count / Pclass_count
Pclass_3_perc = Pclass_3_count / Pclass_count

Pclass_1_perc, Pclass_2_perc, Pclass_3_perc
Pclass_1_survived_perc = data_train.loc[data_train['Pclass'] == 1]['Survived'].sum() / Pclass_1_count 
Pclass_2_survived_perc = data_train.loc[data_train['Pclass'] == 2]['Survived'].sum() / Pclass_2_count
Pclass_3_survived_perc = data_train.loc[data_train['Pclass'] == 3]['Survived'].sum() / Pclass_3_count
Pclass_1_survived_perc, Pclass_2_survived_perc, Pclass_3_survived_perc
data_preprocessed['Pclass'] = data_train['Pclass']
data_preprocessed
data_train['Survived'].loc[data_train['Embarked']== 'S'].sum() / data_train['Survived'].loc[data_train['Embarked']== 'S'].count()
data_train['Survived'].loc[data_train['Embarked']== 'C'].sum() / data_train['Survived'].loc[data_train['Embarked']== 'C'].count()
data_train['Survived'].loc[data_train['Embarked']== 'Q'].sum() / data_train['Survived'].loc[data_train['Embarked']== 'Q'].count()
data_preprocessed['Embarked'] = data_train['Embarked']
data_preprocessed
# Group SibSp and Parch
data_train_1 = data_train.copy()
data_train_1['Family_nr'] = data_train_1['SibSp'] + data_train_1['Parch']
data_train_1['IsAlone'] = 0
data_train_1.loc[data_train_1['Family_nr'] == 0, 'IsAlone'] = 1  
data_train_1[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_preprocessed['IsAlone'] = data_train_1['IsAlone']
data_preprocessed
### NOT WORKING BEFORE FIXING NULL VALUES ###

#data_train = data_train.copy()
#data_train_1['AgeGroup'][data_train_1['Age']<=9] = '0-9'
#data_train_1['AgeGroup'][data_train_1['Age']>=10 & data_train_1['Age']<=15] = '10-15'
#data_train_1['AgeGroup'][data_train_1['Age']>=16 & data_train_1['Age']<=54] = '16-54'
#data_train_1['AgeGroup'][data_train_1['Age']>=55] = '55+'
#data_train_1['AgeGroup'][data_train_1['Age']>=14 & data_train_1['Age']<=15] = '0-14'
#data_train_1
sns.distplot(data_train['Fare'])
data_train['TicketFreq'] = data_train.groupby('Ticket')['Ticket'].transform('count')
data_train['PassengerFare'] = data_train['Fare'] / data_train['TicketFreq']

data_train.head()
data_train[data_train['Ticket'] == '373450']['Ticket'].count()
data_train['FareGroup'][data_train['Fare']<=50] = '0-50'
data_train['FareGroup'][data_train['Fare']>50 & data_train['Fare']<=100] = '50-100'
data_train['FareGroup'][data_train['Fare']>100] = '100+'
data_train.loc[data_train['FareGroup'] == '0-50']['Survived'].sum() / data_train.loc[data_train['FareGroup'] == '0-50'].count()
data_preprocessed['Fare'] = data_train.loc[:,'Fare']
# Declaring inputs and target
x = data_preprocessed.iloc[:,1:]
y = data_preprocessed['Survived']
x.shape, y.shape
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = x
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif

### VIF = 1 -> PERFECT
### 1 < VIF < 10 -> OK
### VIF > 10 -> TO REMOVE
# Encoding
data_to_encode = data_preprocessed.iloc[:,1:-1]
dummies = pd.get_dummies(data_to_encode, drop_first=True)
dummies
# Normalizing

# Seleziono colonne
#dati_to_normalize = dati_with_dummies.drop(['Avg_ODP_MIN'], axis = 1)

# Salvo nomi colonne
#cols= dati_to_normalize.columns.values 

# MinMax scaler
# Uso MixMax scaler così è più facile interpretare i risultati dopo
#scaler = MinMaxScaler()
#scaler.fit(dati_to_normalize)
#dati_norm = scaler.transform(dati_to_normalize)
#dati_norm = pd.DataFrame(dati_norm, columns = cols)
#dati_norm
#Feature selection
from sklearn.feature_selection import f_regression
p_values = pd.DataFrame(dummies, columns=['Features'])
p_values['p_value'] = f_regression(x,y)[1]
p_values['p_value'] = p_values['p_value'].round(3)
p_values.sort_values(by=['p_value'])

### If p_value > 0.05 the feature is not statistically relevant and should be removed ###
# Train Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
Logit = LogisticRegression(random_state=0).fit(x_train, y_train)
y_hat = Logit.predict(x_train)
#y_hat
def confusion_matrix(data,actual_values,model):
        
        # Confusion matrix 
        
        # Parameters
        # ----------
        # data: data frame or array
            # data is a data frame formatted in the same way as your input data (without the actual values)
            # e.g. const, var1, var2, etc. Order is very important!
        # actual_values: data frame or array
            # These are the actual values from the test_data
            # In the case of a logistic regression, it should be a single column with 0s and 1s
            
        # model: a LogitResults object
            # this is the variable where you have the fitted model 
            # e.g. results_log in this course
        # ----------
        
        #Predict the values using the Logit model
        pred_values = model.predict(data)
        # Specify the bins 
        bins=np.array([0,0.5,1])
        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
        # if they are between 0.5 and 1, they will be considered 1
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        # Calculate the accuracy
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        # Return the confusion matrix and the accuracy
        return cm, accuracy
# Confusion matrix with train data
cm = confusion_matrix(x_train,y_train,Logit)
cm
# Confusion matrix with test data
cm = confusion_matrix(x_test,y_test,Logit)
cm
