import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,classification_report, confusion_matrix
import keras 
from keras.layers import Dense
from keras.models import Sequential

%matplotlib inline

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
train.tail()
test = pd.read_csv("../input/test.csv")
test.tail()
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
# We look at the number of missing elements in descending order
train.isnull().sum().sort_values(ascending=False)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='rainbow')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='RdBu_r')
plt.figure(figsize=(12,7))
train['Age'].hist(bins=40,alpha=0.9,color='green')
#As we have the problem with the age we must find a way to answer to this situation
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='seismic')
def About_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(About_age, axis=1)
train['Embarked'].fillna('S')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#We drop the cabin class
train.drop("Cabin",axis=1,inplace=True)
#Let us show the dataframe 
train.head()
#Le us check informations of our data

train.info()
# Coiverting categoritical feature 

sex= pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'], axis=1,inplace=True)
train.head()
train=pd.concat([train,sex,embark], axis=1)
train.head()
train.describe()
#create class
class Titanic_model:
    
    def __init__(self, data=None, cols=None, name='Survived'):
        
        self.name = name # target
        self.data = data # feature
        self.cols = cols # feature columns name
        self.listof_model = {'LinearRegression': LinearRegression(), 
                'KNeighborsRegression':KNeighborsRegressor(),
                'RandomForestRegression': RandomForestRegressor(),
               'GradientBoostingRegression': GradientBoostingRegressor(),
                'XGBoostRegression': XGBRegressor(),
                'adaboost':AdaBoostRegressor()} # list of different learner
    
    #Read csv file
    def read(self, file):
        return pd.read_csv(file)
    
    def multi_categorical_plot(self, data):
    
        """ plot a categorical feature
        
            data: float64 array  n_observationxn_feature
        
        """
        # Find a feature that type is object
        string = []
        for i in data.columns:
            if data[i].dtypes == "object":
                string.append(i)
    
        fig = plt.figure(figsize=(20,5))
        fig.subplots_adjust(wspace=0.2, hspace = 0.3)
        for i in range(1,len(string)+1):
            ax = fig.add_subplot(2,3,i)
            sns.countplot(x=string[i-1], data=data, ax=ax)
            ax.set_title(f" {string[i-1]} countplot")
            
    def distplot_multi(self, data):
        """ plot multi distplot"""
    
        
        from scipy.stats import norm
        cols = []
        
        #Feature that is int64 or float64 type 
        for i in data.columns:
            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':
                cols.append(i)
        
        gp = plt.figure(figsize=(15,10))
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(2,3,i)
            sns.distplot(data[cols[i-1]], fit=norm, kde=False)
            ax.set_title('{} max. likelihood gaussian'.format(cols[i-1]))
            
    def boxplot_multi(self, data):
        
        """ plot multi box plot
            hue for plotting categorical data
        """
    
        cols = []
        for i in data.columns:
            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':
                cols.append(i)
    
        gp = plt.figure(figsize=(15,10))
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(2,3,i)
            sns.boxplot(x = cols[i-1], data=data)
            ax.set_title('Boxplot for {}'.format(cols[i-1]))
            
    def correlation_plot(self, data, vrs= 'Survived'):
    
        """
        This function plot only a variable that are correlated with a target  
        
            data: array m_observation x n_feature
            vrs:  target feature (n_observation, )
            cols: interested features
        """
        
        cols = []
        for i in data.columns:
            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':
                cols.append(i)
                
        feat = list(set(cols) - set([vrs]))
    
        fig = plt.figure(figsize=(15,10))
        fig.subplots_adjust(wspace = 0.3, hspace = 0.25)
        for i in range(1,len(feat)+1):
        
            gp = data.groupby(feat[i-1]).agg('mean').reset_index()
        
            if len(feat) < 3:
                ax = fig.add_subplot(1,3,i)
            else:
                n = len(feat)//2 + 1
                ax = fig.add_subplot(2,n,i)
            
            ax.scatter(data[feat[i-1]], data[vrs], alpha=.25)
            ax.plot(gp[feat[i-1]], gp[vrs], 'r-', label='mean',  linewidth=1.5)
            ax.set_xlabel(feat[i-1])
            ax.set_ylabel(vrs)
            ax.set_title('Plotting data {0} vs {1}'.format(vrs, feat[i-1]))
            ax.legend(loc='best')
            
    # Standardize data
    def standardize(self, data):
        data = (data - data.mean())/data.std()
        return data
            
            
    def VIF(self, data):
        """ 
        This function compute variance inflation factor for data that all feature are multicolinear
        
        if the outcome is 1, it is okay
        if it is between 1 and 5, it shows low to average colinearity, and above 5 generally means highly 
        redundant and variable should be dropped
        """ 
        # Apply the standardize method to each feature and save it to a new data
        std_data = data.apply(self.standardize, axis=0)
    
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    
        vif = pd.DataFrame()
        vif['VIF_FACTOR'] = [variance_inflation_factor(std_data.values, i) for i in range(std_data.shape[1])]
    
        vif['feature'] = std_data.columns
    
        return vif
    
    
    def split_data(self):
        """
        This function splits data to train set and target set
        
        data: matrix feature n_observation x n_feature dimension
        name: target  (n_observation, )
        cols: interested feature
        
        return xtrain, xtest, ytrain, ytest
        """
    
        train = self.data[self.cols]
        target = self.data[self.name]
    
        return train_test_split(train, target, random_state=42, test_size=0.2, shuffle=True)
    
    def spearman_pearson_correlation(self, data):
        
        
        gp = plt.figure(figsize=(15,5))
        cols = ['pearson', 'spearman']
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(1,2,i)
            sns.heatmap(data.corr(method=cols[i-1]), annot=True)
            ax.set_title('{} correlation'.format(cols[i-1]))
        
        
        plt.show()
    
    
    def learner_selection(self):

        """
            This function compute differents score measure like cross validation,
            r2, root mean squared error and mean absolute error.
            listof_model: dictionary type containing different model algorithm.     
        """ 
    
        result = {}
        
        x, _, y, _ = self.split_data() # take only xtrain and ytrain
    
        for cm in list(self.listof_model.items()):
        
            name = cm[0]
            model = cm[1]
        
            cvs = cross_val_score(model, x, y, cv=10).mean()
            ypred = cross_val_predict(model, x, y, cv=10)
            r2 = r2_score(y, ypred)
            mse = mean_squared_error(y, ypred)
            mae = mean_absolute_error(y, ypred)
            rmse = np.sqrt(mse)
        
            result[name] = {'cross_val_score': cvs, 'rmse': rmse, 'mae': mae, 'r2': r2}
        
            print('{} model done !!!'.format(name))
        
        
        return pd.DataFrame(result)
    
    
    def training_evaluate(self, algorithm):
        
        """This function train and evaluate our model to find r2, rmse and mae"""
        
        result = {}
        xtrain, xtest, ytrain, ytest = self.split_data()
        
        learner = self.listof_model[algorithm] # learner selected in model_selection function
        
        model = learner.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        
        r2 = learner.score(xtest, ytest)
        rmse =  np.sqrt(mean_squared_error(ytest, ypred))
        mae = mean_absolute_error(ytest, ypred)
        
        result['car price measure'] = {'r2':round(r2, 3),  'rmse':round(rmse, 3), 'mae':round(mae, 3)}
        
        return  pd.DataFrame(result)

model=Titanic_model()
model.spearman_pearson_correlation(train)
model.correlation_plot(train)
train_cols = ['Parch', 'Pclass', 'SibSp','Fare','Age','PassengerId'] #take columns
model.VIF(train[train_cols])
train_model = Titanic_model(data=train, cols=train_cols) #select best algorithm
train_model.learner_selection()
X_train, X_test, y_train, y_test=train_test_split(train.drop(['Survived'], axis=1), train['Survived'], 
                                                  test_size=0.10,random_state=101)

seq=Sequential()
seq.add(Dense(units=32,init='uniform',activation='relu',input_dim=9))
seq.add(Dense(units=1,init='uniform',activation='sigmoid',input_dim=9))
seq.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
seq.fit(X_train,y_train, batch_size=32,nb_epoch=300,verbose=0)
seq_pred=seq.predict(X_test)
seq_pred=[1 if y>0.5 else 0 for y in seq_pred]
seq_pred
print(confusion_matrix(y_test, seq_pred))
print(classification_report(y_test, seq_pred))