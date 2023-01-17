import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For Plotting and Visualization

import seaborn as sns # For Plotting and Visualization

from sklearn.model_selection import train_test_split # For Splitting the Dataset into Training and Test Dataset

import statsmodels.api as sm # For Building Linear Model

from statsmodels.stats.outliers_influence import variance_inflation_factor # To Calculate Variance Inflation Factor Between Indepenent Variables

import os

print(os.listdir("../input"))

    
class PredictGreScore():

    

    def __init__(self,x,y,model):

        self.predictordata=x

        self.actualresponsevalue=y

        self.model=model

        # Calculate Degree of Freedom of Predictor Variable Variance

        self.df_pred=x.shape[0]-1

        # Calculate Degree of Freedom of Population Error Variance

        self.df_error=x.shape[0]-x.shape[1]-1

    

    # Calculate Total Sum of Squares

    

    def TSS(self):

        avg_y=np.mean(self.actualresponsevalue)

        squared_errors=((self.actualresponsevalue)-(avg_y))**2

        return np.sum(squared_errors)

    

    # Calculate Residual Sum of Squares

    

    def RSS(self):

        ActualValue=self.actualresponsevalue

        PredictedValue=self.model.predict(self.predictordata)

        ResidualError = (ActualValue-PredictedValue)**2

        return np.sum(ResidualError)

    

    # Calculate R-Squared Value

    

    def r_squared(self):

        return 1 - self.RSS()/self.TSS()

    

    # Calculate Adjusted R-Squared Value

    

    def adj_rsquared(self):

        return 1-(self.RSS()/self.df_error)/(self.TSS()/self.df_pred)

    

    # Plot Residual Analysis of Error Data

    

    def plot_residualanalysis(self):

        fig=plt.figure()

        sns.distplot(self.actualresponsevalue-self.model.predict(self.predictordata))

        plt.xlabel('Errors',fontsize=18)

        

    
def print_statsresults(stats_obj):

    items=(('Residual Sum of Squares:',stats_obj.RSS()),('Total Sum of Squares:',stats_obj.TSS()),('R-Squared:',stats_obj.r_squared()),('Adjusted R-Squared:',stats_obj.adj_rsquared()))

    for item in items:

        print('{0:8}{1:.4f}'.format(item[0],item[1]))
# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Read the input dataset

input=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

input.head()
# Rows and Columns in the dataset

input.shape

# Datatypes of each Column in the Dataset

input.info()
# Understand the Correlation between each Column in the dataset



sns.set(style='ticks',color_codes=True)

sns.pairplot(input)

plt.show()

# check for missing values in the dataset



input.isnull().sum()
# Dataset is divided into Training and Testing Data using train_test_split imported from sklearn.model_selection



np.random.seed(0)



# split the dataframe



input_train,input_test=train_test_split(input,train_size=0.6,test_size=0.4,random_state=None)
# Perform Correlation on the Training Data to identify the Predictor Variables Highly Correlated with the Response Variable



plt.figure(figsize = (25, 10))

sns.heatmap(input_train.corr(),annot=True,cmap='YlGnBu')

plt.show()

y_train=input_train.pop('GRE Score')

x_train=input_train
# Adding the Constant



x_train_lm=sm.add_constant(x_train[['TOEFL Score','CGPA','Chance of Admit ']])

# Create Linear Regression Model



lr=sm.OLS(y_train,x_train_lm).fit()
x_train_lm.head()
print(lr.summary())
# Creating a dataframe which contains the list of Predictor Variables and their VIF's



vif=pd.DataFrame()

vif['Features']=x_train_lm.columns

vif['vif']=[variance_inflation_factor(x_train_lm.values,i) for i in range(x_train_lm.shape[1])]

vif['vif']=round(vif['vif'],2)

vif=vif.sort_values(by="vif",ascending=False)

vif
ResidualAnalysis=PredictGreScore(x_train_lm,y_train,lr)

ResidualAnalysis.plot_residualanalysis()
stats=PredictGreScore(x_train_lm,y_train,lr)

print_statsresults(stats)
# Building a new model by Considering TOEFL Score and Chance of Admit as Predictor Variables and removed CGPA which has high VIF 

x_train_model2=sm.add_constant(x_train[['TOEFL Score','Chance of Admit ']])
# Run the Model

lr_model2=sm.OLS(y_train,x_train_model2).fit()
print(lr_model2.summary())
# Plotting Residual Analysis to see if error terms are normally dsitributed

ResidualAnalysis=PredictGreScore(x_train_model2,y_train,lr_model2)

ResidualAnalysis.plot_residualanalysis()
stats=PredictGreScore(x_train_model2,y_train,lr_model2)

print_statsresults(stats)
x_train_model3=sm.add_constant(x_train[['TOEFL Score','CGPA','University Rating','Chance of Admit ']])
lr_model3=sm.OLS(y_train,x_train_model3).fit()
print(lr_model3.summary())
# Plotting Residual Analysis to see if error terms are normally dsitributed

ResidualAnalysis=PredictGreScore(x_train_model3,y_train,lr_model3)

ResidualAnalysis.plot_residualanalysis()
stats=PredictGreScore(x_train_model3,y_train,lr_model3)

print_statsresults(stats)
# Let us split the test data into x_test and y_test

y_test=input_test.pop('GRE Score')

x_test=input_test
# Now lets use Model 1 to make Predictions and select the Predictor Variables used in Model 1



# Drop the Constant Variable Column from the train columns used in the Model 1

x_train_new=x_train_lm.drop(['const'],axis=1)



x_test_new=x_test[x_train_new.columns]



# Adding a Constant Variable

x_test_new=sm.add_constant(x_test_new)



#Making the Predictions



y_pred=lr.predict(x_test_new)
stats=PredictGreScore(x_test_new,y_test,lr)

print_statsresults(stats)

# Plotting the y_test and y_pred to Understand the Spread



fig=plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred',fontsize=20)

plt.xlabel('y_test',fontsize=18)

plt.ylabel('y_pred',fontsize=16)