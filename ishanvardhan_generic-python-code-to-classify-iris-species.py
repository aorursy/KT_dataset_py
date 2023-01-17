#import your libraries

import pandas as pd

import sklearn as sk

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score
#A Function to load a csv from the local computer

def loadfile(file):

    return  pd.read_csv(file)

dataset=loadfile('/kaggle/input/iris/Iris.csv')
class EDA():

    def __init__(self,df):

        """Initialising the constructor to save data in the different attributes"""

        self.data=df

        self.heading(df)

        self.descriptivestats(df)

        self.nulls(df)

        self.visualisations(df) 

        

    def heading(self,df):

        """Using pandas library to get a glimpse of the dataset"""

        self.fewlines=df.head()

        self.matrix=df.shape

        self.cols=df.columns

    

    def descriptivestats(self,df):

        """Displays the pearson correlation coefficient,type of data and all the descriptive stats of the dataset"""

        self.information=df.info()

        self.description=df.describe()

        self.corelation=df.corr()

        

    def nulls(self,df):

        """Inspects the whole dataset and returns a matrix of True and False. True means a null value"""

        self.missing=df.isnull()

        

    def visualisations(self,df):

        """Display the descriptive statistics visually, visualise the histograms to inspect skewness"""

        self.descriptions=df.boxplot()

        self.histogram=df.hist()

        self.sm=scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='hist')

     

    
#Instantiating an object of the EDA Class to store the dataset

eda=EDA(dataset)
#Extracting the relevant data from the attributes of the object

eda.heading(dataset)

eda.fewlines
eda.cols
#Usind drop method of pandas library to remove the ID column

dataset=dataset.drop(['Id'],axis=1)
eda=EDA(dataset)
eda.missing
eda.corelation
#Visualisations using Seaborn

#Categorical Plots

def categoricalplots(df):

    figure2,axes=plt.subplots(2,2,figsize=(12,12))

    SwarmPlot_1=sns.swarmplot(x='Species',y='PetalLengthCm',

                           data=dataset,

                          ax=axes[0,0])

    SwarmPlot_2=sns.swarmplot(x='Species',y='PetalWidthCm',

                           data=dataset,

                          ax=axes[0,1])

    SwarmPlot_3=sns.swarmplot(x='Species',y='SepalWidthCm',

                           data=dataset,

                          ax=axes[1,1])

    SwarmPlot_4=sns.swarmplot(x='Species',y='SepalLengthCm',

                           data=dataset,

                          ax=axes[1,0])

    

    figure3,axes1=plt.subplots(2,2,figsize=(12,12))

    BarPlot_1=sns.barplot(x='Species',y='PetalLengthCm',

                           data=dataset,

                          ax=axes1[0,0])

    BarPlot_2=sns.barplot(x='Species',y='PetalWidthCm',

                           data=dataset,

                          ax=axes1[0,1])

    BarPlot_3=sns.barplot(x='Species',y='SepalWidthCm',

                           data=dataset,

                          ax=axes1[1,1])

    BarPlot_4=sns.barplot(x='Species',y='SepalLengthCm',

                           data=dataset,

                          ax=axes1[1,0])

    

    figure4 = sns.lmplot(x="SepalLengthCm", y="PetalLengthCm", col="Species",

                         data=dataset, aspect=1, x_jitter=.1)

    

    return figure2,figure3,figure4

    

categoricalplots(dataset)
class Engineering():

    def __init__(self,df,var1,var2):

        """Initialising the constructor to save the data"""

        self.data = df

        self.variable1=var1

        self.variable2=var2

        self.featuredivision(var1,var2)

        self.featuremultiplied(var1,var2)

        

    

    def featuredivision(self,var1,var2):

        """calculates ratio of given two variables"""

        self.dividedfeature=var1/var2

        

    

    def featuremultiplied(self,var1,var2):

        """calculates product of given two variables"""

        self.multipliedfeature=var1*var2

       
#Instantiating two objects of the Engineering class

ratio1=Engineering(dataset,dataset['PetalLengthCm'],dataset['PetalWidthCm'])

ratio2=Engineering(dataset,dataset['SepalLengthCm'],dataset['SepalWidthCm'])



#Using division and multiplication methods from the class on the first object

ratio1.featuredivision(dataset['PetalLengthCm'],dataset['PetalWidthCm'])

ratio1.featuremultiplied(dataset['PetalLengthCm'],dataset['PetalWidthCm'])



#Using division and multiplication methods from the class on the second object

ratio2.featuredivision(dataset['SepalLengthCm'],dataset['SepalWidthCm'])

ratio2.featuremultiplied(dataset['SepalLengthCm'],dataset['SepalWidthCm'])



#Saving the data of above two methods into new columns of the original dataset. 

dataset['Petalratio']=ratio1.dividedfeature

dataset['Petalproduct']=ratio1.multipliedfeature

dataset['Sepalratio']=ratio2.dividedfeature

dataset['Sepalproduct']=ratio2.multipliedfeature

#Inspecting the transformed dataset, after feature engineering

ratio1.data
class Preprocess():

    def __init__(self,df):

        """Initialising the constructor method and saving the data in attributes"""

        self.data=df

        self.seperatedata(df)

        

        

    def seperatedata(self,df):

        """Separating the x and y columns using slicing method of pandas library"""

        self.target=df.iloc[:,4]

        self.features=df.iloc[:,lambda df:[0,1,2,3,5,6,7,8]]

           
#Instantiating an object of Preprocess class

pr=Preprocess(dataset)
#Inspecting the data in the attributes of the Preprocess class and saving the data in two different variables

target_variable=pr.target

data_features=pr.features
def labelencoded(df):

    """Using Labelencoding to transform the categorical varibles into numerals before starting data modelling"""

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()

    variable_1 = encoder.fit_transform(df) 

    encoded_dataframe=pd.DataFrame(variable_1)

    return encoded_dataframe

    
#Saving the encoded column in a new variable

final_target_variable=labelencoded(target_variable)
# Define train and test variables by splitting the original X and Y datasets

data_features_train, data_features_test, final_target_variable_train, final_target_variable_test = train_test_split(data_features,final_target_variable,test_size=0.20, random_state=123)

  
class Modelling():

    def __init__(self,var1,var2):

        """Initialising the constructor method to save the data"""

        self.X=var1

        self.Y=var2

        self.baselinemodel(var1,var2)

        self.forest(var1,var2)

        self.boosting(var1,var2)

        self.neighbours(var1,var2)

        self.vectors(var1,var2)

        

    def baselinemodel(self,var1,var2):

        """We start by building a simple logistic regression and then improve upon the results"""

        self.model=LogisticRegression()

        self.model_1=self.model.fit(var1,var2)

        

    def forest(self,var1,var2):

        """Use the ensemble methods to improve on the baseline model"""

        self.model=RandomForestClassifier()

        self.model_2=self.model.fit(var1,var2)

        

    def boosting(self,var1,var2):

        """Implemetation of Boosting algorithm to understand differences from the ensemble ones"""

        self.model=GradientBoostingClassifier()

        self.model_3=self.model.fit(var1,var2)

        

    def neighbours(self,var1,var2):

        """A simple model based on nearest neighbours methodology"""

        self.model=KNeighborsClassifier(n_neighbors = 3)

        self.model_4=self.model.fit(var1,var2)

        

    def vectors(self,var1,var2):

        """implementing support vector machines"""

        self.model=SVC(kernel = 'linear', C = 1)

        self.model_5=self.model.fit(var1,var2)
#Instantiating an object of the Modelling class

mod=Modelling(data_features_train,final_target_variable_train)
#Saving the data of each of the five models trained in new variables

lr=mod.model_1

rf=mod.model_2

gb=mod.model_3

kn=mod.model_4

sv=mod.model_5
#Storing all the model valriables in a new list

list_of_models=[lr,rf,gb,kn,sv]
class Eval():

    def __init__(self,v1,v2,v3):

        """Initialising the constructor and saving all the variable data"""

        self.X_test=v1

        self.Y_test=v2

        self.model=v3

        self.pred(v1,v3)

        self.report(v1,v2,v3)

        

    def pred(self,v1,v3):

        """Predicting the target variable"""

        self.predicted_value=v3.predict(v1)

    

    def report(self,v1,v2,v3):

        """Prints the classification report for the respective model"""

        print(metrics.classification_report(v2,v3.predict(v1)))

        
#Instantiating objects for the five different models

eval1=Eval(data_features_test,final_target_variable_test,lr)

eval2=Eval(data_features_test,final_target_variable_test,rf)

eval3=Eval(data_features_test,final_target_variable_test,gb)

eval4=Eval(data_features_test,final_target_variable_test,kn)

eval5=Eval(data_features_test,final_target_variable_test,sv)