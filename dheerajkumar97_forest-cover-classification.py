import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path ="/kaggle/input/kagg-train/kaggle_train.csv"
path1 ="/kaggle/input/kagg-test/kaggle_test.csv"
class DataFrame_Loader():

    
    def __init__(self):
        
        print("Loadind DataFrame")
        
    def read_csv(self,data):
        self.df = pd.read_csv(data)
        
    def load_csv(self):
        return self.df
load= DataFrame_Loader()
load.read_csv(path)
dftrain = load.load_csv()
dftrain.head()
dftest = load.load_csv()
dftest.head()
class DataFrame_Information():
    

    def __init__(self):
        
        print("Attribute Information object created")
        
        
        
    def Attribute_information(self,df):
        
        """
        This method will give us a basic
        information of the dataframe like
        Count of Attributes,Count of rows,
        Numerical Attributes, Categorical 
        Attributes, Factor Attributes etc..
        """
    
        data_info = pd.DataFrame(
                                columns=['No of observation',
                                        'No of Variables',
                                        'No of Numerical Variables',
                                        'No of Factor Variables',
                                        'No of Categorical Variables',
                                        'No of Logical Variables',
                                        'No of Date Variables',
                                        'No of zero variance variables'])


        data_info.loc[0,'No of observation'] = df.shape[0]
        data_info.loc[0,'No of Variables'] = df.shape[1]
        data_info.loc[0,'No of Numerical Variables'] = df._get_numeric_data().shape[1]
        data_info.loc[0,'No of Factor Variables'] = df.select_dtypes(include='category').shape[1]
        data_info.loc[0,'No of Logical Variables'] = df.select_dtypes(include='bool').shape[1]
        data_info.loc[0,'No of Categorical Variables'] = df.select_dtypes(include='object').shape[1]
        data_info.loc[0,'No of Date Variables'] = df.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0,'No of zero variance variables'] = df.loc[:,df.apply(pd.Series.nunique)==1].shape[1]

        data_info =data_info.transpose()
        data_info.columns=['value']
        data_info['value'] = data_info['value'].astype(int)


        return data_info

    def __get_missing_values(self,data):
        
        """
        It is a Private method, so it cannot 
        be accessed by object outside the 
        class. This function will give us 
        a basic information like count 
        of missing values
        """
        
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values
        
    def Agg_Tabulation(self,data):
        
        
        """
        This method is a extension of 
        schema will gives the aditional 
        information about the data
        like Entropy value, Missing 
        Value Percentage and some observations
        """
        
        print("=" * 110)
        print("Aggregation of Table")
        print("=" * 110)
        table = pd.DataFrame(data.dtypes,columns=['dtypes'])
        table1 =pd.DataFrame(data.columns,columns=['Names'])
        table = table.reset_index()
        table= table.rename(columns={'index':'Name'})
        table['No of Missing'] = data.isnull().sum().values    
        table['No of Uniques'] = data.nunique().values
        table['Percent of Missing'] = ((data.isnull().sum().values)/ (data.shape[0])) *100
        table['First Observation'] = data.loc[0].values
        table['Second Observation'] = data.loc[1].values
        table['Third Observation'] = data.loc[2].values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(data[name].value_counts(normalize=True), base=2),2)
        return table
    
        print("=" * 110)
        
    def __iqr(self,x):
        
        
        """
        It is a private method which 
        returns you interquartile Range
        """
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self,x):
        
        
        """
        It is a private method which 
        returns you outlier present
        in the interquartile Range
        """
        upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self,df):
        
        
        """
        This method will returns 
        you the information about
        numerical attributes like
        Positive values,Negative Values
        Unique count, Zero count 
        positive and negative inf-
        nity count and count of outliers
        etc 
        
        """
        
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i=0
        for c in  df_num.columns:
            data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
            data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
            data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
            data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
            data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
            data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
            data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
            data_info_num.loc[c,'Count of outliers']= self.__outlier_count(df_num[c])
            i = i+1
        return data_info_num
    
    def statistical_summary(self,df):
        
        
        """
        This method will returns 
        you the varoius percentile
        of the data including count 
        and mean
        """
    
        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
            data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num
info = DataFrame_Information()
info.Attribute_information(dftrain)
info.Agg_Tabulation(dftrain)
info.num_count_summary(dftrain)
info.statistical_summary(dftrain)
class DataFrame_Preprocessor():
    

    def __init__(self):
        print("Preprocessor object created")
        
        
    def __split_numbers_chars(self,row):
        head = row.rstrip('0123456789')
        tail = row[len(head):]
        return head, tail
    
    def reverse_one_hot_encode(self,dataframe, start_loc, end_loc, numeric_column_name):
        dataframe['String_Column'] = (dataframe.iloc[:, start_loc:end_loc] == 1).idxmax(1)
        dataframe['Tuple_Column'] = dataframe['String_Column'].apply(self.__split_numbers_chars)
        dataframe[numeric_column_name] = pd.to_numeric(dataframe['Tuple_Column'].apply(lambda x: x[1]),errors='coerce')
        dataframe.drop(columns=['String_Column','Tuple_Column'], inplace=True)

Preprocessor = DataFrame_Preprocessor()
Preprocessor.reverse_one_hot_encode(dftrain,14,54,'soil_type')
Preprocessor.reverse_one_hot_encode(dftrain,10,14,'wilderness')
dftrain.head()
Preprocessor.reverse_one_hot_encode(dftest,14,54,'soil_type')
Preprocessor.reverse_one_hot_encode(dftest,10,14,'wilderness')
dftest.head()
col_list = ['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10',
  'Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19',
  'Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28',
  'Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37',
  'Soil_Type38','Soil_Type39','Soil_Type40']

col_list1 = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']

class Column_Dopper():

    
    def __init__(self):
        print("Column Dopper object created")
    
    
    def dropper(self,x):
        
        """
        This method helps
        to drop the columns
        in our original 
        dataframe which is 
        available in the 
        col_list and return 
        us final dataset
        """
        
        data=[]
        for i in x.columns:
            if i not in col_list:
                data.append(i)
        return x[data]

col_drop = Column_Dopper()
dftrain = col_drop.dropper(dftrain)
dftrain.head()
dftest = col_drop.dropper(dftest)
dftest.head()
class DataFrame_Feature_Engineering():

    def __init__(self):
        print("Feature Engineering object created")
        
    def Make_Features_for_Train(self,dftrain):
        
        dftrain['Hydro_fire'] = dftrain['Horizontal_Distance_To_Fire_Points'] +  dftrain['Horizontal_Distance_To_Hydrology']
        dftrain['Hydro_Road'] = dftrain['Horizontal_Distance_To_Roadways'] +  dftrain['Horizontal_Distance_To_Hydrology']
        dftrain['Road_fire'] = dftrain['Horizontal_Distance_To_Fire_Points'] +  dftrain['Horizontal_Distance_To_Roadways']
        dftrain['Hydro_Road_sub'] = np.abs(dftrain['Horizontal_Distance_To_Roadways'] -  dftrain['Horizontal_Distance_To_Hydrology'])
        dftrain['Hydro_fire_sub'] = np.abs(dftrain['Horizontal_Distance_To_Fire_Points'] -  dftrain['Horizontal_Distance_To_Hydrology'])
        dftrain['Road_fire_sub'] = np.abs(dftrain['Horizontal_Distance_To_Fire_Points'] -  dftrain['Horizontal_Distance_To_Roadways'])
        dftrain['RAD_SLOPE'] = dftrain['Slope'].apply(lambda x: x*(np.pi/180))
        dftrain['RAD_Aspect'] = dftrain['Aspect'].apply(lambda x: x*(np.pi/180))
        dftrain['EL_DIS'] = dftrain['Elevation'] - dftrain['Horizontal_Distance_To_Hydrology']*0.2
        dftrain['EL_Fire'] = dftrain['Elevation'] - dftrain['Horizontal_Distance_To_Fire_Points']*0.2
        dftrain['EL_Road'] = dftrain['Elevation'] - dftrain['Horizontal_Distance_To_Roadways']*0.2
        
        
    def Make_Features_for_Test(self,dftest):
        
        dftest['Hydro_fire'] = dftest['Horizontal_Distance_To_Fire_Points'] +  dftest['Horizontal_Distance_To_Hydrology']
        dftest['Hydro_Road'] = dftest['Horizontal_Distance_To_Roadways'] +  dftest['Horizontal_Distance_To_Hydrology']
        dftest['Road_fire'] = dftest['Horizontal_Distance_To_Fire_Points'] +  dftest['Horizontal_Distance_To_Roadways']
        dftest['Hydro_fire_sub'] = np.abs(dftest['Horizontal_Distance_To_Fire_Points'] -  dftest['Horizontal_Distance_To_Hydrology'])
        dftest['Hydro_Road_sub'] = np.abs(dftest['Horizontal_Distance_To_Roadways'] -  dftest['Horizontal_Distance_To_Hydrology'])
        dftest['Road_fire_sub'] = np.abs(dftest['Horizontal_Distance_To_Fire_Points'] -  dftest['Horizontal_Distance_To_Roadways'])
        dftest['RAD_SLOPE'] = dftest['Slope'].apply(lambda x: x*(np.pi/180))
        dftest['RAD_Aspect'] = dftest['Aspect'].apply(lambda x: x*(np.pi/180))
        dftest['EL_DIS'] = dftest['Elevation'] - dftest['Horizontal_Distance_To_Hydrology']*0.2
        dftest['EL_Fire'] = dftest['Elevation'] - dftest['Horizontal_Distance_To_Fire_Points']*0.2
        dftest['EL_Road'] = dftest['Elevation'] - dftest['Horizontal_Distance_To_Roadways']*0.2
FE = DataFrame_Feature_Engineering()
FE.Make_Features_for_Train(dftrain)
dftrain.head()
FE.Make_Features_for_Test(dftest)
dftest.head()
class DataFrame_numerical_Imputer():
    

    def __init__(self):
        print("numerical_Imputer object created")

        
   
    def KNN_Imputer(self,df):
        
        """
        This method is for
        imputation, behalf
        of all methods KNN
        imputation performs
        well, hence this method
        will helps to impute
        missing values in 
        dataset
        """
        
        knn_imputer = KNNImputer(n_neighbors=5)
        df.iloc[:, :] = knn_imputer.fit_transform(df)
        return df
imputer = DataFrame_numerical_Imputer()
dftrain = imputer.KNN_Imputer(dftrain)
dftrain.head()
dftrain =  dftrain.drop(['Id'],axis=1)
dftrain.head()
from sklearn.model_selection import KFold, cross_val_score,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x = dftrain.drop(['Cover_Type'],axis=1)
y = dftrain['Cover_Type']
x_train,x_test,y_train,y_test=train_test_split(x\
                ,y,test_size=0.30,random_state=42)

class Model_Selector():
    
    

    def __init__(self,n_estimators=100,\
            random_state=42,max_depth=10):
        print("Model Selector object created")
        
    """
    This method helps to select
    the best machine learning 
    model to compute the relationship
    betweem i/p and d/p variable
    
    """    
        
        
    def Classification_Model_Selector(self,df):
        seed = 42
        models = []
        models.append(("LR", LogisticRegression()))
        models.append(("RF", RandomForestClassifier(n_estimators=100,\
            random_state=42,max_depth=10)))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("CART", DecisionTreeClassifier()))
        models.append(("XGB", XGBClassifier()))
        result = []
        names = []
        scoring = 'accuracy'
        seed = 42
        
        

        for name, model in models:
            kfold = KFold(n_splits = 5, random_state =seed)
            cv_results = cross_val_score(model, x_train,\
                    y_train, cv = kfold, scoring = scoring)
            result.append(cv_results)
            names.append(name)
            msg = (name, cv_results.mean(), cv_results.std())
            print(msg)
            
            
            
        fig = plt.figure(figsize = (8,4))
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(1,1,1)
        plt.boxplot(result)
        ax.set_xticklabels(names)
        plt.show()
MS = Model_Selector()
MS.Classification_Model_Selector(dftrain)
class Data_Modelling():
    

    def __init__(self,n_estimators,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    max_leaf_nodes,
                    bootstrap,
                    class_weight,
                    min_child_weight,
                    learning_rate,
                    Subsample,
                    Alpha,
                    Lamda,
                    random_state,
                    criterion):
        
        self.n_estimators = 500
        self.max_depth = 5
        self.min_samples_split = 3
        self.min_samples_leaf = 3
        self.max_leaf_nodes = None
        self.bootstrap = True
        self.class_weight = 'balanced'
        self.min_child_weight = 3
        self.learning_rate = 0.07
        self.Subsample = 0.7
        self.Alpha = 0
        self.Lamda = 1.5
        self.random_state = 29 
        self.criterion = 'entropy'
        
        print("Data Modelling object created")
        
        
    def Random_Forest_Model(self,df):
        
        Classifier = RandomForestClassifier(n_estimators = 500,
                    max_depth = 5,
                    min_samples_split = 3,
                    min_samples_leaf = 3,
                    max_leaf_nodes = None,
                    bootstrap = True,
                    class_weight= 'balanced',
                    criterion = 'entropy')
        
        Classifier.fit(x_train,y_train)
        
        RF_pred=Classifier.predict(x_test)
        
        print(metrics.accuracy_score(y_test, RF_pred))
        
        print(metrics.confusion_matrix(y_test, RF_pred))
        
        print(metrics.classification_report(y_test, RF_pred))
        
    def Extreme_Gradient_Boosting_Model(self,df):
        
        XGB_Classifier = XGBClassifier(n_estimators = 500,
                    learning_rate = 0.07,
                    max_depth = 5,
                    min_child_weight = 3,
                    random_state = 29,
                    Subsample = 0.7,
                    Alpha = 0,
                    Lamda = 1.5)
        
        XGB_Classifier.fit(x_train,y_train)
        
        XGB_pred=XGB_Classifier.predict(x_test)
        
        print(metrics.accuracy_score(y_test, XGB_pred))
        
        print(metrics.confusion_matrix(y_test, XGB_pred))
        
        print(metrics.classification_report(y_test, XGB_pred))
Basemodell = Data_Modelling(500,5,3,3,None,True,'balanced',3,0.07,0.7,0,1.5,29,'entropy')
Basemodell.Random_Forest_Model(dftrain)
Basemodell.Extreme_Gradient_Boosting_Model(dftrain)
class Model_Classifier_HyperParameter_Tuning():
    

    def __init__(self):
        
        print("HyperParameter_Tuning object created")
        
    class XGB_Classifier_HyperParameter_Tuning():
    

        def __init__(self):

            print("XGB HyperParameter_Tuning object created")


        def Fit_XGB_HyperParameter_Tuner(self,dftrain):
            

            xgb_clf = XGBClassifier(tree_method = "exact", predictor = "cpu_predictor",
                                        objective = "multi:softmax")


            parameters = {"learning_rate": [0.1, 0.01, 0.001],
                           "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
                           "max_depth": [2, 4, 7, 10],
                           "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
                           "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
                           "reg_alpha": [0, 0.5, 1],
                           "reg_lambda": [1, 1.5, 2, 3, 4.5],
                           "min_child_weight": [1, 3, 5, 7],
                           "n_estimators": [100, 250, 500, 1000]}

            from sklearn.model_selection import RandomizedSearchCV

            xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions = parameters, scoring = "f1_micro",
                                         cv = 3, random_state = 29 )

            # Fit the model
            model_xgboost = xgb_rscv.fit(x_train, y_train)
            return model_xgboost
        
        
        def XGB_Get_Best_Prams(self):
            
            print("Learning Rate: ", Xgb_model.best_estimator_.get_params()["learning_rate"])
            print("Gamma: ", Xgb_model.best_estimator_.get_params()["gamma"])
            print("Max Depth: ", Xgb_model.best_estimator_.get_params()["max_depth"])
            print("Subsample: ", Xgb_model.best_estimator_.get_params()["subsample"])
            print("Max Features at Split: ", Xgb_model.best_estimator_.get_params()["colsample_bytree"])
            print("Alpha: ", Xgb_model.best_estimator_.get_params()["reg_alpha"])
            print("Lamda: ", Xgb_model.best_estimator_.get_params()["reg_lambda"])
            print("Minimum Sum of the Instance Weight Hessian to Make a Child: ",Xgb_model.best_estimator_.get_params()["min_child_weight"])
            print("Number of Trees: ", Xgb_model.best_estimator_.get_params()["n_estimators"])


        
        def get_classification_report(self,modelname,y_test):
            
            
            
            ypred = modelname.predict(x_test)
            report = metrics.classification_report(y_test, ypred,output_dict=True)
            print(metrics.confusion_matrix(y_test, ypred))
            print(metrics.accuracy_score(y_test, ypred))
            df_classification_report = pd.DataFrame(report).transpose()
            return df_classification_report
        
        class RF_Classifier_HyperParameter_Tuning():
    

            def __init__(self):

                print("RF HyperParameter_Tuning object created")


            def Fit_RF_HyperParameter_Tuner(self,dftrain):
                
                

                param_grid = {"max_depth": [1, 3, 5, 7, 9, 10],
                              "max_features": [1, 3, 10, 20,40, 50,80],
                              "min_samples_split": [1, 3, 10, 15, 20],
                              "min_samples_leaf": [1, 3, 5, 10],
                              "bootstrap": [True, False],
                              "criterion": ["gini", "entropy"],
                              "n_estimators": [100, 250, 500, 1000]}

                clf = RandomForestClassifier(random_state=29, class_weight='balanced', n_jobs=-1)
                model = RandomizedSearchCV(clf, param_grid, scoring = 'f1_micro', cv=3)

                model.fit(x_train, y_train)

                return model

            def RF_Get_Best_Prams(self):
                
                
                

                print("n_estimators: ", RF_model.best_estimator_.get_params()["n_estimators"])
                print("Max Depth: ", RF_model.best_estimator_.get_params()["max_depth"])
                print("min_samples_split: ", RF_model.best_estimator_.get_params()["min_samples_split"])
                print("min_samples_leaf: ", RF_model.best_estimator_.get_params()["min_samples_leaf"])
                print("max_leaf_nodes: ", RF_model.best_estimator_.get_params()["max_leaf_nodes"])
                print("bootstrap: ", RF_model.best_estimator_.get_params()["bootstrap"])
                print("class_weight: ", RF_model.best_estimator_.get_params()["class_weight"])
                print("criterion: ",RF_model.best_estimator_.get_params()["criterion"])
                print("Number of Trees: ", RF_model.best_estimator_.get_params()["n_estimators"])

            def Evaluation_Report(self,modelname,y_test):
                
                
                return HP_XGB.get_classification_report(RF_model,y_test)
HP_XGB = Model_Classifier_HyperParameter_Tuning().XGB_Classifier_HyperParameter_Tuning()
HP_RF = Model_Classifier_HyperParameter_Tuning().XGB_Classifier_HyperParameter_Tuning().RF_Classifier_HyperParameter_Tuning()
Xgb_model = HP_XGB.Fit_XGB_HyperParameter_Tuner(dftrain)
Xgb_model
HP_XGB.XGB_Get_Best_Prams()
HP_XGB.get_classification_report(Xgb_model,y_test)
RF_model = HP_RF.Fit_RF_HyperParameter_Tuner(dftrain)
RF_model
HP_RF.RF_Get_Best_Prams()
HP_RF.Evaluation_Report(RF_model,y_test)
from sklearn.feature_selection import RFE
from catboost import CatBoostRegressor

class Feature_Selection():

    def __init__(self,n_estimators,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    max_leaf_nodes,
                    bootstrap,
                    class_weight,
                    random_state,
                    criterion):
        
        self.n_estimators = 500
        self.max_depth = 5
        self.min_samples_split = 5
        self.min_samples_leaf = 3
        self.max_leaf_nodes = None
        self.bootstrap = True
        self.class_weight = 'balanced'
        self.random_state = 29
        self.criterion = 'entropy'
        print("Feature Selection object created")
        
    def Classification_Feature_Selector(self,data):
        estimator = RandomForestClassifier(n_estimators = 500,
                    max_depth = 5,
                    min_samples_split = 5,
                    min_samples_leaf = 3,
                    max_leaf_nodes = None,
                    bootstrap = True,
                    class_weight= 'balanced',
                    random_state = 29,
                    criterion = 'entropy')
        
        selector = RFE(estimator,6,step=1)
        selector = selector.fit(x_train,y_train)
        rank =pd.DataFrame(selector.ranking_,\
                        columns=['Importance'])
        Columns = pd.DataFrame(x_train.columns,\
                            columns=['Columns'])
        Var = pd.concat([rank,Columns],axis=1)
        Var.sort_values(["Importance"], axis=0,\
                    ascending=True, inplace=True) 
        return Var
    
    def Feature_visualizer(self,data):
        RF_Selector = RandomForestClassifier(n_estimators = 500,
                      max_depth = 5,
                      min_samples_split = 5,
                      min_samples_leaf = 3,
                      max_leaf_nodes = None,
                      bootstrap = True,
                      class_weight= 'balanced',
                      random_state = 29,
                      criterion = 'entropy')
        
        RF_Selector = RF_Selector.fit(x_train,y_train)
        importances = RF_Selector.feature_importances_
        std = np.std([tree.feature_importances_ for tree \
                          in RF_Selector.estimators_],
                         axis=0)
        indices = np.argsort(importances)[::-1]

            # Print the feature ranking
        print("Feature ranking:")
        for f in range(x_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f],\
                                        importances[indices[f]]))

            # Plot the feature importances of the forest

        plt.figure(1, figsize=(14, 13))
        plt.title("Feature importances")
        plt.bar(range(x_train.shape[1]), importances[indices],
                   color="g", yerr=std[indices], align="center")
        plt.xticks(range(x_train.shape[1]), \
                    x_train.columns[indices],rotation=90)
        plt.xlim([-1, x_train.shape[1]])
        plt.show()    
FSS = Feature_Selection(500,5,5,3,None,True,'balanced',29,'entropy')
FSS.Classification_Feature_Selector(dftrain)
FSS.Feature_visualizer(dftrain)
col_list = ['Elevation','EL_DIS','Road_fire','EL_Fire','soil_type','EL_Road','Horizontal_Distance_To_Roadways','Cover_Type']
col_list1 = ['Elevation','EL_DIS','Road_fire','EL_Fire','soil_type','EL_Road','Horizontal_Distance_To_Roadways']
class Column_Dopper_After_Feature_Selction():

    
    def __init__(self):
        print("Column Dopper object created")
    
    
    def dropper(self,x):
        
        """
        This method helps
        to drop the columns
        in our original 
        dataframe which is 
        available in the 
        col_list and return 
        us final dataset
        """
        
        data=[]
        for i in x.columns:
            if i in col_list:
                data.append(i)
        return x[data]

CD = Column_Dopper_After_Feature_Selction()
New_dftrain = CD.dropper(dftrain)
New_dftrain.head()
New_dftest = CD.dropper(dftest)
New_dftest.head()
x1 = New_dftrain.drop(['Cover_Type'],axis=1)
y1 = New_dftrain['Cover_Type']
x_train1,x_test1,y_train1,y_test1=train_test_split(x1\
                ,y1,test_size=0.30,random_state=42)

RF_Selector = RandomForestClassifier(n_estimators = 250,
                    max_depth = 10,
                    min_samples_split = 3,
                    min_samples_leaf = 3,
                    max_leaf_nodes = None,
                    bootstrap = True,
                    class_weight= 'balanced',
                    criterion = 'entropy')
RF_Selector = RF_Selector.fit(x_train1,y_train1)
rf_pred = RF_Selector.predict(New_dftest)
rf_pred
pred = pd.DataFrame(rf_pred,columns=['Pred'])
pred.head()
import joblib
joblib.dump(RF_Selector,  'RF_Modeljob.pkl',compress=3)
joblib.__version__
RF_Selector = joblib.load('RF_Modeljob.pkl')
RF_Selector
