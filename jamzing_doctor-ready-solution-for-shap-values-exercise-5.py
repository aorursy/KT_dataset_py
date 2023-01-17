# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#init libraries

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

import eli5

from eli5.sklearn import PermutationImportance

import shap  # package used to calculate Shap values

from IPython.display import display





class patient_data():

    

    features = []

    query = '' # future proofing idea for finer details of patient data analysis (not implemented yet)

    groupby = []

    my_model = RandomForestClassifier()

    

    val_X = pd.DataFrame()

    val_y = pd.DataFrame()

    data_for_prediction = pd.DataFrame()

    data = pd.DataFrame()

    train_X = pd.DataFrame()

    train_y = pd.DataFrame()

    

    @classmethod

    def _initdata(self, train_data):

        self.train_data = train_data

        self.data = pd.read_csv(self.train_data)

       # data.columns

        

        y = self.data.readmitted

        base_features = [c for c in self.data.columns if c != "readmitted"]

        X = self.data[base_features]

        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, y, random_state=1)

        self.my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(self.train_X, self.train_y)

        self.data_for_prediction = self.val_X.iloc[0,:]  # use 1 row of data here. Could use multiple rows if desired

        

    def __init__(self, train_data):

        self.train_data = train_data

        self._initdata( self.train_data)



class patient_risk():

    

    dp = patient_data('../input/hospital-readmissions/train.csv')

  

    def _initdata(self):

        print('PR init')    

    

    def __init__(self):

        self._initdata()

            

    @classmethod        

    def permutation_importance(cls):        

        perm = PermutationImportance( cls.dp.my_model, random_state=1).fit( cls.dp.val_X, cls.dp.val_y)

        display(eli5.show_weights(perm, feature_names = cls.dp.val_X.columns.tolist()) )



    @classmethod     

    def partial_dependence(cls):

    #**Calculate and show partial dependence plot:**

    # Create the data that we will plot

        pdp_goals = pdp.pdp_isolate(model=cls.dp.my_model, dataset = cls.dp.val_X, model_features=cls.dp.val_X.columns.tolist(), feature= cls.dp.features)

    # plot it

        pdp.pdp_plot(pdp_goals, cls.dp.features)

        plt.show()

    

    @classmethod        

    def shap_view(cls):

       # Create object that can calculate shap values

        explainer = shap.TreeExplainer(cls.dp.my_model)

        shap_values = explainer.shap_values( cls.dp.data_for_prediction)

        shap.initjs()

        display(shap.force_plot(explainer.expected_value[0], shap_values[0], cls.dp.data_for_prediction))

        

    @classmethod        

    def show_visuals(cls):

        cls.permutation_importance()

        cls.partial_dependence()

        cls.shap_view()

    

    @classmethod

    def show_risk_analysis(cls):

       # data_analysis = pd.concat([cls.dp.train_X, cls.dp.train_y], axis=1) 

        #data[['time_in_hospital', 'readmitted']]  # use 1 row of data here. Could use multiple rows if desired or choose individual rows in question

        #data_for_prediction.groupby(['time_in_hospital']).head() #.mean()

        cls.dp.data.groupby(cls.dp.groupby ).mean().readmitted.plot()

        plt.show()      

        

        

def patient_risk_factors():

    

      

    try:

        pr = patient_risk() 

        pr.dp.features=['number_inpatient', 'number_diagnoses','num_procedures']

        pr.show_visuals()

        

        pr.dp.groupby = ['time_in_hospital']

        pr.show_risk_analysis()

        

    except Exception as e:

        return print( "Error: ", e )

   

   

#usage:

patient_risk_factors()