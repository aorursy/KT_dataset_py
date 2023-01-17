import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statsmodels.regression.linear_model import OLS

from statsmodels.tools import add_constant

from itertools import combinations
df = pd.read_csv('/kaggle/input/sat-score-data-by-state/Guber1999data.csv')

df.head()
X = df[["Spend"]]

y = df["SATT"]

regr = OLS(y, add_constant(X)).fit()

regr.summary()
# Seaborn visualization library

import seaborn as sns

# Create the default pairplot

sns.pairplot(df);
likelihood1 = np.exp(-regr.bic/2)
print(np.log(likelihood1))

print(regr.llf)
X = df[["Spend", "PrcntTake"]]

y = df["SATT"]

regr = OLS(y, add_constant(X)).fit()

likelihood2 = np.exp(-regr.bic/2)

regr.summary()
print(likelihood2/likelihood1)
X = df[["Spend", "StuTeaRat", "Salary", "PrcntTake"]]

y = df["SATT"]

regr = OLS(y, add_constant(X)).fit()

bic4 = regr.bic

likelihood4 = np.exp(-bic4/2)

regr.summary()
class BMA:

    

    def __init__(self, y, X, **kwargs):

        # Setup the basic variables.

        self.y = y

        self.X = X

        self.names = list(X.columns)

        self.nRows, self.nCols = np.shape(X)

        self.likelihoods = np.zeros(self.nCols)

        self.coefficients = np.zeros(self.nCols)

        self.probabilities = np.zeros(self.nCols)

        self.names = list(X.columns)

        # Check the max model size. (Max number of predictor variables to use in a model.)

        # This can be used to reduce the runtime but not doing an exhaustive sampling.

        if 'MaxVars' in kwargs.keys():

            self.MaxVars = kwargs['MaxVars']

        else:

            self.MaxVars = self.nCols  

        # Prepare the priors if they are provided.

        # The priors are provided for the individual regressor variables.

        # The prior for a model is the product of the priors on the variables in the model.

        if 'Priors' in kwargs.keys():

            if np.size(kwargs['Priors']) == self.nCols:

                self.Priors = kwargs['Priors']

            else:

                print("WARNING: Provided priors error.  Using equal priors instead.")

                print("The priors should be a numpy array of length equal tot he number of regressor variables.")

                self.Priors = np.ones(self.nCols)  

        else:

            self.Priors = np.ones(self.nCols)  

        

    def fit(self):

        # Perform the Bayesian Model Averaging

        

        # Initialize the sum of the likelihoods for all the models to zero.  

        # This will be the 'normalization' denominator in Bayes Theorem.

        likelighood_sum = 0

        

        # To facilitate iterating through all possible models, we start by iterating thorugh

        # the number of elements in the model.  

        for num_elements in range(1,self.MaxVars+1): 

            

            # Make a list of all index sets of models of this size.

            model_index_sets = list(combinations(list(range(self.nCols)), num_elements)) 

            

            # Iterate through all possible models of the given size.

            for model_index_set in model_index_sets:

                

                # Compute the linear regression for this given model. 

                model_X = self.X.iloc[:,list(model_index_set)]

                model_regr = OLS(self.y, model_X).fit()

                

                # Compute the likelihood (times the prior) for the model. 

                model_likelihood = np.exp(-model_regr.bic/2)*np.prod(self.Priors[list(model_index_set)])

                print("Model Variables:",model_index_set,"likelihood=",model_likelihood)

                # Add this likelihood to the running tally of likelihoods.

                likelighood_sum = likelighood_sum + model_likelihood

                

                # Add this likelihood (times the priors) to the runny tally

                # of likelihoods for each variable in the model.

                for idx, i in zip(model_index_set, range(num_elements)):

                    self.likelihoods[idx] = self.likelihoods[idx] + model_likelihood

                    self.coefficients[idx] = self.coefficients[idx] + model_regr.params[i]*model_likelihood



        # Divide by the denominator in Bayes theorem to normalize the probabilities 

        # sum to one.

        self.probabilities = self.likelihoods/likelighood_sum

        self.coefficients = self.coefficients/likelighood_sum

        

        # Return the new BMA object as an output.

        return self

        

    def summary(self):

        # Return the BMA results as a data frame for easy viewing.

        df = pd.DataFrame([self.names, list(self.probabilities), list(self.coefficients)], 

             ["Variable Name", "Probability", "Avg. Coefficient"]).T

        return df  
X = df[["Spend", "StuTeaRat", "Salary", "PrcntTake"]]

y = df["SATT"]

result = BMA(y,add_constant(X)).fit()

result.summary()
# This is an excessively commented version of the code for Bayeisan Model Averaging.



class BMA:

    # After the class definition, we have a sequence of 'methods' associated with the class, 

    # which are just functions that are connected to the class and get internal access 

    # to the internal variables of the class.  The 'self' variable within the class always 

    # refers to class itself.  Varialbes with names of the form self.varname will be 

    # accesible in the methods within the class, and accesible outside the class as 

    # classname.varname.

    

    def __init__(self, y, X, **kwargs):

        # This __init__ function is the initilization method and runs when the class 

        # is created. Here we just use it to attached the X and y variables to the class, 

        # compute basic shape variables and attach them, and build some placeholder 

        # variables for our BMA analysis.  The variable names are self-explanatory.

        self.y = y

        self.X = X

        self.names = list(X.columns)

        self.nRows, self.nCols = np.shape(X)

        self.likelihoods = np.zeros(self.nCols)

        self.coefficients = np.zeros(self.nCols)

        self.probabilities = np.zeros(self.nCols)

        self.names = list(X.columns)

        # Check the max model size. (Max number of predictor variables to use in a model.)

        # This can be used to reduce the runtime but not doing an exhaustive sampling.

        if 'MaxVars' in kwargs.keys():

            self.MaxVars = kwargs['MaxVars']

        else:

            self.MaxVars = self.nCols  

        # Prepare the priors if they are provided.

        # The priors are provided for the individual regressor variables.

        # The prior for a model is the product of the priors on the variables in the model.

        if 'Priors' in kwargs.keys():

            if np.size(kwargs['Priors']) == self.nCols:

                self.Priors = kwargs['Priors']

            else:

                print("WARNING: Provided priors error.  Using equal priors instead.")

                print("The priors should be a numpy array of length equal tot he number of regressor variables.")

                self.Priors = np.ones(self.nCols)  

        else:

            self.Priors = np.ones(self.nCols)  

        

    def fit(self):

        # In this fit method, we are doing our model averaging.  This is a Baeysian 

        # process, where in general we consider different values of the parameters and 

        # use Bayes Theorem to compute a probability for each set of parameter values, 

        # resulting in a probability distribution for the parameters.

        

        # The parameters we want to estimate is whether or not to include each available 

        # predictor variable.  This means we want to assign a probability to the options 

        # {include, do not include} for each variable. This gives the probability of 

        # including the variable in the model.

        

        # From a Bayesian Statistics sampleing perspective, we are going to compute all 

        # possible parameter combinations as the default.  This is only feasible for 

        # problems with only a few predictor variables.  The keyward MaxVars can be used 

        # to reduce the number of models.

        

        # We initialize the sum of the likelihoods for all the models to zero.  

        # This will be the denominatory in Bayes Theorem, and we will apply it to 

        # normalize in the end.

        likelighood_sum = 0

        

        # To facilitate iterating through all possible models, we start by iterating thorugh

        # the number of elements in the model.  The number of elements is the number of 

        # predictor variables.

        for num_elements in range(1,self.MaxVars+1): 

            

            # Make a list of all index sets of models of this size.

            # For example, if there are 4 predictor variables, this will output the list

            # [(0,1), (0,2), (0,3), (1,2), (1,2), (2,3)].

            model_index_sets = list(combinations(list(range(self.nCols)), num_elements)) 

            # We now iterate through all possible models of the given size.

            for model_index_set in model_index_sets:

                

                # This is where the model averaging happens.

                # First, we compute the linear regression for this given model.  

                # (In other words, we select the set of input variables in model_index_set, 

                # and compute the linear regression model using just these variables.)  

                # We do this using OLS from the statsmodels package.  In our notation, 

                # henceforth any variable beginning with model_ is a variable that is 

                # just for this specific model.

                model_X = self.X.iloc[:,list(model_index_set)]

                model_regr = OLS(self.y, model_X).fit()

                

                # We compute the likelihood for the model from the BIC provided by OLS. 

                # This could alternatively be computed using AIC, the likelihood provided 

                # by OLS, or from RSS using the formula described previously.

                # NOTE:  This is actually the likelihood times the prior.

                model_likelihood = np.exp(-model_regr.bic/2)*np.prod(self.Priors[list(model_index_set)])

                print("Model Variables:",model_index_set,"likelihood=",model_likelihood)

                # Add this likelihood to the running tally of likelihoods for the denominator

                # in Bayes theorem.

                likelighood_sum = likelighood_sum + model_likelihood

                

                # The key step in model averaging for regression is that for each 

                # predictor variable, the probability that the variable should be included 

                # is the sum of all probabilities for all models that include the given 

                # variable.  This is equal to the sum of likelihoodsfor all models 

                # that include the given variable divided by the total likelihood.  

                

                # The other component of model average is that we can compute the 

                # average value of any any varaible over all the models, where this 

                # average is weighted by the probability of each model.  This gives the 

                # expected value for the variable given all the models being considered. 

                # In the following loop we compute the average value for the coefficients

                # for each variable.

                

                # following loop we iterate through all variables in the model 

                # (using their indexes), add the likelihood for the model to the vaiable 

                # tracking the likelihoods for each variable present in the model.  

                # We also add the coefficent for each variable (weighted by 

                # its likelihood) to the variable for the averaged coefficients.

                # [NOTE: idx is the index for the predictor variable within the set of 

                # all predictor variables and i is the index for this same 

                # predictor variable in the current regression model.]

                for idx, i in zip(model_index_set, range(num_elements)):

                    self.likelihoods[idx] = self.likelihoods[idx] + model_likelihood

                    self.coefficients[idx] = self.coefficients[idx] + model_regr.params[i]*model_likelihood



        # Now we divide by the denominator in Bayes theorem to normalize the probabilities to one.

        self.probabilities = self.likelihoods/likelighood_sum

        self.coefficients = self.coefficients/likelighood_sum

        

        # Having updated all the internal varaibles with our Bayeisan Model Averaing, 

        # we return the new BMA object as an output.

        return self

        

    def summary(self):

        # This function just takes the Bayesian Model Averaging analysis and returns

        # it as a data frame which makes it easier to veiw.

        df = pd.DataFrame([self.names, list(self.probabilities), list(self.coefficients)], 

             ["Variable Name", "Probability", "Avg. Coefficient"]).T

        return df        