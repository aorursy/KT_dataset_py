import numpy as np
from scipy.stats import norm, multinomial
original_data = norm.rvs(loc=1.0, scale=0.5, size=1000, random_state=1386)
original_data[:20]
#Now replace every other element with the mean 1.0
missing_elements = np.asarray([0,1]*500)
updated_data = original_data * (1-missing_elements) + missing_elements
updated_data[:20]
#Now, let's get mean and std of the new distribution:
mean, std = norm.fit(updated_data)
print(f'Mean: {mean}, std: {std}')
from sklearn.base import BaseEstimator, TransformerMixin
import numpy.ma as ma
from sklearn.utils.validation import check_is_fitted
class NumericalUnbiasingImputer(BaseEstimator, TransformerMixin):
    """Un-biasing imputation transformer for completing missing values.
        Parameters
        ----------
        std_scaling_factor : number
            We will multiply std by this factor to increase or decrease bias
    """
    def __init__(self, std_scaling_factor=1, random_state=7294):
        self.std_scaling_factor = std_scaling_factor
        self.random_state = random_state

        
    def fit(self, X: np.ndarray, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : NumericalUnbiasingImputer
        """
        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        mask = np.isnan(X)
        masked_X = ma.masked_array(X, mask=mask)

        mean_masked = np.ma.mean(masked_X, axis=0)
        std_masked = np.ma.std(masked_X, axis=0)
        mean = np.ma.getdata(mean_masked)
        std = np.ma.getdata(std_masked)
        mean[np.ma.getmask(mean_masked)] = np.nan
        std[np.ma.getmask(std_masked)] = np.nan
        self.mean_ = mean
        self.std_ = std * self.std_scaling_factor

        return self
    
     
    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, ['mean_', 'std_'])

        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        mask = np.isnan(X)
        n_missing = np.sum(mask, axis=0)
        
        def transform_single(index):
            col = X[:,index].copy()
            mask_col = mask[:, index]
            sample = np.asarray(norm.rvs(loc=self.mean_[index], scale=self.std_[index], 
                                         size=col.shape[0], random_state=self.random_state))
            col[mask_col] = sample[mask_col]
            return col
            
        
        Xnew = np.vstack([transform_single(index) for index,_ in enumerate(n_missing)]).T
        

        return Xnew
    

imputer = NumericalUnbiasingImputer()
missing_indicator = missing_elements.copy().astype(np.float16)
missing_indicator[missing_indicator == 1] = np.nan
data_with_missing_values = original_data + missing_indicator
data_with_missing_values = np.vstack([data_with_missing_values, original_data*5]).T
imputer.fit(data_with_missing_values)
transformed = imputer.transform(data_with_missing_values)
print(transformed[:20,:])
transformed.shape

#Let's see how it is different from the original array:
new_mean, new_std = norm.fit(transformed[:,0])
print(f'Mean: {new_mean}, Std: {new_std}')
import pandas as pd
class CategoricalUnbiasingImputer(BaseEstimator, TransformerMixin):
    """Un-biasing imputation transformer for completing missing values.
        Parameters
        ----------
        std_scaling_factor : number
            We will multiply std by this factor to increase or decrease bias
    """
    def __init__(self, scaling_factor=1, random_state=7294):
        self.scaling_factor = scaling_factor
        self.random_state = random_state

        
    def fit(self, X: np.ndarray, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : NumericalUnbiasingImputer
        """
        if len(X.shape) < 2:
            X = X.reshape(-1,1)

        def fit_column(column):
            mask = pd.isnull(column)
            column = column[~mask]
            unique_values, counts = np.unique(column.data, return_counts=True)
            total = sum(counts)
            probabilities = np.array([(count/total)**(1/self.scaling_factor) 
                    for count in counts])
            total_probability = sum(probabilities)
            probabilities /= total_probability
            return unique_values, probabilities


        self.statistics_ = [fit_column(X[:,column]) for column in range(X.shape[1])]

        return self
    
     
    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, ['statistics_'])

        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        
        def transform_single(index):
            column = X[:,index].copy()
            mask = pd.isnull(column)
            values, probabilities = self.statistics_[index]

            sample = np.argmax(multinomial.rvs(p=probabilities, n=1,
                                         size=mask.sum(), random_state=self.random_state), axis=1)
            column[mask] = np.vectorize(lambda pick: values[pick])(sample);
            return column
            
        
        Xnew = np.vstack([transform_single(index) for index in range(len(self.statistics_))]).T
        

        return Xnew
names = np.array(['one', None, 'two', 'three', 'four', 'one', None, 'one', 'two'])
names = names.reshape(-1,1)
cat_imp = CategoricalUnbiasingImputer(random_state=121)
cat_imp.fit(names)
print(cat_imp.statistics_)
imputed = cat_imp.transform(names)
imputed
titanic = pd.read_csv("../input/train.csv")
titanic.isna().sum(axis=0)
titanic.info()
n_imputer = NumericalUnbiasingImputer()
titanic.Age = n_imputer.fit(titanic.Age.values).transform(titanic.Age.values)
c_imputer = CategoricalUnbiasingImputer()
titanic.Cabin = c_imputer.fit(titanic.Cabin.values).transform(titanic.Cabin.values)
#Let's see how it transformed Age
titanic.Age.head(20)
print(titanic.Age.isnull().sum())
print(titanic.Cabin.isnull().sum())
#Unique values of the Cabin
titanic.Cabin.unique()
