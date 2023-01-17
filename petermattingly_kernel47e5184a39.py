'''

Part A

'''

def model(x, alpha, beta, sigma):

    '''

    A model to apply to the input x

    x should be iterable

    has parameters alpha, beta, sigma

    '''

    import pandas as pd



    #wrap the input in a spreadsheet-like object, a Data Frame

    df_ret = pd.DataFrame(x)



    #this internal function is used to apply the model to all rows in the data frame

    #in an efficient way

    def _model(_x):

        import numpy as np

        #draw a normally distributed epsilon with mean 0, and variance sigma

        epsilon = np.random.normal(scale=sigma**2)



        return (1 + alpha*_x**(-beta) + epsilon)



    #efficiently apply the model to the data

    #and return the values in a data frame

    return df_ret.apply(_model)