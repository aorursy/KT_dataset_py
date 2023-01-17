from sklearn.impute import MissingIndicator



def cleanup2(X):

    """

    Fixes the numerical features. 

    

    

    Each feature has different specific problems, but they usually have

    garbage values (usually zero) that should really be read as NANs.

    

    I want to fix those values, but I also want to take note of the 

    datapoints where they happened.  I do this because I assume that 

    missing values tell us something about the well that our model

    might be able to pick up later.

    

    

    Parameters

    ----------

    X : pandas.DataFrame

        DF with raw numerical features

    

    Returns

    ----------

    X2 : pandas.DataFrame

         DF with cleaned numerical features and a new matrix of former

         garbage locations within those features.

    

    """

    

    

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    # I make a list of the numerical columns and a dict of their 

    # garbage values that really should be nulls

    numericals = ['amount_tsh',

                    'date_recorded',

                    'gps_height',

                    'longitude',

                    'latitude',

                    'num_private',

                    'population',

                    'construction_year']



    null_values = {'amount_tsh':0,

                     'date_recorded':0,

                     'gps_height':0,

                     'longitude':0,

                     'latitude':-2.000000e-08,

                     'num_private':0,

                     'population':0,

                     'construction_year':0}



    # I replace all garbage values with NANs.

    for feature, null in null_values.items():

        X2[feature] = X2[feature].replace(null, np.nan)



    # construction_year occasionally claims years far in the future, and 

    # could presumably also contain years way in the past.  I'll turn anything

    # not between 1960 and 2019 into a NAN.

    X2['construction_year'] = [i if 1960 < i < 2019 else np.nan for i in X2['construction_year']]

    

    

    # Creating indicator columns.

    # ---------------------------------------------------------------

    # These columns mark the locations of all the NANs 

    # in the numericals. Note that MissingIndicator returns a numpy array.

    

    indicator = MissingIndicator()

    trash_array = indicator.fit_transform(X2[numericals]) # Bool array

    trash_array = trash_array.astype('float64')     # Float64 array



    # Create a titles for the columns in num_trashmarker

    trashy_names = [numericals[i] + '_trash' for i in indicator.features_]



    # Create num_trashmarker

    trash_df = pd.DataFrame(trash_array, columns=trashy_names)



    # I add trash_df to X2

    X2 = pd.concat([X2,trash_df], sort=False, axis=1)

    

    

    # Fixing the numerical columns.

    # ---------------------------------------------------------------

    # Whenever possible, a good replacement value for a NAN is the 

    # mean or median value for the geographic region around it.



    # Replaces the NANs in a ward with the mean of the other rows in that 

    # same ward. If all the rows in a ward are NANs, though, they remain.

    for feature in numericals:

        replacements = X2.groupby('ward')[feature].transform('mean')

        X2[feature] = X2[feature].fillna(replacements)



    # Replaces the NANs in a region with the mean of the other rows in that 

    # same region (which are much larger than wards)

    for feature in numericals:

        replacements = X2.groupby('region')[feature].transform('mean')

        X2[feature] = X2[feature].fillna(replacements)

    

    return X2