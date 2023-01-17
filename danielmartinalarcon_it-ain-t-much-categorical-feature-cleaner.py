def cleanup3(X):

    """

    Fixes the categorical features. 

    

    

    Each feature has different specific problems, but they usually have

    garbage values (usually 'unknown') that should really be read as NANs.

    

    This function cleans up garbage, clusters together different labels

    that should be equivalent but are coded differently (e.g., different

    spellings of the same thing), and removes labels with so few members

    that they're unlikely to be informative.

    

    

    Parameters

    ----------

    X : pandas.DataFrame

        DF with raw categorical features, except for the changes

        already included in cleanup1.

    

    Returns

    ----------

    X2 : pandas.DataFrame

         DF with cleaned categorical features.

    

    """

    

    # Make a clean copy, to ensure we're not changing the original DF

    X2 = X.copy()

    

    # Create list of categorical features

    categoricals = df2.select_dtypes(exclude='number').columns.tolist()



    # Make all strings lowercase, to collapse together some of the categories

    X2[categoricals] = X2[categoricals].applymap(lambda x: x.lower())



    # Replace common NAN values

    nan_list = ['not known','unknown','none','-','##','not kno','unknown installer']

    X2 = X2.replace(nan_list, np.nan)



    # Any feature values with fewer than 100 rows gets turned into a NAN

    for feature in X2[categoricals]:

        # Determine which feature values to keep

        to_keep = X2[feature].value_counts()[X2[feature].value_counts() > 100].index.tolist()

        # Turn those into NANs (using a copy, to prevent warnings)

        feature_copy = X2[feature].copy()

        feature_copy[~feature_copy.isin(to_keep)] = np.nan

        X2[feature] = feature_copy



    # Fix all NANs

    X2[categoricals] = X2[categoricals].fillna('other')

    

    

    return X2