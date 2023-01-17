from sklearn.preprocessing import OneHotEncoder
def ohe(train,features):
    """
    The functions takes the df with the list of arguments to be one hot encoded and returns a df
    train is the df and features is a list
    """
    for v in features:
        df = train[v].values
        train = train.drop([v],axis=1)
        oh = OneHotEncoder(sparse=False)
        df = df.reshape(len(df),1)
        df = oh.fit_transform(df)
        df = df[:,1:]
        train = pd.concat([train,pd.DataFrame(df)],axis=1,sort=False)
    return train