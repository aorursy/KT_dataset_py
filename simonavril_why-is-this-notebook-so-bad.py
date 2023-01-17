import pandas as pd



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# gonna put 0 as price of the test dataset so we dont drop that column

test_df['SalePrice'] = 0

df = pd.concat([train_df, test_df], keys=['train', 'test'])



y_train = df.ix['train']['SalePrice']



df.drop(['SalePrice'], axis=1, inplace=True)



# gonna drop columns with NaN for now

df.dropna(how='any', axis=1, inplace=True)



columns_not_float = []

first_row = df.iloc[0]

for name, value in first_row.iteritems():

    print(name, value)

    try:

        int(value)

    except:

        columns_not_float.append(name)

    

df = pd.get_dummies(df, columns = columns_not_float )



print("Columns not float %s" % columns_not_float)



X_train, X_test, y_train, y_test = train_test_split(df.ix['train'], y_train, test_size=0.4, random_state=0)



lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

print(lr.score(X_test, y_test))

submit_y = lr.predict(df.ix['test'])

submit_df = pd.DataFrame()

submit_df['Id'] = df.ix['test']['Id']

submit_df['SalePrice'] = submit_y



submit_df.to_csv('/tmp/house_price_simple_dummy.csv', index = False)
