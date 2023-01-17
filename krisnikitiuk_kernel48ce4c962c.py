import pandas as pd



def main():

    train_dataframe = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', sep=',')

    train_df = PrepareData(train_dataframe).prepare_test_data()



    RandomForest(train_df).run_model()





if __name__ == "__main__":

    main()
import pandas as pd





class PrepareData(object):



    def __init__(self, dataframe):

        self.dataframe = dataframe



    def _encode_str_features(self, dataframe):



        cat_feat = list(dataframe.dtypes[dataframe.dtypes == object].index)

        num_feat = dataframe.select_dtypes(include='int')



        dataframe[cat_feat] = dataframe[cat_feat].fillna('nan')

        encoded_data = pd.get_dummies(dataframe[cat_feat], columns=cat_feat)

        encoded_cols = list(set(encoded_data))

        encoded_data = encoded_data[encoded_cols]



        prepared_data = pd.concat([encoded_data, num_feat.fillna(-999)], axis=1)



        return prepared_data



    def prepare_test_data(self):

        encoded_df = self._encode_str_features(self.dataframe)

        return encoded_df

from datetime import datetime



from sklearn import feature_selection

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd





class RandomForest(object):



    def __init__(self, train_dataframe):

        self.train_dataframe = train_dataframe



    def _result_to_csv(self, X_test, y_pred):

        d = {'Id': X_test['Id'], 'SalePrice': y_pred}

        data = pd.DataFrame(d, columns=['Id', 'SalePrice'])

        data.to_csv('result.csv', index=False)



    def _calculate_chi2(self, dataframe):

        chi2_selector = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=50)

        X = dataframe.drop(['Id', 'SalePrice'], axis=1)

        y = dataframe['SalePrice']

        X_kbest = chi2_selector.fit_transform(X, y)

        print('Reduced number of features:', X_kbest.shape[1])

        return X_kbest



    def run_model(self):

        X_train = self._calculate_chi2(self.train_dataframe)

        dataset = pd.DataFrame(X_train)

        X_train = pd.concat([self.train_dataframe['Id'], dataset], axis=1)



        y_train = self.train_dataframe['SalePrice']

        dt = datetime.now()

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=dt.second)



        regressor = RandomForestRegressor(n_estimators=100, random_state=dt.second, max_depth=25)

        regressor.fit(X_train, y_train)



        y_pred = regressor.predict(X_test)

        self._result_to_csv(X_test, y_pred)



        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))