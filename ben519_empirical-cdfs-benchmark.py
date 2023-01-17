import numpy as np

import pandas as pd

from kaggle.competitions import nflrush



# run this once!

env = nflrush.make_env()
# read train

train = pd.read_csv(

    filepath_or_buffer = '/kaggle/input/nfl-big-data-bowl-2020/train.csv',

    usecols = ['PlayId', 'NflId', 'NflIdRusher', 'PossessionTeam', 'FieldPosition', 'YardLine', 'Yards'],

    low_memory = False

)
class BenchmarkCDFs():



    def __init__(self):

        self.cdfs = None



    def transform_yardline(self, x):

        # Create YL := transformed version of YardLine on the scale 1 - 100 such that

        # the possessing team is always heading towards YL 100

        # (Here, x should be one row of a DataFrame. This method is meant to be

        #  called by DataFrame.apply())



        if (x.YardLine == 50 or x.PossessionTeam == x.FieldPosition):

            return x.YardLine

        else:

            return 50 + (50 - x.YardLine)



    def fit(self, train):

        # learns the empirical CDF given the current line of scrimmage position (YL)

        # saves the lookup table as self.cdfs



        # Subset rows where the player is the rusher. This should create a complete set of unique PlayIds

        plays = train.loc[train.NflId == train.NflIdRusher].copy()



        # Insert YL (modified YardLine on scale 1 - 99)

        plays.loc[:, 'YL'] = plays.apply(self.transform_yardline, axis = 1)



        # Build lookup table rowset (cdfs)

        dfList = [None] * 99

        for i in range(1, 100):

            # Build dataframe with current YL and all possible Yards

            dfList[i - 1] = pd.DataFrame({

                'YL': i,

                'Yards': np.arange(start = -99, stop = 100, step = 1)

            })



        # Combine into one dataframe

        cdfs = pd.concat(dfList)

        cdfs.set_index(keys = ['YL', 'Yards'], inplace = True)



        # Calculate empirical stats

        empiricals = plays.groupby(['YL', 'Yards']).size()

        counts = plays.groupby('YL').size()

        pdfs = empiricals / counts



        # Merge to cdfs and calculate CDF

        cdfs = cdfs.merge(pdfs.rename('PDF'), how = 'left', left_index = True, right_index = True)

        cdfs.fillna(0, inplace = True)

        cdfs.loc[:, 'CDF'] = cdfs.groupby(['YL'])['PDF'].cumsum()

        cdfs.loc[:, 'CDF'] = np.minimum(1.0, cdfs.CDF.values)



        # Save table to this object

        self.cdfs = cdfs



    def predict(self, test):

        # make predictions for a dataframe of play attributes

        # test should be all the rows associated with a single PlayId (although we'll only use the 1st row)

        # returns a 1-row DataFrame with columns {Yards-99, Yards-98, ... Yards98, Yards99}



        if(self.cdfs is None):

            raise Exception('Call the fit() method first!')



        if(test.PlayId.nunique() != 1):

            raise Exception('test should have a single PlayId!')



        # Extract one row from the test set and insert YL

        temp = test.iloc[[0]].loc[:, ['PlayId', 'PossessionTeam', 'FieldPosition', 'YardLine']].copy()

        temp.loc[:, 'YL'] = temp.apply(self.transform_yardline, axis = 1)

        temp.set_index('YL', inplace = True)



        # Lookup the CDF for the given YL

        cdf = temp.merge(self.cdfs, how = 'left', left_index = True, right_index = True)



        # Format the output

        result = cdf.reset_index().pivot(index = 'PlayId', columns = 'Yards', values = 'CDF')

        result = result.reset_index(drop = True)

        result.rename_axis(None, axis = 1, inplace = True)

        result = result.add_prefix('Yards')

        result.index = list(result.index)  # Convert range index to int index

        return result
# Create benchmark model

mymodel = BenchmarkCDFs()

mymodel.fit(train)



# Loop thru test data and make predictions

for (test_df, sample_prediction_df) in env.iter_test():

    predictions_df = mymodel.predict(test_df)

    env.predict(predictions_df)

    

# Write submisison

env.write_submission_file()