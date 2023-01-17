# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read the training data into a DataFrame

training = pd.read_csv("../input/train.tsv", sep = '\t')

# let's see it

training
# group the table entries by category, condition, shipping, then do a mean over the groups

mean_regressor = training.groupby(['category_name', "item_condition_id", "shipping"]).mean().to_dict()['price']

mean_regressor
# the input value will be a tuple of the same format as the mean_regressor keys

def predict(values):

    # check if our product is in our price list by doing a triple match on category, condition and shipping

    if values in mean_regressor:

        return mean_regressor[values]

    # wellp, tough luck on this one

    else:

        return 0.0
# alrighty. let's read the test data

test = pd.read_csv('../input/test.tsv', sep = '\t', header = 0)

# let's see this one as well

test
prices = []

# iterate through all the elements of the test DataFrame

for row in test.itertuples():

    # predict the price based on the required criteria and append it to the price list

    prices.append(predict((row.category_name, row.item_condition_id, row.shipping)))



prices
# alrighty. We have all we need, now let's rebuild a DataFrame for output

df = pd.DataFrame.from_items([('test_id', range(len(prices))), ('price', prices)])

print(df.head(5))
# last but not least ... just create the output .csv

df.to_csv("mean_submission.csv", index=False)