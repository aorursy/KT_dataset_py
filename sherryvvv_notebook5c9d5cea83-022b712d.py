train_df = pd.read_csv('../input/train.csv') 	# Load in the csv file using pd library

                                                # store in a data-frame called train_df 

train_df.head() 						        # To take a peep at the input data, use head() to show the first 5 rows of the input data

                                                # 

                                                # read dictionary in the website for meaning(metadata) of

                                                # each column(attribute)https://www.kaggle.com/c/titanic/data

                                                # link: https://www.kaggle.com/c/titanic/data

# The left most column (not part of the data) is the index of each row

# The top row is the head(name of the attributes)
train_df.info()       # info() is a helpful method in dataFrame class that shows infomation of 

                      # attributes in the dataFrame
train_df.describe()         # describe() is an alternative method in dataFrame class

                            # that shows statistics information 

                            # however, only for those numeric attributes!


# Set some variables

number_passengers = np.size(train_df["PassengerId"].astype(np.float))

number_survived = np.sum(train_df["Survived"].astype(np.float))

proportion_survivors = number_survived / number_passengers 

# shows the survial rate 

print("The survival rate was: {:.2%}".format(proportion_survivors))
# I can now find the stats of all the women on board,

# by making an array that lists True/False whether each row is female

women_only_stats = train_df["Sex"] == "female" 	# This finds where all the women are

men_only_stats = train_df["Sex"] != "female" 	# This finds where all the men are (note != means 'not equal')

type(men_only_stats)                            # Series is a one-dimension array

# men_only_stats.head()                           # now take a peek of the series
train_df[women_only_stats].head()       # by using the Series, women_only_stats, you get all data where

                                        # Sex column is female
women_onboard = train_df[women_only_stats]["Survived"]

men_onboard = train_df[men_only_stats]["Survived"]



# and derive some statistics about them

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)

proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)



print('Proportion of women who survived is %.4f' % proportion_women_survived)

print('Proportion of men who survived is %.4f' % proportion_men_survived)



# Now that I have my indicator that women were much more likely to survive,

# so my rule is thumb for prediction is: women survive, man drown(better luck next time!)
# I am ready to predict now! 

test_df = pd.read_csv('../input/test.csv')       # input the test csv file

# Prediciton means given a set of attribute, determine(more like guess) wheteher the passenger would survive

test_df.head()         # now take a peek at the test data
test = test_df.loc[:, ["PassengerId","Sex"]]

test.head()
# Add another column called Survived for prediction, initializing as 1

test["Survived"] = 1

test.head()
men_only_row = test["Sex"] == "male"

men_only_row.head()
# this doesn't work

# test[men_only_row]["Survived"] = 1



test.loc[men_only_row,["Survived"]] = 0

test.head()
# Last step! The submission requires to output file only with two columns "PassengerId" and "Survived"

# drop the column "Sex"

del test['Sex']

# transform the dataFrame into a csv file

test.to_csv('titanic.csv', index = False)