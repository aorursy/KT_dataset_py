# importing the library

import pandas as pd   

# importing as 'pd' makes it easier to refer as we can use 'pd' instead of whole 'pandas' 
# Pandas read_csv(filepath) function is used to read the csv file

# Excel files can be read using read_excel() function.

# I have already added the data to this notebook (See top right corner)



# Titanic dataset contains two data files, train.csv, test.csv. We shall import them seperately.

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print("Pandas - We have read the whole data hooman, within fraction of seconds, you slow snail. Haha..")
# We can print the first few and last few rows of the dataset using pandas head(number_of_rows), tail(number_of_rows) function.

# By default, number_of_rows = 5

# Lets print first 3 rows of train and last 5 rows of test dataset 
print("Me - I'll go for you head Pandas")

train.head(3)
print("Me - ...and then for you tail... , I am the mighty Hooman")

test.tail()
# We can get the shape of each dataset as follows:

print("Shape of training data",train.shape)

print("Shape of test data", test.shape)
# Its clear from above that test data has one column less which is the ground truth column.

# To get a list of column names, we do as follows:

print("Train set columns:", train.columns)

print("Test set columns:", test.columns)
# selecting a particular column by its name

train['Name']
# Due to space limitations, only a few columns are shown

# This can also be done in a different way as given below:

train.Name
# Slicing is one of the most important concept

# To select rows from 5-10, we need to do the following:

print("Me - I summon the rows from 5 to 10, bring them to me.")

print("Pandas - At your service, Hooman")

train[5:11]

# Pandas uses last index as exclusive index so if we want rows till 10, we need to write 10+1 i.e. 11.
# This printed all the columns, but if we want a specific column, we can do the following:

train['Name'][0:5]
# To select multiple columns:

train[['Name', 'Age', 'Sex']]
# Illustration

print(type(train[['Name']]))

print(type(train['Name']))
# Pandas provides different ways to deal with columns and rows. loc, iloc are two such very powerful ways.

# loc - used with index names

# iloc - used with index numbers



print("Me - Pandas, show me the magic...")

print("Le Pandas...")

print()

print(train.iloc[5]) # Returns specific row

print(type(train.iloc[5]))
# To get multiple rows

print("Type: ",type(train.iloc[1:5])) # 5 is exclusive i.e. rows from 1-4 will be displayed

train.iloc[1:5]
# if we want a specific row and a specific column

train.iloc[2,1]
# Although most of the time we don't need an iterartor, pandas provides a method, df.iterrow()

# This can be used to iterate the dataframe
# Counterpart of iloc is loc, which allows using index names

train.loc[2]

# Note that 2 is being interpreted as the index name not as index number
# There is a pandas method which provides deatiled analysis of DataFrame.

print("Me - Pandas...Is it everything?")

print("Pandas - Don't under estimate the power of Pandas, you hooman...")

train.describe()
# We can sort the values using df.sort_values() method

train.sort_values('Age')

# We can provide extra arguement ascending = False if we want to sort in descending order. By default ascending =True
# Or if we want to sort using multiple columns, we can do as follows:

train.sort_values(['Age', 'Name'], ascending=[1,0])

# 1- ascending for Age, 0 - Descending for Name
# Let's say we want to select Pclass = 1 passengers...We wanna see how many rich kids

# But before that, let's combine the test and train data into one...

total = train.append(test) # There are other ways too, nbut I find it simple
# Let's see the shape of total

print(total.shape)

print("Yes! We have appended the data successfully")
# Now let' find out the rich kids...:)

total[total['Pclass']==1]
# Ye! We have all the rich kids!!!

# Pandas - But wait, kids?, 58 years old kid? On what planet Hooman?

# So let's find out real kids, lets say of age less than 16
print("Rich kids:")

print()

total[(total['Pclass']==1) & (total['Age']<16)]  # Don't miss the paranthesis, mighty Pandas warns you Hooman
# Before we move further, I, as a responsible creater of this notebook, wanna give you a bonus...:)

print("Finding count of null values in all columns....")

total.isnull().sum()
# One more bonus - We can find unique values and count of each for each column...

total['Pclass'].value_counts()
# Now that we have got a lot of things... I wanna put these rich kids to a seprate class of Pclass=0

# Note that no such class actually exists... We shall make a new class for these rich kids..:(

# To do this, we will summon the mighty loc method...I told you its powerful...

print(type(total))

total.loc[(total['Pclass']==1) & (total['Age']<16), 'Pclass']=0

print("I hope it does the task :(")
# Let's check if we did what we wanted...

print("New Pclass should be zero")

total[(total['Pclass']==0) & (total['Age']<16)]
print("See we added the new Pclass that was not before")

print("To confirm: I summon the bonus I gave to you:::)))")

total['Pclass'].value_counts()
# At last, I wanna combine Parch and SibSp to Family columns as Siblings and Parent/Children are a part of family

# Creaters of Dataset - Yes, we know but we want you to work more :(((

total['Family'] = total['Parch']+total['SibSp']

total.head()
# As you can see at the end there is a column named 'Family'...Hurray.. We united the family...
# Let's find the mean and median of Age

print("Mean of Age:", total['Age'].mean())

print("Median of Age:",total['Age'].median())
# Let's find out max, min of age -- :p...I wanna know the oldest grandpa out there in titanic and youngest seet child

print("Minimum Age:",total['Age'].min())

print("Maximum Age:",total['Age'].max())
# At last, a few more...

print("Sum of all age (IDK why I am finding it): ", total['Age'].sum())
# Aggregation is yet another powerful weapon of Pandas...

print("It gives mean of ages grouped by Pclass:")

total.groupby(['Pclass']).mean()
# Now at last, we want to save the data... But wait another bonus... We want to drop some columns first

# Lets drop SibSp and Parch as they are combined into family already...

total.drop(['Parch','SibSp'],inplace = True, axis=1)

# inplace = True modofies the 'total' inplace. axis = 1 specifies the operation is column-wise

total.head()
# As you can see there is no SibSp and Parch...

# Now let's save this to a new file...

total.to_csv('modified.csv')