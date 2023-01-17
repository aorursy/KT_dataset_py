# Reading the dataset

import pandas as pd



data = pd.read_csv('../input/googleplaystore.csv')

data
# Get all the columns of the dataset

data.columns.tolist() # get the list of the columns
# Get information about the columns 

data.info() # Number of rows, nul/non-null, datatype
# Check if there are any empty values in the dataset

print("Any Empty Values? : ", data.isna().values.any())  # True or False
# Summary of dataset



data.shape # Get the shape of the data (rows, columns) 
# Get all the columns with empty values



print("Columns with empty values : ", data.columns[data.isnull().any()].tolist())
# How many total values versus null values in each column



print("Column | Total Records | No. Of Nulls\n")

for col in data.columns:

    print("%s | %s | %s" % (col, len(data[col]), data[col].isnull().sum()))
import matplotlib.pyplot as plt



# Removing rows with nulls in category and filtering data by installs = 1,000,000,000+

category_installs = data[data['Category'].notnull() & (data['Installs'] == '1,000,000,000+')]

# Group data by category so that we can count number of catgories with installs more than 1 billion.

grouped = category_installs.groupby('Category')['Category'].count()

# Plot the data

grouped.plot(kind='barh')

plt.xlabel('Count')

plt.ylabel('Category')

plt.title('Most installed (Over 1 Billion) categories')

plt.show()
cat_data = data[data['Category'].notnull()]

cat_groups = cat_data.groupby('Category').size().nlargest(10)

cat_groups.plot(kind = 'bar')

plt.ylabel("Count of Apps in category")

plt.xlabel("Category")

plt.title("Top 10 categories")

plt.show()
data = data[data['Installs'] != 0]

data = data[data['Installs'] != 'Free']



# Removing '+' and ',' and converting to int.

data['Installs'] = data['Installs'].apply(lambda num : str(num.replace(',',''))).apply(lambda num : str(num.replace('+',''))).astype('int')



# group by category and sum installs for that category

installs_per_cat = data.groupby('Category')['Installs'].sum().nlargest(10)

installs_per_cat.plot(kind = 'bar')

plt.title("Top 10 installed categories")

plt.xlabel("Categories")

plt.ylabel("Total Installs")

plt.show()
data = data[data['Type'] == 'Paid']

data = data[data['Installs'] != 0]

# # data['Installs'] = data['Installs'].apply(lambda num: str(num.replace(',', ''))).apply(lambda num: str(num.replace('+', ''))).astype('int')

installs_per_cat_paid = data.groupby('Category')['Installs'].sum().nlargest(10)

installs_per_cat_paid.plot(kind = 'bar')

plt.title('Top 10 Paid Categories')

plt.xlabel('Categories')

plt.ylabel('Total Installs')

plt.show()
data = pd.read_csv('../input/googleplaystore.csv')

data = data[data['Category'] == 'FAMILY']

genres = data.groupby('Genres').size().nlargest(10)

genres.plot(kind = 'bar')

plt.xlabel('Genres')

plt.ylabel('Count')

plt.title('Top 10 Genres of Family Category')

plt.show()
data = pd.read_csv('../input/googleplaystore.csv')

data = data[data['Type'] == 'Free']

data = data[data['Installs'] != 0]

data['Installs'] = data['Installs'].apply(lambda num: str(num.replace(',', ''))).apply(lambda num: str(num.replace('+', ''))).astype('int')

free_cat_data = data.groupby('Category')['Installs'].sum().nlargest(10)

free_cat_data.plot(kind = 'bar')

plt.xlabel('Categories')

plt.ylabel('Total Installs')

plt.title('Top 10 Free Categories')

plt.show()
data = pd.read_csv('../input/googleplaystore.csv')

data = data[data['Category'] == 'GAME']

genre_group = data.groupby('Genres').size().nlargest(10)

genre_group.plot(kind = 'bar')

plt.xlabel('Genres')

plt.ylabel('Count')

plt.title('Top 10 genres of Game Category')

plt.show()