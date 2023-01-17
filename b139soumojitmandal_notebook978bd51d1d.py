# Pandas program to select the specified columns and rows from a given data frame

import pandas as pd

import numpy as np



exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("Select specific columns and rows:")

print(df.iloc[[1, 3, 5, 6], [1, 3]])
# Program to store height of 50 students in inches. Now while the data was beign recorded manually there has been some typing mistake and therefore height of 2 students have been recorded as 172 inch and 2 students have been recorded as 12 inch. Graphically plot and show how you can seggregate correct data from abnormal data.

from matplotlib import pyplot as plt

heights=[72,71,56,45,67,89,54,58,67,77,77,78,77,73,73,172,72,71,56,45,67,

         89,54,58,67,172,77,78,77,73,73,172,12,54,64,75,75,77,88,66,70,12,54,64,75,75,77,88,66,70]

def plot_his(heights):

    start=min(heights)-min(heights)%10

    end=max(heights)+10

    bins=list(range(start,end,5))

    plt.hist(heights,bins,histtype='bar',rwidth=0.5,color='#FF2400')

    plt.xlabel('heights in inches')

    plt.ylabel('No. of Students')

    plt.title("Heights chart")

    plt.show()

print("Abnormal Data")

plot_his(heights)

heights=list(filter(lambda x: not x==172 and not x==12, heights))

print("Correct Data")

plot_his(heights)

#create a pandas dataframe

#dataframe.sum() method

# import numpy library as np



import numpy as np 



# import pandas library as pd 

import pandas as pd 



# List of Tuples 

students = [('Ankit', 22, 'Up', 'Geu'), 

		('Ankita', np.NaN, 'Delhi', np.NaN), 

		('Rahul', 16, 'Tokyo', 'Abes'), 

		('Simran', 41, 'Delhi', 'Gehu'), 

		('Shaurya', np.NaN, 'Delhi', 'Geu'), 

		('Shivangi', 35, 'Mumbai', np.NaN ), 

		('Swapnil', 35, np.NaN, 'Geu'), 

		(np.NaN, 35, 'Uk', 'Geu'), 

		('Jeet', 35, 'Guj', 'Gehu'), 

		(np.NaN, np.NaN, np.NaN, np.NaN) 

			] 



# Create a DataFrame object from 

# list of tuples with columns 

# and indices. 

details = pd.DataFrame(students, columns =['Name', 'Age', 

										'Place', 'College'], 

						index =['a', 'b', 'c', 'd', 'e', 

								'f', 'g', 'i', 'j', 'k']) 



details 

# Count total NaN at each column in DataFrame.

# import numpy library as np

import numpy as np 



# import pandas library as pd 

import pandas as pd 





# List of Tuples 

students = [('Ankit', 22, 'Up', 'Geu'), 

		('Ankita', np.NaN, 'Delhi', np.NaN), 

		('Rahul', 16, 'Tokyo', 'Abes'), 

		('Simran', 41, 'Delhi', 'Gehu'), 

		('Shaurya', np.NaN, 'Delhi', 'Geu'), 

		('Shivangi', 35, 'Mumbai', np.NaN ), 

		('Swapnil', 35, np.NaN, 'Geu'), 

		(np.NaN, 35, 'Uk', 'Geu'), 

		('Jeet', 35, 'Guj', 'Gehu'), 

		(np.NaN, np.NaN, np.NaN, np.NaN) 

			] 



# Create a DataFrame object from list of tuples 

# with columns and indices. 

details = pd.DataFrame(students, columns =['Name', 'Age', 

										'Place', 'College'], 

						index =['a', 'b', 'c', 'd', 'e', 

								'f', 'g', 'i', 'j', 'k']) 



# show the boolean dataframe			 

print(" \nshow the boolean Dataframe : \n\n", details.isnull()) 



# Count total NaN at each column in a DataFrame 

print(" \nCount total NaN at each column in a DataFrame : \n\n", 

	details.isnull().sum()) 

#Count total NaN at each row in DataFrame .

# import numpy library as np 

import numpy as np 



# import pandas library as pd 

import pandas as pd 





# List of Tuples 

students = [('Ankit', 22, 'Up', 'Geu'), 

		('Ankita', np.NaN, 'Delhi', np.NaN), 

		('Rahul', 16, 'Tokyo', 'Abes'), 

		('Simran', 41, 'Delhi', 'Gehu'), 

		('Shaurya', np.NaN, 'Delhi', 'Geu'), 

		('Shivangi', 35, 'Mumbai', np.NaN ), 

		('Swapnil', 35, np.NaN, 'Geu'), 

		(np.NaN, 35, 'Uk', 'Geu'), 

		('Jeet', 35, 'Guj', 'Gehu'), 

		(np.NaN, np.NaN, np.NaN, np.NaN) 

			] 



# Create a DataFrame object from 

# list of tuples with columns 

# and indices. 

details = pd.DataFrame(students, columns =['Name', 'Age', 

										'Place', 'College'], 

						index =['a', 'b', 'c', 'd', 'e', 

								'f', 'g', 'i', 'j', 'k']) 



# show the boolean dataframe			 

print(" \nshow the boolean Dataframe : \n\n", details.isnull()) 



# index attribute of a dataframe 

# gives index list 



# Count total NaN at each row in a DataFrame 

for i in range(len(details.index)) : 

	print(" Total NaN in row", i + 1, ":", 

		details.iloc[i].isnull().sum()) 
