# Hint: this platform is called a jupyter notebook, which is a popular software used to prototype python code. 
# A jupyter notebook is organised into cells
# [Ctrl] + [Enter] to run cells containing python code

# Run your first python code with [Ctrl] + [Enter]
print('Hello World!')
type(5)
type(5.7)
type(True)
type(False)
type('abc')
type(['a', 'b', 'c'])
type({'1':'one', '2':'two'})
type('1847.492')
# Is this a float?
type({1:['OCBC','DBS', 'UOB']})
# Is this a string?
type(false)
# Is this a boolean?
# Hint: you can use the "help" function to understand other functions
help(type)
# Hint: You can comment code using the "#" symbol. Any code behind the "#" symbol will not be recognised

# this is a comment
"""this is not a comment, this is a string""" # this is a comment
a = 5 # assign 5 to the variable "a"
a # return the variable "a"
# Question: what is the type of the variable "a"? (run to find out)
type(a)
b = a # assign the variable "a" to the variable "b"
b
# Hint: to delete a variable (which will clear your RAM), assign None to the variable
a = None
print(a)
b + 1 # add 1 to the variable "b"
b # b is still 5
b = b + 1 # add 1 to the variable "b" AND make this the new "b"
b # b is now 6
# Hint: cells in the jupyter notebook returns only the last result. Use the print() function to see what is in-between
a = 1 # step 1
print(a) # print result after step 1
a = a + 1 # step 2
print(a) # print result after step 2
a = a * 2 # step 3
print(a) # print result after step 3
print('is the final result')
# Is 2 greater than 1?
2 > 1
# Is 1 greater than 2?
1 > 2
# Hint: True is actually equal to 1, and False is equal to 0
1 == True
# Quiz: what does this code do? (run to find out)
a = 1
a = (a == True)
a = a * 2
a = (a > 1)

print(a)
# assign a list of strings to the variable "planets"
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
# the first planet in the list is:
planets[0]
# the last planet in the list is:
planets[-1]
# Quiz: what does this code do? (run to find out)
print('The planet closest to the sun is',planets[0])
print(planets[-4],'and',planets[-3], 'are also known as Gas Giants')
# indexing can also be done on strings
earth = planets[2] # assign the second indexed element of the variable "planets" to the variable "earth"
earth[1] # get the first indexed element of the variable "earth"
# The first three planets are:
planets[0:3]
planets[:3] # If we omit the starting index, it is assumed to be 0
# The rest of the planets are:
planets[3:]
# Quiz: what does this code do? (run to find out)
planets[-999999:]
# Replace the third indexed (fourth) planet
planets[3] = 'SpaceX'
planets
# Replace the first 3 planets
planets[:3] = ['M','V','E']
planets
# Put them back
planets[:4] = ['Mercury','Venus','Earth','Mars']
planets
fruits_i_like = ['Durian','Rambutan']
fruits_i_dislike = ['Soursop','Oranges']
fruits = fruits_i_like + fruits_i_dislike # combine lists using the "+" operator
fruits
planets.append('Pluto') # add Pluto to the list of planets
planets
planets.remove('Pluto') # removing 'Pluto'
planets
# Hint: in-built python Methods directly modify the variable, and do not return anything
nothing = planets.append('Pluto') # if we do this,
print(nothing) # None assigned (returned) to the variable "nothing"
print(planets) # but the variable "planets" still got appended!
# Quiz: what does this code do? (run to find out)
planets + planets * 100
# Quiz: what does this code do? (run to find out)
number_list = [1, 2, 3, 4, 5]
number_list * 2
# Hint: elements can also be removed using indexing/slicing!
planets = planets[1:] # remove the first element of the "planets" list by slicing and reassigning the slice to the same variable
planets
# assign a list of strings to the variable "planets"
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

for one_planet in planets: # for every planet in the "planets" list,
    print(one_planet) # print the planet
# the same can be done for strings
for alphabet in planets[2]: # for every alphabet in the thrid planet,
    print(alphabet) # print the alphabet
# Quiz: what does this code do? (run to find out)
for one_planet in planets:
    print('this is planet ' + one_planet)
# Hint: the range() function can be used to generate integers in sequence
list( range(0, 10) )
# Quiz: what does this code do? (run to find out)
for index in range(0, 3):
    planets[index] = index

print(planets)
import math # import the math library, a standard python library, into the RAM
# Use the exponential function from the math library
math.exp(1)
# Quiz: what does this code do? (run to find out)
integer_list = []
for integer in range(0, 11):
    integer_list.append(math.exp(integer))

integer_list
import pandas as pd # import the pandas library, abbreviated as the variable "pd"
import numpy as np # import the numpy library, abbreviated as the variable "np"
df = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
display(df) # Hint: This display() function is similar to the print() function but is unique to Jupyter notebooks
ramen_df = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv")
ramen_df.head()
ramen_df.shape
ramen_df.size
ramen_df['Style'].value_counts()
ramen_df.describe(include='all')
# If you are wondering how to know what the syntax of a function is, you can hold "shift" + "tab" 
# and a prompt will appear on your screen with more information about the item you are looking at! 
# You can also use the help function eg. help(ramen_df.describe)
ramen_df.dtypes
ramen_df['Country']
ramen_df['Country'][0]
ramen_df.loc[0]
ramen_df.loc[0:3] 

# note that unlike with list slicing, both ends of the range are 
# inclusive. ie. loc will pull out index 0,1,2,3 instead of 
# stopping at index 2
ramen_df.loc[:, 'Country']

# Quiz: Recall what we learn about list slicing. What does ":" mean? Why did we use it here?
ramen_df.loc[:, ['Stars','Country','Brand']]

# Hint: An interesting thing to note here is that loc will extract the columns in the sequence in which you indicate the headers.

# Can you think of any situations in which this would be useful?
# Quiz: What is the expected output of this code?

ramen_df.loc[1, ['Country']]
ramen_df[ramen_df['Country'] == 'Japan']

# As you can see, we have now filtered 352 rows out from the original 2580
# Practice Question: Filter the ramen_df DataFrame to obtain only rows that have more than 3 stars.

suanla = pd.DataFrame([{"Review #":2581,"Brand":"Hai Chi Jia","Variety":"Suan La Fen","Style":"Cup","Country":"China","Stars":5}])

ramen_df = ramen_df.append(suanla,ignore_index=True)

ramen_df
fruit_df = pd.DataFrame({"Fruit":["Banana","Apple", "Cherry"],"Origin":["Indonesia","Japan","United States"]})
price_df = pd.DataFrame({"Fruit":["Apple", "Banana", "Cherry"],"Price":[0.5, 1,5]})

display(fruit_df)
display(price_df)
price_df.merge(fruit_df,on='Fruit') # Hint: Note that the sequence of values in the 'Origin' column has now changed.
price_df.rename(columns={"Price":"Cost"})
ramen_df = ramen_df.drop(2580)
display(ramen_df)
price_df.drop(columns = "Price")
ramen_df['Stars'].mean()
# Quiz: What happens if we try to obtain the sum of all the values in the Stars column?

ramen_df['Stars'].sum()
ramen_df['Stars'].unique()
# Quiz: Recall our exercise to filter the dataframe by country what is the main difference here?

ramen_df[(ramen_df['Stars'] != 'Unrated')]
ramen_df = ramen_df[(ramen_df['Stars'] != 'Unrated')]
ramen_df['Stars'].unique()
ramen_df['Stars'] = ramen_df['Stars'].astype('float')
ramen_df['Stars'].mean()
ramen_df['Top Ten'].unique()
#Step 1 replace '\n' with nan

ramen_df['Top Ten'].replace('\n', np.nan, inplace=True) 

# Hint(1): The "inplace" parameter indicates wheter we want to make the change in the dataframe itself, or whether to do it on a copy.
# Hint(2): An alternative way to get the same outcome would be to exclude the "inplace" and use variable assignment instead.
# Hint(3): i.e. ramen_df['Top Ten'] = ramen_df['Top Ten'].replace('\n', np.nan)

ramen_df['Top Ten'].unique()
#Step 2: use fillna to replace all nan values with 'No'

ramen_df['Top Ten'] = ramen_df['Top Ten'].fillna('No')

ramen_df['Top Ten'].unique()
ramen_df.rename(columns={'Stars':'Stars_Raw'})
ramen_df['Stars_Adj'] = ramen_df['Stars']*0.9
ramen_df
ramen_df['Country'].nunique()
ramen_grouped = ramen_df.groupby('Country', as_index = False).mean() 

# Hint 1: The aggregation methods only apply to numeric columns

ramen_grouped
ramen_grouped2 = ramen_df.groupby(['Country','Style'], as_index = False).size()
ramen_grouped2
ax = ramen_df['Stars'].plot.hist()

ax.set_title('Ramen Stars Histogram')
ax = ramen_df['Country'].value_counts().plot.barh(figsize=(10,15))
ax.set_title('Ramen Count by Country')
ax = ramen_df['Stars'].plot.box()

ax.set_title('Ramen Stars Box Plot')
import matplotlib.pyplot as plt
ramen_grouped.sort_values('Stars_Adj',inplace=True) # Sorting the rows of the grouped dataset by the values in the 'Stars_Adj Column'

ramen_grouped.reset_index(inplace=True,drop=True) # Resetting the row index to fix the new order of the DataFrame

# Step 1:
fig, ax = plt.subplots(figsize=(25,10)) # First, we create an empty plot

# Step 2:
ax.hlines(y=ramen_grouped['Country'], # Now we create horizontal lines for each country on the y axis
          xmin=1,                     # Setting the minimum limit for the x axis
          xmax=4,                     # Setting the maximum limit for the x axis
          color='gray',               # Setting the horizontal lines to be gray in colour
          alpha=0.7,                  # Reducing the transparency of the horizontal lines to 0.7 so that the chart will be more readable
          linewidth=1,                # Specifying the width of the horizontal lines
          linestyles='dashdot')       # Specifying the line style, default is solid line

# Step 3:
ax.scatter(y=ramen_grouped['Country'],   # y value for the scatter plot
           x=ramen_grouped['Stars_Adj'], # x value for the scatter plot
           s=75,                         # size of the markers
           color='firebrick',            # Color of the markers
           alpha=0.7)                    # Setting the transparency of the markers to 0.7

ax.set_title('Average Stars By Country')

ax.set_xlabel('Stars')

ax.set_ylabel('Country')
pivot = ramen_grouped2.pivot(index='Country',columns='Style',values='size') # Creating a pivot table from the DataFrame
pivot.head()
pivot = pivot.fillna(0).astype(int) # replace the NaN values with 0 and change the datatype from float to integer
pivot.head()
pivot.plot.bar(stacked=True,figsize=(25,5),title='Count of Ramen By Style and Country',ylabel='Count')
fig, axes = plt.subplots(nrows=4,ncols=2) # Create a chart of 8 charts in a 4 by 2 arrangement

fig.set_figheight(25) # Setting the height of the chart. Note that this is not the height of each individual subplot
fig.set_figwidth(25)  # Setting the height of the chart. Note that this is not the height of each individual subplot


pivot['Bar'].plot(ax=axes[0,0],kind='bar',title='Count of Bar Ramen by Country',color='blue') # Plotting the first bar chart in the subplot. 
                                                                                              # Note that we must first indicate the index of the subplot we want to plot in

pivot['Bowl'].plot(ax=axes[0,1],kind='bar',title='Count of Bowl Ramen by Country',color='orange')

pivot['Box'].plot(ax=axes[1,0],kind='bar',title='Count of Box Ramen by Country',color='green')

pivot['Can'].plot(ax=axes[1,1],kind='bar',title='Count of Can Ramen by Country',color='red')

pivot['Cup'].plot(ax=axes[2,0],kind='bar',title='Count of Cup Ramen by Country',color='purple')

pivot['Pack'].plot(ax=axes[2,1],kind='bar',title='Count of Pack Ramen by Country',color='magenta')

pivot['Tray'].plot(ax=axes[3,0],kind='bar',title='Count of Tray Ramen by Country',color='pink')

fig.delaxes(axes[3,1]) # deleting the 8th subplot since we only need 7

plt.tight_layout() # Setting the layout of the subplots so that they do not overlap
