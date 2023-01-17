import pandas as pd # importing pandas = > useful for creating dataframes
import numpy as np   # importing numpy = > useful for creating numpy arrays 

x1 = [1, 2, 3, 4, 5] # list format 
x2 = [10, 11, 12, 13]  # list format 

# Creating a data frame using explicits lists
X = pd.DataFrame(columns = ["X1","X2"]) 
X

X["X1"] = pd.Series(x1) # Converting list format into pandas series format
X["X2"] = pd.Series(x2) # Converting list format into pandas series format

# accessing columns using "." (dot) operation
X.X1
X["X1"]
X[["X1","X2"]]
# Accessing elements using ".iloc" : accessing each cell by row and column 
# index values
X.iloc[0:3,1]
X.iloc[:,:] # to get entire data frame 
# checking the type of variable 
type(X.X1) # pandas series object

# to create a data frame 
x = pd.DataFrame(columns=["A","B","C"])
# np.random.randint(a,b,c) 
# a - > starting number
# b - > Ending number
# c - > no. of numbers to be generated 
x["A"] = pd.Series(list(np.random.randint(1,100,50)))

# np.random.choice([a,b],size=c)
# a and b = > choosing elements from a or b 
# c = > number of elements to be generated choosing from a or b
x["B"] = pd.Series(list(np.random.choice([0,1],size = 50)))
x["C"] = 10 # going to fill all the rows in "C" with value 10
mba = pd.read_csv('../input/mbamtcars/mba.csv')





# Importing necessary libraries
import pandas as pd # importing pandas = > useful for creating dataframes
import numpy as np   # importing numpy = > useful for creating numpy arrays 


x1 = [1, 2, 3, 4, 5] # list format 
x2 = [10, 11, 12, 13]  # list format 

# Creating a data frame using explicits lists
X = pd.DataFrame(columns = ["X1","X2"]) 
X

X["X1"] = pd.Series(x1) # Converting list format into pandas series format
X["X2"] = pd.Series(x2) # Converting list format into pandas series format

# accessing columns using "." (dot) operation
X.X1
# accessing columns alternative way
X["X1"]

# Accessing multiple columns : giving column names as input in list format
X[["X1","X2"]]

# Accessing elements using ".iloc" : accessing each cell by row and column 
# index values
X.iloc[0:3,1]

X.iloc[:,:] # to get entire data frame 

# checking the type of variable 
type(X.X1) # pandas series object

# to create a data frame 
x = pd.DataFrame(columns=["A","B","C"])

# np.random.randint(a,b,c) 
# a - > starting number
# b - > Ending number
# c - > no. of numbers to be generated 
x["A"] = pd.Series(list(np.random.randint(1,100,50)))

# np.random.choice([a,b],size=c)
# a and b = > choosing elements from a or b 
# c = > number of elements to be generated choosing from a or b
x["B"] = pd.Series(list(np.random.choice([0,1],size = 50)))

x["C"] = 10 # going to fill all the rows in "C" with value 10



# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv("E:\\Excelr Data\\Python Codes\\Basic Statistics _ Visualizations\\mba.csv")
type(mba) # pandas data frame

mba.columns # accessing column names 
mba.datasrno # Accessing datasrno using "." (dot) operation

mba["workex"]
mba[["datasrno","workex"]] #  accessing multiple columns 
mba.iloc[45:51,1:3] # mba.iloc[i,j] 
# i => row index values  | j => column index values

mtcars = pd.read_csv("E:\\Excelr Data\\Python Codes\\Basic Statistics _ Visualizations\\mtcars.csv")

# to get the count of each category from a specific column 
mtcars.gear.value_counts()

# Accessing elements using conditional input 

# and operation (&) 
mtcars_4 = mtcars[(mtcars.gear==3) & (mtcars.mpg > 19.2)] #  and operation (&)

# or operation (or)
mtcars_5 = mtcars[(mtcars.gear==3) | (mtcars.mpg>19.2)]

# Gear 4 and 5 cars only 
mtcars_4_5 = mtcars[(mtcars.gear==5) | (mtcars.gear==4) | (mtcars.gear==6)]

# isin operator which functions similar to that of "or" operator 
mtcars_4_5 = mtcars[mtcars.mpg.isin(list(range(15,21,1)))]

mtcars_15_19 = mtcars[(mtcars.mpg>15) & (mtcars.mpg<19)]

# line 76 and 79 will return same output 
                    
# Creating custom function 

def MEAN(i): # taking only one parameter 
    a = sum(i)  
    b =len(i)
    print (a/b)

x = [1,2,3,4,5,6,8]
MEAN(x)  



# using if and else conditional statements 
def check_even(i):
    # function body
    if i%2==0:
        print ("even")
    else:
        print ("odd")

check_even(10)
check_even(101)

# using if, elif and else conditional statements 
def is_even_odd(i):
    if i<5:
        print ("<5")
    elif i<10:
        print ("<10")
    elif i<20:
        print ("<20")
    else:
        print ("None")
    


x = [1,2,[1,2,4],43,54] # list format 


# For loop syntax 
for i in x:
    print (i)
    
for i in range(10):
    print (i)

for i in range(1,10,1):
    print (i)
    
# range(a,b,c)

for i in range(1,10,3):
    print (i)

# while loop syntax 
i = 1
while i < 10:
    print (i)
    i= i+1
    

x = [[1,2],[1,2,4],[43],[54],[2,3,4]]

# using break operation to terminate any for loop in middle
for i in x:
    if len(i)>2:
        break
    else:
        print (i)
    
a = [] # creating empty list 
a.append(1) # appending new element to list variable "a"

a.append([1,2]) # appending new list to list variable "a"
a.extend([1,2,3]) # appending each element separately using extend 
a


# Giving input from the console at the time of execution of python code
# and appending each elements when we enter from console 

for i in range(1,10):
    a.append(int(input()))

# Creating a data frame manually from lists 
new_df = pd.DataFrame(columns=["A","B","C"])
new_df["A"] = pd.Series([1,2,3,4,5,6])    
new_df["B"] = pd.Series(["A","B","C","D"])   
new_df["C"] = pd.Series([True,False,True,False])

# Dropping rows and columns 

new_df.drop(["A"],axis=1) # Dropping columns 
# axis = 1 represnts drop the columns 
# new_df.drop(["A","B"],axis =1, inplace=True) # Dropping columns 
# inplace = True  = > action will be effective on original data 


# Dropping rows => axis =0
mba.drop(mba.index[[5,9,19]],axis=0) # Dropping rows 
# selecting specific rows using their index values in list format
#  X.index[[1,2,3,4,5]] => dropping 1,2,3,4,5 rows 

# X.drop(X.index[[5,9,19]],axis=0, inplace = True)

#X.drop(["X1","X2"],aixs=1) # Dropping columns
#X.drop(X.index[[0,2,3]],axis=0) # Dropping rows 

# Creating a data frame using dictionary object 
x = {"A":pd.Series([1,2,3,4,5,7,8,10]),"B":pd.Series(["A","B","C","D","E","F","G"]),"C":pd.Series([1,2,3,4,5,7,8])}
new_x = pd.DataFrame(x)

# Dictionary object
dict_new = {"A":[1,2,3,4,5,7,8],"B":["A","B","C","D","E","F","G"],"C":[1,2,3,4,5,7,8]}
dict_new.keys() 
dict_new.values()
dict_new["A"] # accessing values using the key
# In any dictionary object we have unique keys and keys must not be repeated
# values can be of any size and can be repeated 


# Finding mean,median,mode
mba['gmat'].mean() # mba.gmat.mean()
mba['gmat'].median()
mba['gmat'].mode()
mba['gmat'].var()
mba['gmat'].std()

# variance & Standard Deviation for Sample
mba['gmat'].var() # 860
mba['gmat'].std() # 29.39

# Variacne & Standard Deviation for Population
np.var(mba['gmat']) # 859.70
np.std(mba['gmat']) # 29.32


# calculating the range value 
range = max(mba['gmat'])-min(mba['gmat']) # max(mba.gmat)-min(mba.gmat)
range

# calculating the population standard deviation and variance 
np.var(mba.gmat) # population variance 
np.std(mba.gmat)  # population standard deviation

# Importing necessary libraries
import pandas as pd # importing pandas = > useful for creating dataframes
import numpy as np   # importing numpy = > useful for creating numpy arrays 
x1 = [1, 2, 3, 4, 5] # list format 
x2 = [10, 11, 12, 13]  # list format 
# Creating a data frame using explicits lists
X = pd.DataFrame(columns = ["X1","X2"]) 
X
X["X1"] = pd.Series(x1) # Converting list format into pandas series format
X["X2"] = pd.Series(x2) # Converting list format into pandas series format
X
X.X1
X["X1"]
# accessing columns using "." (dot) operation
X.X1
# accessing columns alternative way
X["X1"]

# Accessing multiple columns : giving column names as input in list format
X[["X1","X2"]]
# Accessing elements using ".iloc" : accessing each cell by row and column 
# index values
X.iloc[0:3,1]

X.iloc[:,:] # to get entire data frame 

# checking the type of variable 
type(X.X1) # pandas series object

X.iloc[0:3,1]
X.iloc[:,:]
type(X.X1)
X.iloc[0:3,1]
X.iloc[:,:] # to get entire data frame
# to create a data frame 
x = pd.DataFrame(columns=["A","B","C"])

# np.random.randint(a,b,c) 
# a - > starting number
# b - > Ending number
# c - > no. of numbers to be generated 
x["A"] = pd.Series(list(np.random.randint(1,100,50)))

# np.random.choice([a,b],size=c)
# a and b = > choosing elements from a or b 
# c = > number of elements to be generated choosing from a or b
x["B"] = pd.Series(list(np.random.choice([0,1],size = 50)))

x["C"] = 10 # going to fill all the rows in "C" with value 10



x = pd.DataFrame(columns=["A","B","C"])
x["A"] = pd.Series(list(np.random.randint(1,100,50)))

x["B"] = pd.Series(list(np.random.choice([0,1],size = 50)))
x
# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv("/home/abcd/Downloads/mba.csv")
type(mba) # pandas data frame
mba.columns # accessing column names 
mba.datasrno # Accessing datasrno using "." (dot) operation
mba["workex"]
mba[["datasrno","workex"]] #  accessing multiple columns 
mba.iloc[45:51,1:3] # mba.iloc[i,j] 
# i => row index values  | j => column index values

mtcars = pd.read_csv("/home/abcd/Downloads/mtcars.csv")
# to get the count of each category from a specific column 
mtcars.gear.value_counts()
# Accessing elements using conditional input 

# and operation (&) 
mtcars_4 = mtcars[(mtcars.gear==3) & (mtcars.mpg > 19.2)] #  and operation (&)

mtcars_4

# or operation (or)
mtcars_5 = mtcars[(mtcars.gear==3) | (mtcars.mpg>19.2)]
mtcars_5
# Gear 4 and 5 cars only 
mtcars_4_5 = mtcars[(mtcars.gear==5) | (mtcars.gear==4) | (mtcars.gear==6)]
mtcars_4_5
# Creating custom function 

def MEAN(i): # taking only one parameter 
    a = sum(i)  
    b =len(i)
    print (a/b)

x = [1,2,3,4,5,6,8]
MEAN(x)  
# using if and else conditional statements 
def check_even(i):
    # function body
    if i%2==0:
        print ("even")
    else:
        print ("odd")

check_even(10)
check_even(101)
# using if, elif and else conditional statements 
def is_even_odd(i):
    if i<5:
        print ("<5")
    elif i<10:
        print ("<10")
    elif i<20:
        print ("<20")
    else:
        print ("None")
    
is_even_odd(21)
x = [1,2,[1,2,4],43,54] # list format 


# For loop syntax 
for i in x:
    print (i)
    
for i in range(10):
    print (i)

for i in range(1,10,1):
    print (i)
    
for i in range(1,10,3):
    print (i)

i = 1
while i < 10:
    print (i)
    i= i+1
x = [[1,2],[1,2,4],[43],[54],[2,3,4]]
for i in x:
    if len(i)>2:
        break
    else:
        print (i)
a = [] # creating empty list 
a.append(1) # appending new element to list variable "a"

a.append([1,2]) # appending new list to list variable "a"
a.extend([1,2,3]) # appending each element separately using extend 
a
for i in range(1,10):
    a.append(int(input()))
new_df = pd.DataFrame(columns=["A","B","C"])
new_df["A"] = pd.Series([1,2,3,4,5,6])    
new_df["B"] = pd.Series(["A","B","C","D"])   
new_df["C"] = pd.Series([True,False,True,False])
new_df
new_df.drop(["A"],axis=1) # Dropping columns 
# axis = 1 represnts drop the columns 
# new_df.drop(["A","B"],axis =1, inplace=True) # Dropping columns 
# inplace = True  = > action will be effective on original data 


# Dropping rows => axis =0
mba.drop(mba.index[[5,9,19]],axis=0) # Dropping rows 
# selecting specific rows using their index values in list format
#  X.index[[1,2,3,4,5]] => dropping 1,2,3,4,5 rows 

# X.drop(X.index[[5,9,19]],axis=0, inplace = True)

#X.drop(["X1","X2"],aixs=1) # Dropping columns
#X.drop(X.index[[0,2,3]],axis=0) # Dropping rows 

# Creating a data frame using dictionary object 
x = {"A":pd.Series([1,2,3,4,5,7,8,10]),"B":pd.Series(["A","B","C","D","E","F","G"]),"C":pd.Series([1,2,3,4,5,7,8])}
new_x = pd.DataFrame(x)
new_x
dict_new = {"A":[1,2,3,4,5,7,8],"B":["A","B","C","D","E","F","G"],"C":[1,2,3,4,5,7,8]}
dict_new.keys() 
dict_new.values()
dict_new["A"] # accessing values using the key
# In any dictionary object we have unique keys and keys must not be repeated
# values can be of any size and can be repeated 

# Finding mean,median,mode
mba['gmat'].mean()
mba['gmat'].median()
mba['gmat'].mode()
mba['gmat'].var()
mba['gmat'].std()
range = max(mba['gmat'])-min(mba['gmat']) # max(mba.gmat)-min(mba.gmat)
range
np.var(mba.gmat) # population variance 
np.std(mba.gmat)
np.var(mba.gmat)















