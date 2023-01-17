import numpy as np    # Using the alias np
ls = [ 1, 2, 3, 4, 5]

a = np.array( ls, dtype = np.int )   # np.int => all the values are of integer type

a
b = np.array( [0.5, 0.6, 0.7, 0.8, 0.9], dtype = np.float ) # np.float => all values are of float type

b
######## Attributes  !



print('Size Gives total number of elements in the array :: ',  a.size )

print('Data type Gives the data type of elements in the array :: ', a.dtype)

print('Itemsize Gives the memory of each element in bytes :: ', a.itemsize )  # Each item takes 8 bytes in memory  , so 5*8 = 40 bytes

print('Ndim Gives the Dimensions :: ', a.ndim)

print('Shape Gives the Shape of the Array ( Here since dimensions is 1, it is (number of elements,) ):: ', a.shape)
###### Some operations on arrays

print()

print('Before adding 5 :: ', a )

print( 'After adding 5 ::', a +5 )   ##### You can notice that 5 is added to each element in the array !!



print()

print( 'Array a :: ', a )

print( 'Array b :: ', b )

print('Adding 2 numpy arrays :: ', a + b)   ## Again we see  elementwise addition 



#### Another way to add them is using np.add()



print()

print( 'Array a :: ', a )

print( 'Array b :: ', b )

print('Another way to add arrays :: ',np.add(a, b) )

print('Subtraction of 2 arrays ::', np.subtract(a, b) )

print('Multiplication of 2 arrays :: ', np.multiply( a, b ))   # We can see that it is elementwise multiplication

print('Division of 2 arrays :: ', np.divide( a, b ))   # Elementwise Division



c = np.array( [ 1, 2, 3, 4])

d = np.array([1, 2, 3, 4])



print('Power operation a^b :: ', np.power(c, d ))  # 1*1, 2*2, 3*3, 4*4



# You can explore more of these functions in the documentation at 

# https://numpy.org/
a =  [ [1, 2, 3],

       [4, 5, 6 ] ]

b = np.array(a ,dtype = np.int)



print( 'Creating a 2dimensional array :: \n', b    )

print('Check type of a :: ' , type(a))

print('Check type of b :: ' , type(b))



print()

print('Totally we have 6 elements here in b :: ', b)

print( 'Shape of b :: ', b.shape)  # We have 2 rows and 3 columns



# Array b is of datatype int , lets convert it into float

# For this we use 



print()

d = b.astype( np.float )

print( 'Array d :: ', d )

print( 'Datatype of d :: ', d.dtype )  # Now we can see that its not integer type anymore , its float type 
#### Getting the List out of numpy array ( ndarray to list )

a = d.tolist()

print( 'd, is of type Ndarray', type( d ))

print( 'a, is of type list ', type( a ))
a = [ [ 1, 2, 3, 100], 

      [4, 5, 6, 101], 

      [7, 8, 9, 102], 

      [10, 11, 12, 103]]



a = np.array( a )

print( a )
### An example below shows how to get particular rows from the ndarray 



print( 'Getting 1st row :: ', a [ 0 ] )

print( 'Getting 2st row :: ', a [ 1 ] )



print()

# Zero indicates the start postion, 2 indicates the end position ( Here 2 is not included )( i.e 2-1 = 1 ) ( So row 0 and 1 get printed)

print( 'Getting 1st and 2nd Rows ::\n', a[ 0 : 2 ] )   



# Here 0 is the Beginning Row, 4-1 = 3 The ending Row, and each time we jump 2 times so , we get Rows 0 , 2, therefore we skip Rows 2 and 4 

print( 'Getting even Rows ::\n', a [ 0: 4 : 2 ]) 





print()

print( 'Similarly we can Select Columns too !! :: ')

print( 'Selecting 1st Column and all Rows :: ', a[ : , 0])  # Note. ':,' Includes all rows and 0 includes 1st column

print( 'Selecting first 3  Columns and all Rows :: ', a[ : , 0 : 3 ])  # Note. ':,' Includes all rows and First 3 Columns





print()

print('More Slicing :: ')

print( 'Lets Say That we want elements 9, 102, 12, 103 from previous array created ::')

## We do like this

print( a [ 2: , 2:])    # '2:' => Grab all rows from index 2 till end and '2:' indicates grab all columns from index 2 till end
### Grabbing elements based on a condition 



print('Array a ::\n', a )

print()

print( 'A simple condition i.e a < 5 checks each element in array if its less than 5, if so, The position is indicated by True else False\nBut What can we do with this ?? ')

print()

print( 'After condition :: \n', a < 5 )

print()



# We Can see that from our Original Array a , first 3 elements of 1st row and 1st element of 2nd row are the only elements lesser than 5 !!

# This can be clearly seen in the boolean array , where values are True ( if condition satisfied) else False 

print('We can Grab those elements based on the condition !! \n', a [ a < 5 ])    
# Lets say that you have a 2 x 3 matrix ( 2 rows and 3 columns)

# Now you can reshape this matrix into a 3 x 2 by doing the below



a = np.array( [ [1, 2],[5, 6], [11, 10] ] )

print('Array a, before Reshaping ::\n', a )

print('Shape of Array a, before Reshaping ::\n', a.shape )



print()



print('After Reshaping !! ( We converted a 3 x 2 matrix into a 2 x 3)')

b = a.reshape( (2,3))

print( b )
print( ' If you have missing values in your list you can see them as np.nan  (NaN stands for Not a Number) ::\n' )

a = np.array( [[1, 2, np.nan], [5, 6, 7]])

print( a )



print()

print('Fetching those NaN values ::\n', a [ np.isnan(a) ] )
a = np.array([1,1, 2, 3, 4, 5, 6])

print( 'Mean of Array a is :: ',  a.mean())

print( 'Sum of all Array elements :: ', sum( a ))

print( 'Standard deviation of Array a ::', a.std())

print( 'Cumulative Sum ::', a.cumsum())
### Flattening array

arr = np.array( [ [1, 5, 9], [2, 4, 8] ])



print( 'We can see that Array arr is of shape ( 2, 3) and dimension 2  ::\n', arr.shape, a.ndim)

print( 'If we flatten it , its shape becomes (6,) and ndim becomes 1\n', arr.flatten(), arr.flatten().shape, arr.flatten().ndim )
# Creates Random integers between the range 1 and 30 and gives use 10 random numbers 

rand_arr = np.random.randint( 1, 30, size = 10)

print( rand_arr )



print()

print('We can reshape Random arrays as shown below ::\n')

print( rand_arr.reshape( 2,5 ) )



print()

print('Generating numbers using arange ( similar to python\'s Range Function):: \n')

print( np.arange( start = 1, stop = 10, step = 2 ,))  # Start and Stop => Start at 1 ,and Stop at 10 - 1 = 9  and Take steps of 2, so our sequence becomes 1, 3, 5, 7 ...





print()

print('Generating 50 values sampled from Normal Distribution ::\n', np.random.randn( 50))
print('np.random.randn() Gives different 50 values each time its run, There we use np.random.speed() to return the same set of values ')

np.random.seed( 0 )  # Need not be 0 can be any number

print()

print('Please note that same set of values are returned no matter how many times the cell is executed !!:: \n')

print('Generating 50 values sampled from Normal Distribution ::\n', np.random.randn( 50))



print()

print('Generating Linearly spaced values using linspace ::\n')

print( np.linspace( start = 1, stop = 10, num = 5))
### Sorting values 



q = np.array( [ 9, 8, 45, 1, 0, 65, 1105])

print('Before sorting :\n', q)

print('After sorting :\n', np.sort(q) )



print()



### Argsort  Lets say you want indices of sorted elements



print( 'Using Argsort to get only the indices ::')

# We can see that here  9 appears in 4th position after sorting, similarly 8 occurs in 3rd position and so on

print( np.argsort(q ))  



print('To get the index of the maximum value we use argmax ::\n')

print( 'Index of max value is :: ' , np.argmax( q ) , 'And the value at that index is :: ', q [ np.argmax( q ) ] )
### Generating ZERO , One value arrays



print('Generating array full of ones ::\n' , np.ones( shape = (3, 3) ) )

print()

print('Generating array full of zeros ::\n' , np.zeros( shape = (3, 2) ) )

print()

print('Using the "full" method ::\n', np.full( fill_value = 9999, shape = (5, 1)))  # We filled an Array of shape (5, 1) with value 9999
### Using the Tile method to replicate 



a = np.array( [ 1, 2, 3, 4])

print( 'Repeats reps number of times ( Considering elementwise reps) ::\n', np.tile( a , reps = 3) )



b = np.random.randint( 10, 50, size = 15).reshape((3, 5))

print('Original Array ::\n ', b)

print( 'Repeats reps number of times ( Considering elementwise reps) ::\n', np.tile( b , reps = 3) )

### Getting the unique and number of unique elements



a = np.array([ 1, 5, 6, 9, 7,10, 1, 1, 1, 3, 5])



print('Unique elements from Array a :: \n', np.unique( a ))

uniq, cnt =  np.unique( a, return_counts = True )

print('Number of unique elements from Array a :: \n', cnt )  # 1 occurs 4 times, 3 occurs 1 time , 5 occurs 2 times  and so on
### Where method::



print('Original Array a ::\n', a )



print()

print('Now lets replace all those values in Array a where value is <=5 with 9999 if value > 5 replace them with -1')

b = np.where( a <= 5 ,9999, -1 )

print( b )
#### Splitting arrays 



b = np.array( [[1, 2, 3], [5, 9, 100], [565, 9898, 167]])



print('Now  we split this array first based on row ( axis = 0) and second based on column ( axis = 1) ' )

print('Based on Row::\n')

print( np.split(b , [ 2, 3], axis = 0 ))  # We can see that we got 3 Array objects  !!!



print()

print('Based on Column::\n')

print(np.split(b , [ 2, 3], axis = 1 ))  # We can see that we got 2 Array objects  !!!
#### Concatenate arrays ::



a = np.ones( shape = (3, 3))

print('Array a :: \n', a)



b = np.zeros( shape = (3, 3))

print('Array b :: \n', b)



print()

print('Concatenating across AXIS ZERO AS shown in the IMAGE above ::\n')

print('After concatenating :: \n', np.concatenate( (a, b), axis = 0 ) )

print()



print('Concatenating across AXIS ONE AS shown in the IMAGE above ::\n')

print('After concatenating :: \n', np.concatenate( (a, b), axis = 1 ) )
#### Similarly we can use hstack and vstack functions as well !!



# Note, hstack and vstack can be done using np.stack with axis parameter :)



print('hstack results ::\n' , np.hstack( (a, b )))  # Stack Horizontally !!  ----->>> Direction



print()

print('vstack results ::\n' , np.vstack( (a, b )))  # Stack Vertically  !!  



# v

# v

# v

# v   Direction 
## Reading Files :: !!



data = np.genfromtxt('../input/population-hares-lynxes-carrots/populations.txt')

data
print('What is the shape of the data ?? ', data.shape )  # 21 rows and 4 columns

print( 'What is the first row ?? ', data[0])

print('What is the 3rd column??\n', data[ :, 2])

## Lets save the first 5 rows into a new text file !!



np.savetxt('populations_first5_rows.txt', data[:5])



## You can See on the top right under Kaggle Working directory our File appeared :) 
### We can ALSO save the first five rows into a format called npy



np.save( 'populations_5rows.npy', data[ : 5]) # Binary data



## Lets reload and print the data :)



print('USING npy format !! :: ')

dat_5rows = np.load('./populations_5rows.npy')

print( dat_5rows)



print()

print('USING npz format !! :: ')

np.savez( 'populations_5rows', data[ : 5]) # Binary data

dat_5rowsz = np.load('./populations_5rows.npz')

print( dat_5rowsz)



print()

print(' We can also add more than 1 data file in npz ::')



dat1 = data[ :5]

dat2 = data[ 5: 10 ]

dat3 = data[ 10: 15]



np.savez('populations_splitted', dat1, dat2, dat3)

dat_5rowsz = np.load('./populations_splitted.npz')

print( 'Arrays inside :: \n',dat_5rowsz.files )   # We can see arr_0 , arr_1 and arr_2

print()

print('Printing all Array contents ::')

print('****'*10)

for array in dat_5rowsz.files :

    print( dat_5rowsz[ array]  )

    print('****'*10)