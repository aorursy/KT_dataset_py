import numpy as np
img=np.zeros((2,2),dtype=float) #initialize the image to be zeros

no_cols=img.shape[1]

no_rows=img.shape[0]

print(img)
#saving the attenuation values

horizontal_projection=np.array([0.6,0.3]) #top to bottom

vertical_projection=np.array([0.4,.5])    #left to right

diagonal_projection=np.array([0.3])      #top to bottom

reverse_projection=np.array([0.6])        #top to bottom

print(horizontal_projection,vertical_projection,diagonal_projection,reverse_projection)
#horizontal phase 

#a vectorized effecient code that doesn't use loops

horizontal=np.sum(img,axis=1) #sums over rows

error=(horizontal_projection-horizontal)/no_cols

img+=error[:,np.newaxis] #broadcasting the addition to the rows

print("after horizontal phase:\n",img)
#vertical phase

##a vectorized effecient code that doesn't use loops

vertical=np.sum(img,axis=0) #sums over columns

error=(vertical_projection-vertical)/no_rows

img+=error[np.newaxis,:] #broadcasting the addition to the columns

print("after vertical phase:\n",img)
#main diagonals phase

#to be vectorized

a=0

for a in range(img.shape[0]-1):

    main=img[a,a]+img[a+1,a+1]

    error=diagonal_projection[a]-main

    img[a,a]=img[a,a]+(error/2)

    img[a+1,a+1]=img[a+1,a+1]+(error/2)

print("after main diagonal phase:\n",img)
#reverse diagonals phase

#to be vectorized

b=0

for b in range(img.shape[0]-1):

    reverse=img[b,b+1]+img[b+1,b]

    error=reverse_projection[b]-reverse

    img[b+1,b]=img[b+1,b]+(error/2)

    img[b,b+1]=img[b,b+1]+(error/2)

print("after reverse diagonal phase:\n",img)