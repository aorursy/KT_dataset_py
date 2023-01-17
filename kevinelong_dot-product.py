# ZIP
#convert two arrays

X = ["A","B","C"]

Y = [
    "APPLE",
    "BANANA",
    "CHERRY"
]

Z = zip(X,Y)
print(Z)
print(dict(Z))

X = [6,5,4]
Y = [3,2,1]
dotproduct=0
for i,j in zip(X,Y):
    dotproduct += i*j
print('Dot product is : ' , dotproduct)
np.dot()