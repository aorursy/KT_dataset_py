!pip install jovian --upgrade --quiet
import numpy as np
import math as m

m=int(input('Enter the value of m:-'))
n=int(input('Enter the value of n:-'))
alphabets=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
x=m*n
first_array=[]
final_array=[]
for i in range(x):
        y=input('Enter the {}-th character:-'.format(i))
        first_array.append(y.upper())
        for j in alphabets:
            if y.upper()==j:
                final_array.append(alphabets.index(j))
                break
            else:
                continue
 
arr1=np.array(first_array)
arr2=np.array(final_array)
arr3=np.reshape(arr2,(m,n))
arr4=np.transpose(arr3)
arr4
arr5=np.reshape(arr1,(m,n))
plain_text=np.transpose(arr5)
plain_text


key_matrix_1=np.array([[8,16,5],
                     [9,5,7],
                     [2,3,21]])

det=np.linalg.det(key_matrix_1)
y=float(det%26)
y


def modInverse(a, m) : 
    a = a % m; 
    for x in range(1, m) : 
        if ((a * x) % m == 1) : 
            return x 
    return 1
modInverse(det,26)
arr4
x=m*n
key_array=[]
for i in range(x):
        y=int(input('Enter the {}-th key element:-'.format(i)))
        key_array.append(y)

mat1=np.array(key_array)
pre_key_matrix=np.reshape(mat1,(m,n))

key_matrix=np.transpose(pre_key_matrix)


key_matrix
multiplied_matrix=np.matmul (arr4,key_matrix)
multiplied_matrix
np.mod(multiplied_matrix,26)
key_matrix_1=np.array([[8,16,5],
                     [9,5,7],
                     [2,3,21]])
C=(key_matrix_1 @ arr4)%26
C
transpose=np.transpose(C)
cipher_array=np.reshape(transpose,(1,m*n))
cipher_list=cipher_array.tolist()
cipher_list1=cipher_list[0]
cipher_list1

for i in cipher_list1:
    print(alphabets[i],end='')
inverse_key_matrix=np.array([[10,11,15],
                            [3,12,25],
                            [11,22,0]])

P=(inverse_key_matrix @ C)%26
P
transpose=np.transpose(P)
plain_array=np.reshape(transpose,(1,m*n))
plain_list=plain_array.tolist()
plain_list1=plain_list[0]
plain_list1

for i in plain_list1:
    print(alphabets[i],end='')
