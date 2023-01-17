#this code will generate a key pair using primes from 0,1000. In practice the numbers used have hundreds of digits.

import math

import random

def modin(a, m):

   u1, u2, u3 = 1, 0, a

   v1, v2, v3 = 0, 1, m

   

   while v3 != 0:

      q = u3 // v3 #divides floors the result

      v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3

   return u1 % m



lowprimes = lowPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 

   67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 

   157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 

   251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,317, 331, 337, 347, 349, 

   353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 

   457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 

   571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 

   673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 

   797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 

   911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

print(len(lowprimes))

p=lowprimes[random.randrange(0,168)]# randomly pick a prime number:p

q=lowprimes[random.randrange(0,168)]# randomly pick a prime number:q

n=p*q

a=p-1 # meant to be p-1

b=q-1 # meant to be q-1

pek=0 #encryption key or e or public key

prk=0 #decryption key or d or private key

e=2

d=1

f=a*b

fac=[]

can=[]

print('p is',p)

print('q is',q)

print(' p-1 is ',a)

print(' q-1 is ',b)

for i in range(0,168): #This will get the list of primes that are not the factors of (p-1)*(q-1). (also called f)

    if(f%lowprimes[i]!=0): #hence fac holds the public key candidates

        fac.append(lowprimes[i])

pek=fac[random.randrange(0,len(fac))]  #public key

prk=modin(pek,f)#function for modulo inverse

print("n",n)

print("public key; ",pek)

print("private key: ",prk)
#this code will demonstrate the encryption and decryption process with RSA. We will consider the key pair already generated

# eariler. We will encrypted the message with the private key and decrypted it with the public key.

m='Hello World'#the message to be encrypted

e=[]#this will store the encrypted message

f=[]

for i in range(0,len(m)):

    b=(pow(ord(m[i]),prk)%n)

    e.append(b)

print(e)# e will appera like random gibberish because it it encrypted text

for i in range(0,len(e)):

    b=chr((pow(e[i],pek)%n))

    f.append(b)

print(f)

    
#this function will show how the idea will work. First we will sign a real video.

import cv2

import numpy as np

import hashlib as hb

a=[]

file = "/kaggle/input/videos/dkuayagnmc.mp4" # the real video

BLOCK_SIZE = 65536 # The size of each read from the file



file_hash = hb.sha256() # Create the hash object, can use something other than `.sha256()` if you wish

with open(file, 'rb') as f: # Open the file to read it's bytes

    fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above

    while len(fb) > 0: # While there is still data being read from the file

        file_hash.update(fb) # Update the hash

        fb = f.read(BLOCK_SIZE) # Read the next block from the file

#print (file_hash.hexdigest())

c1=file_hash.hexdigest()

print("real video unsigned hash",c1)

e=[]

for i in range(0,len(c1)):

    d=(pow(ord(c1[i]),prk)%n)

    e.append(d)

print("real video signed hash",e)

#now e has the signed hash of the real video
file="/kaggle/input/videos/ahbweevwpv.mp4" #the fake video

hs=[]

with open(file, 'rb') as f: # Open the file to read it's bytes

    fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above

    while len(fb) > 0: # While there is still data being read from the file

        file_hash.update(fb) # Update the hash

        fb = f.read(BLOCK_SIZE) # Read the next block from the file

#print (file_hash.hexdigest())

c=file_hash.hexdigest()

for i in range(0,len(e)): #this loop will decrypt the hash of the real video

    d=chr((pow(e[i],pek)%n))

    hs.append(d)

print("fake video hash",c)

hs=str(hs)

print("real video decrypted hash",hs)
