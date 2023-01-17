threedigitproduct = 0

palindrome= 0

num=[]

for n1 in range(999,99,-1) :

  for n2 in range(n1,99,-1) :

    threedigitproduct = n1 * n2        

    num = str(threedigitproduct)

    if len(num) == 5:

      num = '0'+num

    if num[0] == num[5] and num[1] == num[4] and num[2]== num[3]:

      if threedigitproduct >  palindrome :

         palindrome = threedigitproduct

    num=[]

print( palindrome)