def fact(n):

    result = 1

    for i in range(1,n+1):

        result *= i

    return result



def combin(n,k):

    result = fact(n)//(fact(n-k)*fact(k))

    return result



def gcd(a,b):

    if(a == 0):

        return b

    elif(b == 0):

        return a

    else:

        return gcd(b,a%b)

box = int(input("Input number of boxes : "))

item = int(input("Input number of items in each box : "))

# result = 0

result = 2*((item-1)**box)-(item-2)**box

# for i in range(1,box):

#     for j in range(1,box-i+1):

#         result += combin(box,i) * combin(box-i,j) * ((box-i-j)**(box-i-j))

all_cases = item ** box

result = all_cases-result

gcd_result = gcd(result,all_cases)

# print(result)

# print(all_cases)

print("Result = {}".format(result/all_cases))

15738871948381855/2567836929097728