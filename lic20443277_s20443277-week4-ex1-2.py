#Exercise 1

print("Enter a number for convering from miles into kilometers and meters: ")
miles=float(input())

kms=miles / 0.62137
ms=kms * 1000

print(miles,"miles is equitvalent to",'%.4f'%kms,"km /",'%.4f'%ms,"meters")
#Exercise 2

print("Please enter your name: ")
name=input()
print("Please enter your age: ")
age=input()
    
age_in_2047=int(age)-2020+2047
print("Hi ",name,"! in 2047 you will be ",age_in_2047,".",sep="")
