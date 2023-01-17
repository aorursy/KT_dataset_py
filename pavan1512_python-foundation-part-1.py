x = 5

if (x == 0):

    print("zero")

elif (x > 0):

    print("positive")

else:

    print("negative")
# if(condtions, true, flase)



"positive" if (x > 0) else "negative"
x = 5

"zero" if (x == 0) else "positive" if (x > 0) else "negative"
x = 1

while(x <= 10):

    print(x)

    x += 1
# for loop uses range function for iteration**



for x in range(1, 11):

    print (x)