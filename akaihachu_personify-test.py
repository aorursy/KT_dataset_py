rowCount=6

character=['H','I','J','K','L','M']

for i in range(rowCount):

    temp=''

    for j in range(rowCount):

        if (i+j>=rowCount-1):

            temp=temp+character[i]

        else: temp=temp+' '    

    print(temp)

        