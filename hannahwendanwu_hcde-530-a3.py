NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"



boys = {}  # create an empty dictionary of key:value pairs for the boys

girls = {} # create an empty dictionary of key:value pairs for the girls



for line in open(NAMES_LIST, 'r').readlines(): 

    # iterate through the lines and separate each line on its commas

    # since we know there are three items on each line, we can assign each of them to a variable

    # the line below has three variables - name, gender, count (one variable for each of the three items in a line)

    name, gender, count = line.strip().split(",")



    # Since 'count' is actaully a string of text and not an integer, 

    # we need to turn it into an integer to store that number in the dictionary so we can use it. 

    # later to do arithmetic that we couldn't do, if it was just text. This is called 'casting'.

    

    count = int(count)   # Cast the string 'count' to an integer

    

    if gender == "F":    # If it's a girl, save it to the girls dictionary, with its count

        girls[name.lower()] = count # store the current girls name we are working with in the dictionary, with its count

    elif gender == "M": # Otherwise store it in the boys dictionary

        boys[name.lower()] = count



# We need to format our text so that it can show both the text and an integer.

# But the print() function only takes strings and we have a string and an integer.

# To deal with this, we use the % sign to say what text goes where. In the example 

# below the %d indicates to put a decimal value in the middle of the sentence, and

# that decimal value - the length of 'girls' is indicated at the end of the function: len(girls)



print("There were %d girls' names." %len(girls))

print("There were %d boys' names." %len(boys))



# We did this for you at the end of the previous homework.

# It's a little weird to stuff numbers into sentences this way, but once you get 

# used to it, it's easy. You can do lots of other formatting like this.

# Here's an explanation of how it works: https://www.geeksforgeeks.org/python-output-formatting/
# iterate through the boys dictionary. for each key see if it is: 'john'

for name in boys.keys():

    if name == "john":

        # if it is 'john', get the value associated with john and use that value for the print statement

        # because the value is an integer, we have to cast it to a string in the print statement, with str().

        print("There were " + str(boys[name]) + " boys named " + name)

for name in boys.keys():

    if name in girls.keys():

        print(name)
for name in boys.keys():

    if 'king' in name:

        print(name + " " + str(boys[name]))



for name in girls.keys():

    if 'queen' in name:

        print(name + " " + str(girls[name]))







#see if there are more girl names than boy names



if len(girls) > len(boys):

    print("There are MORE girl names(%d) than boy names(%d)" %(len(girls),len(boys)))



elif len(girls) < len(boys):

    print("There are LESS girl names(%d) than boy names(%d)" %(len(girls),len(boys)))

else:

    print("There are EQUAL girl names(%d) than boy names(%d)" %(len(girls),len(boys)))





# to generate a list contains letter "a" to "z"

letterAZ = []

for i in range(97,97+26):

    letterAZ.append(chr(i))

    



letterCountF = 0

letterCountM = 0



print ("\n")



# iterately check how many names in a specific letter and compare between boys and girls



for a in letterAZ:

    for name in girls.keys():

        if name[0] == a:

            letterCountF += 1

    for name in boys.keys():

        if name[0] == a:

            letterCountM += 1

    if letterCountF > letterCountM:

        print("There are MORE girl names(%d) than boy names(%d) with first letter %s" %(letterCountF,letterCountM,a))

    elif letterCountF < letterCountM:

        print("There are LESS girl names(%d) than boy names(%d) with first letter %s" %(letterCountF,letterCountM,a))

    else:

        print("There are EQUAL girl names(%d) than boy names(%d) with first letter %s" %(letterCountF,letterCountM,a))

    letterCountF = 0

    letterCountM = 0

    

    

  

            
babyNum = []



#I know it is unnecessary to re-open the file, I just want to make the code clearer for myself



for line in open(NAMES_LIST, 'r').readlines(): 

    name, gender, count = line.strip().split(",")

    #put all the count value into list babyNum

    babyNum.append(int(count))



# sum the list

babynumSum = sum(babyNum)

print("There are %d babies in the dataset" %babynumSum)

nameList = []



for line in open(NAMES_LIST, 'r').readlines(): 

    name, gender, count = line.strip().split(",")

    

    #put all the name to nameList

    nameList.append(name)

    

max_so_far = 0



#find the length of the longest name



for x in nameList:

    if len(x) > max_so_far:

        max_so_far = len(x)

        

print ("the longest name has %d characters" %max_so_far)



#since it is possible that there are more than one name has 15 characters, so I doubt check use this code



for i in nameList:

    if len(i) == 15:

        print(i)



    
#These two questions are asking the same thing --- how many names are overlapped between two dictionaries. But I still run twice anyway



neutralBoy = []

neutralGirl = []



for name in boys.keys():

    if name in girls.keys():

        neutralBoy.append(name)

print("There are %d boy names are also girl names" %len(neutralBoy))



for name in girls.keys():

    if name in boys.keys():

        neutralGirl.append(name)

print("There are %d girl names are also boy names" %len(neutralGirl))


# Just want to check how long it takes to run this part

import time 



t1 = time.time()



subset = 0



nameLen = []



#calculate the length of each name and put the result in numNames

for n in nameList:

    nameLen.append(len(n))

numNames = len(nameList)



# the logic here is that, if len(A) >len(B), then A cannot be B's subset. I was trying to eliminate the run time

for i in range(numNames):

    for j in range(numNames):

       if nameLen[i] < nameLen[j] and nameList[i] in nameList[j]:

            subset += 1 

            break



t2 = time.time()

print(subset)

print(t2-t1)

# I have the list of overlapping names between girls and boys dictionaries in question 4

# The list name is neutralGirl



max_count = 0

current_count = 0



# in the list of "neutral" names, find the one with maximum counts

for i in neutralGirl:

    current_count = int(girls[i])

    

    # if a name has the max counts, put the name to var popular_name and print it out

    if current_count > max_count:

       max_count = current_count

       popular_name = i

    

print(popular_name, max_count)



    


# revise all the input cases into lower cases

nameInput = input("Enter a baby name: ").lower()



valueF = 0

valueM = 0



if nameInput:

    # have to match completely with the dict keys

    if nameInput in girls.keys():

        # use the input as the key to find designated value, and make the value into integer 

        valueF = int(girls[nameInput])

    if nameInput in boys.keys():

        valueM = int(boys[nameInput])



    print("There are %d girls and %d boys with name %s" %(valueF,valueM,nameInput))



else:

    print("Please enter a name")

        
# revise all the input cases into lower cases



prefix = input("Enter prefix: ").lower()

num_prefixF = []

num_prefixM = []

total_num_prefixF = 0

total_num_prefixM = 0



if prefix:

    for name in girls.keys():

        if prefix in name:

            num_prefixF.append(int(girls[name]))      

    total_num_prefixF = sum(num_prefixF)



    for name in boys.keys():

        if prefix in name:

            num_prefixM.append(int(boys[name]))

    total_num_prefixM = sum(num_prefixM)

    print("There are %d girls and %d boys with the prefix %s" %(total_num_prefixF,total_num_prefixM,prefix))



else:

    print ("Please enter prefix")









        