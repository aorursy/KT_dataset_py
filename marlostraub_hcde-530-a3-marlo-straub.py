NAMES_LIST = "/kaggle/input/a3data/yob2010.txt"

boys = {}  # create an empty dictionary of key:value pairs for the boys
girls = {} # create an empty dictionary of key:value pairs for the girls

for line in open(NAMES_LIST, 'r').readlines():  # iterate through the lines and separate each line on its commas
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


#This is me trying to replicate the example using my own data, which copied the format of yob2010.txt
NAMES_LIST2 = "/kaggle/input/mhstest/namestest.txt"

boys2 = {}  # create an empty dictionary of key:value pairs for the boys
girls2 = {} # create an empty dictionary of key:value pairs for the girls
for line in open(NAMES_LIST2, 'r').readlines():
    name2, gender2, count2 = line.strip().split(",")
    count2 = int(count2)   # Cast the string 'count' to an integer
    if gender2 == "F":    # If it's a girl, save it to the girls dictionary, with its count
        girls2[name2.lower()] = count2 # store the current girls name we are working with in the dictionary, with its count
    elif gender == "M": # Otherwise store it in the boys dictionary
        boys2[name2.lower()] = count2
print(len(name2)) #just testing other commands
print("There were %d girls' names." %len(girls2))
print("There were %d boys' names." %len(boys2))
print(girls2) #just testing other commands
print(boys2) #just testing other commands
print(girls2.items()) #just testing other commands
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

#For 1A, will assume that I didn't see the answer in example 1, and will use two-way conditional statements
if len(boys) >= len(girls): #evaluate this T/F statement
    bgdiff=len(boys)-len(girls) #create the temporary variable bgdiff that is the number of keys in the dictionary 'boys'
                                #minus the number of keys in the dictionary 'girls'
    print("There are "+str(bgdiff)+" more boy names than girl names in 2010") #if the statement is true, print this text 
                                                                            #plus str(bgdiff) so that the number will be be printed
elif len(boys)==len(girls):#if the previous statement is false, evaluate this T/F statement
    print("There are the same number of boy and girl names in 2010") #print this text if true
else: #if neither of the above statements are true...
    gbdiff=len(girls)-len(boys) #...then create the temporary variable gbdiff that is the number of keys in the dictionary 'girls'
                                #minus the number of keys in the dictionary 'boys'
    print("1A: There are "+str(gbdiff)+" more girl names than boy names in 2010")#...and print this text 
                                                                            #plus str(bgdiff) so that the number will be be printed
        

#First, I determine whether or not capitalization of the first letter in each name matters, by creating lowercase copies of the 1A dictionaries 
lowerb=boys.copy() #creating a copy of the dictionary to see if capitalization of names matter
lowerg=girls.copy() #" "
lowerb={k.lower(): v for k, v in lowerb.items()} #update dictionary lowerb so that the keys are now all lowercase (I looked up how to do this up online)
if len(lowerb)==len(boys): #test to see if the original dictionary and lowercase only dictionary for boys are the same size
    print("All the names in yob2010 are capitalized, so subsequent code does not need to factor in differences in capitalization.")
else:
    print("Capitalization of the first letter in the names of yob2010 indeed matters; e.g. ""Alex"" is a separate name than ""alex"", which needs to be reflected in the code")

#Next, create new dictionaries for just names starting with Q
bvalues=boys.keys() #create a list of the keys in the dictionary 'boys'
gvalues=girls.keys() #create a list of the keys in the dictionary 'girls'
qboys={} #create empty dictionary for boy names starting with Q
qgirls={} #create empty dictionary for girl names starting with Q
for qbname in bvalues: #create temporary variable for boy name starting with q in list bvalues
    if qbname[0]=="q":#check to see if the string begins with the letter Q
        qboys[qbname]=1 #if true, put the name in the qboys dictionary with the value one since it is new
for qgname in gvalues: #create temporary variable for girl name starting with q in list gvalues
    if qgname[0]=="q":#check to see if the string begins with the letter Q
        qgirls[qgname]=1 #put the name in the qgirls dictionary with the vaue one since it is new
        
#Repeat code from 1A, updated to compare qboys vs qgirls dictionaries

if len(qboys) >= len(qgirls): #evaluate this T/F statement
    qbgdiff=len(qboys)-len(qgirls) #create the temporary variable bgdiff that is the number of keys in the dictionary 'qboys'
                                #minus the number of keys in the dictionary 'qgirls'
    print("There are "+str(qbgdiff)+" more boy names than girl names starting with Q in 2010.") #if the statement is true, print this text 
                                                                            #plus str(qbgdiff) so that the number will be be printed
elif len(qboys)==len(qgirls):#if the previous statement is false, evaluate this T/F statement
    print("There are the same number of boy and girl names starting with Q in 2010.") #print this text if true
else: #if neither of the above statements are true...
    qgbdiff=len(qgirls)-len(qboys) #...then create the temporary variable qgbdiff that is the number of keys in the dictionary 'qgirls'
                                #minus the number of keys in the dictionary 'qboys'
    print("1A: There are "+str(qgbdiff)+" more girl names than boy names starting with Q in 2010.")#...and print this text 
                                                                            #plus str(qbgdiff) so that the number will be be printed
alpha=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z] #create alphabetical list to reference; I get a "name 'a' is not defined" error and don't understand the search results related to this error
n=0 
dewdictb={}#create blank dictionary for output of function(s) on boys dictionary
newdictg={}#create blank dictionary for output of function(s) on girls dictionary
for allbname in bvalues: #create temporary variable for boy name starting with n character in list bvalues which is a list of all the keys in boys
    if allbname[0]==alpha[n]:#check to see if the index 0 string item begins with the value 'n' in the list alpha
        newdictb=newdictb+n+1 #need a way to create a new dictionary every time n moves on to the next item in the list alpha, but I don't know how to do this 
        allboys[newdictbn]=1 #if the conditional statement above true, put the name in the qboys dictionary I created would have liked to create programmatically above
                            #give it the value of one since it is new
        #need to do something here to keep creating new letter dictionaries
        #I need to do something here to loop through the alphabet
        #I gave up and ran out of time so moved on to other exercises
        
bnametotal=0 #establish a temporary variable "bnametotal" that is the sum of all counts of a given boy's name in the dictionary
bcountvals=boys.values()#create an iterable list out of the values in the dictionary "boys"
for x in bcountvals: #create the temporary variable "x" which will iterate through the values in the dictionary "boys"
    bnametotal= bnametotal+x #accumulate the total values 

gnametotal=0 #establish a temporary variable "gnametotal" that is the sum of all counts of a given girl's name in the dictionary
gcountvals=boys.values()#create an iterable list out of the values in the dictionary "girls"
for y in gcountvals: #create the temporary variable "y" which will iterate through the values in the dictionary "girls"
    gnametotal= gnametotal+y #accumulate the total values 

howmany2010babies=bnametotal+gnametotal #sum the counts from the dictionaries "boys" and"girls"
print("There are "+str(howmany2010babies)+" babies in the dataset") #print this text

#boys["keylen"]=len(boys.keys()) #First, I tried to create and add the new value "keylen" to the dictionary "boys" 
                                #and then set the value of a given "keylen" to the length of each keyin the dictionary "boys" 
                                #but I get the error 'keys() takes no arguments (1 given)''
#the output of print(boys) once running this code only shows the original key:value pairs, so this doens't appear to be working

#I then tried...
#for nm in boys: #create the temp variable 'nm' and iterate through dictionary 'boys'
    #keylen=len(nm) #create the temp variable 'keylen' and set its value to the length of the string 'nm'
    #boys[keylen]=int(keylen) #add 'keyln'to the dictionary 'boys', where the value of keylen is an integer
#but then print(boys) returned "RuntimeError: dictionary changed size during iteration"because I was changing the dictionary size while looping through it

for nm in boys:#create the temp variable 'nm' and iterate through dictionary 'boys'
    keylen=len(nm)#create the temp variable 'keylen' and set its value to the length of the string 'nm'
    #but I get the error "TypeError: object of type 'int' has no len()", which is odd since I didn't get this error when running the code above previously
    boys2={} #create a new empty dictionary 'boys2'
    boys2[keylen]=keylen #here I am trying to add the variable keylen and its associated values to the dictionary 'boys2', and then I would try and add the name keys, but...
    #I have been trying to work on this exercise for a total of 3 hours  but in 30 minute intervals spaced out over 3 days, since those are the only pockets of time I have
    #it has not worked, and I am more confused than ever, so I am giving up :(. It feels so close to my grasp, arghghg!!!

#I also tried this, but I can't remember my trai of thought to update the comments

#for boyname in boys:
    #len(boyname)
    #boys.update(len(boyname)) #here, I alwys got a ""'int' object is not iterable" error, but I didn't understand why this isn't iterable




#I also tried these thigns below, but got interrupted and can't remember where I left off, so am just going to submit this as is

bnameslist=boys.keys() #create a list of the keys in the dictionary "boys"
gnameslist=girls.keys() #create a list of the keys in the dictionary "boys"
bnamecharctlst=[] #create empty list for character count of boys' names
gnamecharctlst=[] #create empty list for character count of girls' names
for namecharct in bnameslist: #create temp variable "namecharct"; iterate through bnameslist
    len(namecharct) #get the length of each name in the list
    

def longestname(names_list): #define the function that operates on the variable "names_list"
    namelen=[] #create a placeholder empty list for the length of a given item in "names_list"
    for n in names_list: #iterate through the given list using temporary variable n
        namelen.append((len)) #get the character count for the list item n, then append that character count to the namelen list 

bngn=[] #create an empty list for boy names that are also girl names
for name in boys.keys(): #iterate through the "name" keys in the dictionary "boys" (Is this what I call this "name" variable established previously?)
    if name in girls.keys(): #1-way conditional statement
       bngn.append(name) #if the above statement is true, append the name variable to the list "bngn"
print("There are "+str(len(bngn))+" boys names that are also girls names in 2010.") #print the length of the list 

#alternatively, I could do the following and it should produce the same answer

gnbn=[] #create an empty list for girl names that are also boy names
for name in girls.keys(): #iterate through the "name" keys in the dictionary "girls" (Is this what I call this "name" variable established previously?)
    if name in boys.keys(): #1-way conditional statement
       gnbn.append(name) #if the above statement is true, append the name variable to the list "gnbn"
print("There are "+str(len(bngn))+" girls names that are also boys names in 2010.") #print the length of the list bngn

gnbndict={ } #create an empty dictionary girl names that are also boy names.
            #we want a dictionary instead of the list since we want to keep the count associated with the name keys
for x in girls.keys(): #iterate through the "name" keys in the dictionary "girls" (Is this what I call this "name" variable established previously?)
    if x in boys.keys(): #1-way conditional statement; see if the name is also in the keys of the dictionary "boys"
       gnbndict.update({x: girls[x]}) #if the above statement is true, update the dictionary "gnbndict" with "x" item as the key and the value called by that key in dictionary "girls" 
        #some of the names also were surprising in this list, but I checked to see if "isabella"
listofsortedgnbn=sorted(gnbndict.items(), key=lambda x:x[1], reverse=True) #I'm trying to use the sorted function to sort the values of the dictionary "gnbndict" from largest to smallest using reverse=True
print(listofsortedgnbn.keys[0])#print the 0 index item in the list of sorted names "listofsortedgnbn". This didn't work, but I ran out of time to troubleshoot]