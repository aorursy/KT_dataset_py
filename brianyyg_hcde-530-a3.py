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

# Make a comparison using number of names 
# if len(boys) > len(girls):
#    print ("There are more boy names")
# else:
#    print ("There are more girl names")

firstl = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

# Create a containers for names with the same first letter
bcaplist = []
gcaplist = []

blist = boys.keys() # Put all boy names in a list
glist = girls.keys() # Put all girl names in a list

# Loop through a-z
for capital in firstl:
    
    for bname in blist: # Loop through boy names
        if bname[0] == capital: # Check if the first letter of each boy name maches specific letter
            bcaplist.append("bname") # If matches, put in the list
    bcapcount = len(bcaplist) # Count the number of boy names with the same first letter
    
    for gname in glist: # Loop through girl names
        if gname[0] == capital: # Check if the first letter of each girl name maches specific letter
            gcaplist.append("gname") # If matches, put in the list
    gcapcount = len(gcaplist) # Count the number of girl names with the same first letter
    
    uppercap = capital.upper() # Capitalize first letter to better display it in the following sentense
    
    # Make comparison
    if bcapcount > gcapcount:
        print ("There are more boy names than girl names start with letter " + f"{uppercap}") # print the sentense 
    else:
        print ("There are more girl names than boy names start with letter " + f"{uppercap}") # print the sentense

    bcaplist.clear() # Empty the boy names container before looping through next letter
    gcaplist.clear() # Empty the girl names container before looping through next letter
        
        
# If want to print certain count use:  print("There are " + f"{bcapcount}" + " boy names start with letter " + f"{uppercap}") # print the sentense   
#    if str(boys.keys())[0] == capital and str(girls.keys())[0] == capital:
#       if str(boys.keys())[0] > str(girls.keys())[0]:
#          print("a")    
#    if capital == 
#        print(name + " " + str(boys[name]))    
#    if boys[0] = capital and 
#    len(boys) > len(girls): 
#    print ("There are more boy names")




# First put all the names in a list
fulllist = []

# Add all boy names to the list
for bbname in blist:
    fulllist.append(bbname)
    
# Add all girl names to the list
for ggname in glist:
    fulllist.append(ggname)
    
# to ensure the total is correct: print (f"{len(fulllist)}")

# Create a container 
longname = []

# Set a start point
namelen = 0

# Loop through all names
for thisname in fulllist:
    
    # If find a longer name
    if len(thisname) > namelen:
        longname = thisname
        namelen = len(thisname)
    
    # If equally long, add it to the list
    elif len(thisname) == namelen:
        longname = longname + " and " + thisname
    
print (longname)
    
# Create a container for subset names
# subnlist = []

# Remove duplicates in full name list
# fulllist = list(dict.fromkeys(fulllist))

# Loop through all names
# for sname in fulllist:
        
# Sort girl names in an descending order
sortgname = sorted(girls.items(), key=lambda x: x[1], reverse=True)

# Create a container for ranked girl names
rankgname = []

# Pull only girl names from the ranked girl names-number touple
# and add to rankgname
for namepair in sortglist:
    rankgname.append(namepair[0])

# turn blist from dict_keys to a list
realblist = list(blist) 

# Check what is the highest ranked girl names that also shows up in boy name list
for targetname in rankgname:
    if targetname in realblist:
        print (targetname) # Print the name out
        break # Stop the thing
