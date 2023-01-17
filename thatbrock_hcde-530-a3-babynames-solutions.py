# See my version of the ssa library here: https://www.kaggle.com/thatbrock/ssadata-py?scriptVersionId=27898265

# To make your own, you need to copy the library add it as a utlity script in your own Kaggle account.

# See instructions for this in my copy of the library.



# Load data from the source files and create an alias to the data

import ssadata  
# the ssadata alias can now be used to access the data

num_boys_names = len(ssadata.boys)

num_girls_names = len(ssadata.girls)



print("total boy names: " + str(num_boys_names))

print("total girl names: " + str(num_girls_names))



for letter in "abcdefghijklmnopqrstuvwxyz":

    count_boys = 0

    count_girls = 0

    for name in ssadata.boys:

        first_letter = name[0]

        if first_letter == letter:

            count_boys = count_boys + 1

    

    for name in ssadata.girls:

        first_letter = name[0]

        if first_letter == letter:

            count_girls = count_girls + 1

    

    if count_boys == count_girls:

        print(letter + ": the same number of boys names and girls names")

    elif count_boys > count_girls:

        print(letter + ": more boys names than girls names")

    else:

        print(letter + ": more girls names than boys names")
maximum_length = 0

longest_names = []



for boys_name in ssadata.boys:

    if len(boys_name ) > maximum_length:

        maximum_length = len(boys_name)

        longest_names = []

        longest_names.append(boys_name)

    elif len(boys_name) == maximum_length:

        longest_names.append(boys_name)



for girls_name in ssadata.girls:

    if len(girls_name) > maximum_length:

        maximum_length = len(girls_name)

        longest_names = []

        longest_names.append(girls_name)

    elif len(girls_name) == maximum_length:

        longest_names.append(girls_name)



print("The longest names in the data set are " + str(maximum_length) + " letters! They are:")



for name in longest_names:

    print(name)
most_common_name_count = 0

most_common_names = []



for name in ssadata.boys:

    if ssadata.boys[name] > most_common_name_count:

        most_common_name_count = ssadata.boys[name]

        most_common_names = [name]

    elif ssadata.boys[name] == most_common_name_count:

        most_common_names.append(name)

    

print("The most common name(s) for boys is:")

for name in most_common_names:

    print(name)

    print("It is given to this many boys: " + str(ssadata.boys[name]))
most_common_name_count = 0

most_common_names = []



for name in ssadata.girls:

    if ssadata.girls[name] > most_common_name_count:

        most_common_name_count = ssadata.girls[name]

        most_common_names = [name]

    elif ssadata.girls[name] == most_common_name_count:

        most_common_names.append(name)

    

print("The most common name(s) for girls is:")

for name in most_common_names:

    print(name)

    print("It is given to this many girls: " + str(ssadata.girls[name]))
least_common_name_count = 0

least_common_names = []



for name in ssadata.boys:

    if least_common_name_count == 0:

        least_common_name_count = ssadata.boys[name]      

    if ssadata.boys[name] < least_common_name_count:

        least_common_name_count = ssadata.boys[name]

        least_common_names = [name]

    elif ssadata.boys[name] == least_common_name_count:

        least_common_names.append(name)



# Commented out for output on Kaggle - the list is really long!    

#print("The least common name(s) for boys is:")

#

#for name in least_common_names:

#    print(name)

#    print("It is given to this many boys: " + str(ssadata.boys[name]))
least_common_name_count = 0

least_common_names = []



for name in ssadata.girls:

    if least_common_name_count == 0:

        least_common_name_count = ssadata.girls[name]      

    if ssadata.girls[name] < least_common_name_count:

        least_common_name_count = ssadata.girls[name]

        least_common_names = [name]

    elif ssadata.girls[name] == least_common_name_count:

        least_common_names.append(name)



# Commented out for output on Kaggle - the list is really long!    

#print("The least common name(s) for girls is:")

#for name in least_common_names:

#    print(name)

#    print("It is given to this many girls: " + str(ssadata.girls[name]))
numberOfBoys = 0

for boysName in ssadata.boys:

        numberOfBoys = numberOfBoys + ssadata.boys[boysName]



numberOfGirls = 0

for girlsName in ssadata.girls:

        numberOfGirls = numberOfGirls + ssadata.girls[girlsName]



print("There are " + str(numberOfBoys) + " boys and " + str(numberOfGirls) + " girls in the data set.")
# Note: these are asking the same question! The result from the code below

# should convince you of that.



boysNameAlsoGirlsName = 0

for boysName in ssadata.boys:

    if boysName in ssadata.girls:

        boysNameAlsoGirlsName = boysNameAlsoGirlsName + 1



print("There are " + str(boysNameAlsoGirlsName) + " boys names that are also girls names.")



girlsNameAlsoBoysName = 0

for girlsName in ssadata.girls:

    if girlsName in ssadata.boys:

        girlsNameAlsoBoysName = girlsNameAlsoBoysName + 1



print("There are " + str(girlsNameAlsoBoysName) + " girls names that are also boys names.")

# Note the CPU load while this runs and the time it take to process the data!



boysNamesSubsets = 0

for boysName in ssadata.boys:

    match = False

    for otherBoysName in ssadata.boys:

        if not match and boysName in otherBoysName and otherBoysName != boysName:

            boysNamesSubsets = boysNamesSubsets + 1

            match = True



print(str(boysNamesSubsets) + " boys names are subsets of other boys names.")



girlsNamesSubsets = 0

for girlsName in ssadata.girls:

    match = False

    for otherGirlsName in ssadata.girls:

        if not match and girlsName in otherGirlsName and otherGirlsName != girlsName:

            girlsNamesSubsets = girlsNamesSubsets + 1

            match = True



print(str(girlsNamesSubsets) + " girls names are subsets of other girls names.")
mostPopularName = ""

mostPopularNameCount = 0



for girlsName in ssadata.girls:

    if girlsName in ssadata.boys and ssadata.girls[girlsName] > mostPopularNameCount:

        mostPopularNameCount = ssadata.girls[girlsName]

        mostPopularName = girlsName



print("The most popular girls name that is also a boys name is " + mostPopularName + ", with " + str(mostPopularNameCount) + " girls.")

print("There were also " + str(ssadata.boys[mostPopularName]) + " boys named " + mostPopularName + ".")


prefix = input("Input a prefix: ")



boysNamesWithPrefix = 0

for boysName in ssadata.boys:

    if boysName.startswith(prefix):

        boysNamesWithPrefix = boysNamesWithPrefix + 1



print("There are " + str(boysNamesWithPrefix) + " boys names that start with " + prefix + ".")



girlsNamesWithPrefix = 0

for girlsName in ssadata.girls:

    if girlsName.startswith(prefix):

        girlsNamesWithPrefix = girlsNamesWithPrefix + 1



print("There are " + str(girlsNamesWithPrefix) + " girls names that start with " + prefix + ".")