import re # as usual we need to import `re` for general regex operations
my_string = "the 80s music hits were the 70s much better that the 90s"



print(re.findall(r"the\s\d+s$", my_string)) #pattern to find the Xs from string



print("\nIt will not return either first or middle 'the Xs' but will return only last one")
my_string = "the 80s music hits were the 70s much better that the 90s"



print(re.findall(r"^the\s\d+s", my_string)) #pattern to find the Xs from string



print("\nIt will not return either last or middle 'the Xs' but will return only first one")
my_string = "the 80s music hits were the 70s much better that the 90s"



print(re.findall(r"the\s\d+s", my_string)) #Not using either ^ or $



print("\nIt will every matched string :)")
my_string = "Elephants are the world's largest land animal! I would love to see an elephant one day"



print(re.findall(r"Elephant|elephant", my_string))
vowels = []

no_vowels = []

names = ["Rishabh","Tridev","Arihant","umang","aniket","ojash","Praful","rahul"]

print(names)

#find all names starting with vowels(Upper and lowercase both)



print("******************Names starting with vowels********************")



regex_pattern = r"^[aeiouAEIOU].*" #starting letter should be one of the letter mentioned in square bracket

for name in names:

    if re.findall(regex_pattern,name):

        vowels.append(name)

print(vowels)

    

#find all names that don't start with vowels(Upper and lowercase both)



print("******************Names not starting with vowels********************")



regex_pattern = r"^[^aeiouAEIOU].*" #starting letter anything except these characters mentioned in square brackets

for name in names:

    if re.findall(regex_pattern,name):

        no_vowels.append(name)

        

print(no_vowels)
links = ["visit this link www.hdtyfdtrxukuyof=+donalsd+trump.com ","i'm giving my google drive link www.gdrive-rishabh.com"]



print(emails)



print("links extracted using dot(.)")

for link in links:

    print(re.findall(r"www.*com",link)) #dot(.) matchs everything
emails = ['n.john.smith@gmail.com', '87victory@hotmail.com', '!#mary-=@msca.net']
#As per constraints given by company



regex_pattern = r"[a-zA-z0-9!#%&*$]*@\w+.com"



for email in emails:

    if re.findall(regex_pattern,email):

        print("{} is a valid email".format(email))

    else:

        print("{} is not a valid email".format(email))
passwords =  ['Apple34!rose', 'My87hou#4$', 'abc123']
regex_pattern = r"[a-zA-z0-9*#$%!&]{8,20}"



for password in passwords:

    if re.findall(regex_pattern,password):

        print("The password {pass_example} is a valid password".format(pass_example=password))

    else:

        print("The password {pass_example} is invalid".format(pass_example=password))   