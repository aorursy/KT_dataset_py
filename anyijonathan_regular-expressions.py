# For example matching a time format



import re



line = "Jan  3 07:57:39 Kali sshd[1397]: Failed password 02:12:36 for root from 172.16.12.55 port 34380 ssh2"

regex = "\d+"

result = re.findall(regex, line) # returns all of the digit matches as a list

first_result = re.findall(regex, line)[0] # returns first match

print(result)

print(first_result)
word = "I just realized how interesting coding is"

regex = "\w+"

result = re.findall(regex, word) # returns each word as a list

print(result)
word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"

regex = "colou?rs?" # ? before a string signifies that the string is optional

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "I just realized how interesting coding is"

regex = "\w{3}"

result = re.findall(regex, word) # returns the first three character of each word as a list

print(result)
word = "[Google](http://google.com), [test] \n [itp](http:itp.nyu.edu)"

regex = "\[.*\]" # ? before a string signifies that the string is optional

result = re.findall(regex, word) # return s all of the matches as a list

print(result)
word = "[Google](http://google.com), [test] \n [itp](http:itp.nyu.edu)"

regex = "\[.*?\]" # ? before a string signifies that the string is optional

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"

# regex = "\w+$" # means 1 or more word characters at the end of a line.

# regex = "^\w+$" # means 1 or more word characters at the beginning and end of a line (equally just a line with just one word).

regex = "^\w+" # means the beginning of a line followed by 1 or more word characters

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"

regex = "\\b\w{3}\\b" # this matches 3 word characters specifically

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"

regex = "\\b\w{5,9}\\b" # this matches 5 to 9 word characters specifically

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
# shallow copy copies by referencing the original value, while deep copy copies with no reference

import copy 

x = [1,[2]] 

y = copy.copy(x) 

z = copy.deepcopy(x) 

y is z 
word = "lynk is not the correct spelling of link"

regex = "l[yi]nk" # this matches either link or lynk

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "I am in my 400L, I am currently XX years of age in the year 2018"

regex = "[0-3]{2}" # this matches characters from 0 to 3 and is max of two characters long

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "I am in my 400L, I am currently XX years of age in the year 2018"

regex = "[^0-3]{2}" # this matches characters from 0 to 3 and is max of two characters long

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "I am in my 400L, I am currently XX years of age in the year 2018. My email addresses are stanleydukor@gmail.com, stanleydukor@yahoo.com, stanleydukor@hotmail.edu"

regex = "\w+@\w+\.(?:com|net|org|live|edu)" # this matches email addresses

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "These are some phone numbers 917-555-1234. Also, you can call me at 646.555.1234 and of course I'm always reachable at (212)867-5509"

regex = "\(?\d{3}[-.)]\d{3}[-.]\d{4}" # this matches phone numbers

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "[Google](http://google.com), [test] \n [itp](http:itp.nyu.edu)"

regex = "\[.*?\]\(http.*?\)" # ? matches the name of a link and the link itself

result = re.findall(regex, word) # returns all of the matches as a list

print(result)
word = "2017-07-05 16:04:18.000000"

regex = "\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"

result = re.findall(regex, word) # returns all of the matches as a list

print(result)