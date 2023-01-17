# normal string vs raw string

path = "C:\desktop\Ravikanth"  #string

print("string:",path)
path= r"C:\desktop\Ravikanth"  #raw string

print("raw string:",path)
import re



#match a word at the beginning of a string



result = re.match('Kaggle',r'Kaggle is the largest data science community of India') 

print(result)



result_2 = re.match('largest',r'Kaggle  is the largest data science community of India') 

print(result_2)
print(result.group())  #returns the total matches
# search for the pattern "founded" in a given string

result = re.search('founded',r'Andrew NG founded Coursera. He also founded deeplearning.ai')

print(result.group())
result = re.findall('founded',r'Andrew NG founded Coursera. He also founded deeplearning.ai')  

print(result)
str = r'Kaggle is the largest data science community of India'



x = re.findall("\AKaggle", str)



print(x)
str = r'Analytics Vidhya is the largest Analytics community of India'



x = re.findall("\AVidhya", str)



print(x)
#Check if there is any word that ends with "est"

x = re.findall(r"est\b", str)

print(x)
str = r'The good, The bad, and The Ugly '



x = re.findall(r"\Bhe", str)



print(x)
str = "2 million monthly visits in Jan'19."



#Check if the string contains any digits (numbers from 0-9):

x = re.findall("\d", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "2 million monthly visits in Jan'19."



# Check if the string contains any digits (numbers from 0-9):

# adding '+' after '\d' will continue to extract digits till encounters a space

x = re.findall("\d+", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "2 million monthly visits in Jan'19."



#Check if the word character does not contain any digits (numbers from 0-9):

x = re.findall("\D", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "2 million monthly visits'19"



#Check if the word does not contain any digits (numbers from 0-9):



x = re.findall("\D+", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "2 million monthly visits!"



#returns a match at every word character (characters from a to Z, digits from 0-9, and the underscore _ character)



x = re.findall("\w",str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "2 million monthly visits!"



#returns a match at every word (characters from a to Z, digits from 0-9, and the underscore _ character)



x = re.findall("\w+",str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "2 million monthly visits9!"



#returns a match at every NON word character (characters NOT between a and Z. Like "!", "?" white-space etc.):



x = re.findall("\W", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "rohan and rohit recently published a research paper!" 



#Search for a string that starts with "ro", followed by three (any) characters



x = re.findall("ro.", str)

x2 = re.findall("ro...", str)



print(x)

print(x2)
str = "Data Science"



#Check if the string starts with 'Data':

x = re.findall("^Data", str)



if (x):

  print("Yes, the string starts with 'Data'")

else:

  print("No match")

  

#print(x)  
# try with a different string

str2 = "Big Data"



#Check if the string starts with 'Data':

x2 = re.findall("^Data", str2)



if (x2):

  print("Yes, the string starts with 'data'")

else:

  print("No match")

  

#print(x2)  
str = "Data Science"



#Check if the string ends with 'Science':



x = re.findall("Science$", str)



if (x):

  print("Yes, the string ends with 'Science'")



else:

  print("No match")

  

#print(x)
str = "easy easssy eay ey"



#Check if the string contains "ea" followed by 0 or more "s" characters and ending with y

x = re.findall("eas*y", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
#Check if the string contains "ea" followed by 1 or more "s" characters and ends with y 

x = re.findall("eas+y", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
x = re.findall("eas?y",str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "Analytics Vidhya is the largest data science community of India"



#Check if the string contains either "data" or "India":



x = re.findall("data|India", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
# try with a different string

str = "Analytics Vidhya is one of the largest data science communities"



#Check if the string contains either "data" or "India":



x = re.findall("data|India", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "Analytics Vidhya is the largest data science community of India"



#Check for the characters y, d, or h, in the above string

x = re.findall("[ydh]", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "Analytics Vidhya is the largest data science community of India"



#Check for the characters between a and g, in the above string

x = re.findall("[a-g]", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "Mars' average distance from the Sun is roughly 230 million km and its orbital period is 687 (Earth) days."



# extract the numbers starting with 0 to 4 from in the above string

x = re.findall(r"\b[0-4]\d+", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "Analytics Vidhya is the largest data sciece community of India"



#Check if every word character has characters than y, d, or h



x = re.findall("[^ydh]", str)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
str = "@AV Largest Data Science community #AV!!"



# extract words that start with a special character

x = re.findall("[^a-zA-Z0-9 ]\w+", str)



print(x)
str = 'Send a mail to rohan.1997@gmail.com, smith_david34@yahoo.com and priya@yahoo.com about the meeting @2PM'

  

# \w matches any alpha numeric character 

# + for repeats a character one or more times 

#x = re.findall('\w+@\w+\.com', str)     

x = re.findall('[a-zA-Z0-9._-]+@\w+\.com', str)     

  

# Printing of List 

print(x) 
text = "London Olympic 2012 was held from 2012-07-27 to 2012/08/12."



# '\d{4}' repeats '\d' 4 times

match = re.findall('\d{4}.\d{2}.\d{2}', text)

print(match)
text="London Olympic 2012 was held from 27 Jul 2012 to 12-Aug-2012."



match = re.findall('\d{2}.\w{3}.\d{4}', text)



print(match)
# extract dates with varying lengths

text="London Olympic 2012 was held from 27 July 2012 to 12 August 2012."



#'\w{3,10}' repeats '\w' 3 to 10 times

match = re.findall('\d{2}.\w{3,10}.\d{4}', text)



print(match)
import pandas as pd



# load dataset

data=pd.read_csv("../input/titanic.csv")
data.head()
# print a few passenger names

data['Name'].head(10)
name = "Allen, Mr. William Henry"

name2 = name.split(".")

name2[0].split(',')
title=data['Name'].apply(lambda x: x.split(".")[0].split(",")[1])

title.value_counts()
def split_it(name):

    return re.findall("\w+\.",name)[0]

title=data['Name'].apply(lambda x: split_it(x))

title.value_counts().sum()