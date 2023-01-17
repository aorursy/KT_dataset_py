import urllib.request, re
from bs4 import BeautifulSoup

#Use urllib to access the 10-k form
urlLink= input("Enter the report URL link: ")
pageOpen = urllib.request.urlopen(urlLink) #open the url link and save as a variable "pageopen"
pageRead = pageOpen.read() #read the opened url link and save as "pageread"

#Save the html into your computer
filename = input("Enter the file name for html and txt: ")
htmlname = filename +".htm"
htmlfile = open(htmlname, 'wb')
htmlfile.write(pageRead)
htmlfile.close()

#Open the saved htmlfile with r, which means read only
fileOpen = open(htmlname, 'r')
fileRead = fileOpen.read()
fileRead
# To select all the contents in item 1a, you need to read the html code and find the unique tag for "ITEM 1A"

#For Google
regex = 'bold;\">ITEM&#160;1A\.(.+?)bold;\">ITEM&#160;1B\.'
match = re.search(regex, fileRead, flags=re.IGNORECASE)
part = match.group(1) #save the extracted part into variable "part"
#For Amazon
regex = 'bold;\">ITEM&#160;1A\.(.+?)bold;\">ITEM&#160;1B\.'
match = re.search(regex, fileRead, flags=re.IGNORECASE)
part = match.group(1) #save the extracted part into variable "part"
#For Apple
regex = 'bold;\">ITEM 1A\.(.+?)bold;\">ITEM 1B\.'
match = re.search(regex, fileRead, flags=re.IGNORECASE)
part = match.group(1) #save the extracted part into variable "part"
#For Facebook
regex ='bold;\">ITEM 1A\.(.+?)bold;\">ITEM 1B\.'
match = re.search(regex, fileRead, flags=re.IGNORECASE)
part = match.group(1) #save the extracted part into variable "part"
#For Netflix
regex ='bold;\">ITEM 1A\.(.+?)bold;\">ITEM 1B\.'
match = re.search(regex, fileRead, flags=re.IGNORECASE)
part = match.group(1) #save the extracted part into variable "part"
#For Red HAt
regex ='bold;\">ITEM 1A\.(.+?)bold;\">ITEM 1B\.'
match = re.search(regex, fileRead, flags=re.IGNORECASE)
part = match.group(1) #save the extracted part into variable "part"
# BeautifulSoup is a lib for html data, we use it to get the texts we need and remove to tag from html
partview = BeautifulSoup(part, 'html.parser')

for table in partview.find_all("table"):
    table.extract()
    
notable = str(partview)

content = BeautifulSoup(notable, 'html.parser').get_text(separator=" ")#
print(content)
# Import the nltk lib
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

text = content.lower()

# We can use "word_tokenize" to make our paragraph into tokens
tokens = word_tokenize(text)
# tokens

#Remove stopwords like "is", "a" in the data
tokens_without_sw = [word for word in tokens if not word in stopwords.words()]

print(tokens_without_sw)
txtname = filename + ".txt"
txtfile = open(txtname, 'w')
txtfile.write(str(tokens_without_sw))
txtfile.close()