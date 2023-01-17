# Import libraries
import pandas as pd
import pdfminer
import re
import pprint
# Extract pdf content to a txt file
!pdf2txt.py -o health.txt ../input/health.pdf
# Read and store the file content in a txt file
pdfTxtFile = '../input/health.txt'
pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')

# loop over all the lines
for line in pdf_txt:
    print(repr(line))
# Extract Country Names
# Create a clean function to format the lines
def clean(line):
        #line = line.strip('\n') # remove leading and training '\n' 
        line = line.strip() # remove leading and trailing while spaces
        line = line.strip('x') # remove leading and trailing x
        line = line.strip('–') # remove leading and trailing -
        return line
    
pdfTxTFile = './health.txt'
pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')
#
# Create a Boolean variable as a switch
#
isCountryName =False
countryNames = []
previous_line = ''

for line in pdf_txt:        
    if line.startswith('and areas'):
        isCountryName = True
    #
    # If isCountryName is turned on, and the line is equal to a new line character,
    # Set isCountryName to False.
    #
    elif isCountryName and re.match(r"^\n+$", line) != None:
        isCountryName = False
        
    #
    # Remove the lines with only digits
    # Remove the extra digits in country name
    # Add the country names into a list
    #     
        
    if isCountryName and re.match(r"^\d+$", line) == None:
        line = re.sub(" \d+", "", line)    # remove digits in the country name
        
        if previous_line.endswith('Republic of \n'):
            line = ' '.join([clean(previous_line), clean(line)])
            del countryNames[-1]
            countryNames.append(line)
        elif re.match(r"^and areas", line) == None: # remove the lines named 'as areas'
            countryNames.append(clean(line))

        previous_line = line  

pprint.pprint(countryNames)
# Verify the number of countries
len(countryNames)
# Count how many countries in each page and store the numbers in a list

pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')
isNewPage = 0
numberOfCountries = 0
recordsPerPage = []
numRecords = 0

# Create a clean function to format the lines
def clean(line):
        line = line.strip('\n') # remove leading and training '\n' 
        line = line.strip()     # remove leading and trailing while spaces
        line = line.strip('x')  # remove leading and trailing x
        line = line.strip('–')  # remove leading and trailing -
        line = line.strip('‡')  # remove leading and trailing ‡
        line = line.strip('*')  # remove leading and trailing *
        return line

for line in pdf_txt:        
    if line.startswith('and areas'):    # Setup a page split switch
        isCountryName = True
        numRecords = 0
    #
    # If isCountryName is turned on, and the line is equal to a new line character,
    # Set isCountryName to False.
    #
    elif isCountryName and re.match(r"^\n+$", line) != None:
        isCountryName = False
        recordsPerPage.append(numRecords)
    #
    # Remove the lines with only digits
    # Remove the extra digits in country name
    # Add the country names into a list
    #     
        
    if isCountryName and re.match(r"^\d+$", line) == None:
        line = re.sub(" \d+", "", line)                  # remove digits in the country name
        
        if previous_line.endswith('Republic of \n'):     # merge the 2 lines of 1 county name
            line = ' '.join([clean(previous_line), clean(line)])
            del countryNames[-1]
            countryNames.append(line)
            
        elif re.match(r"^and areas", line) == None:     # count the number of countries in each page
            countryNames.append(clean(line))
            numRecords += 1
            

        previous_line = line  
        
print(recordsPerPage)        
# Create a clean function to format the lines
def clean(line):
        line = line.strip('\n') # remove leading and training '\n' 
        line = line.strip() # remove leading and trailing while spaces
        line = line.strip('x') # remove leading and trailing x
        line = line.strip('–') # remove leading and trailing -
        line = line.strip('‡') # remove leading and trailing ‡
        line = line.strip('*') # remove leading and trailing *
        return line

def extract(pdfTextFile):    
    pdfTxTFile = './health.txt'
    pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')

    isCountryName =False
    countryNames = []
    previous_line = ''
    numberOfCountries = 0
    recordsPerPage = []
    numRecords = 0

    for line in pdf_txt:        
        if line.startswith('and areas'):
            isCountryName = True
            numRecords = 0
   
        elif isCountryName and re.match(r"^\n+$", line) != None:
            isCountryName = False
            recordsPerPage.append(numRecords)
    
        
        if isCountryName and re.match(r"^\d+$", line) == None:
            line = re.sub(" \d+", "", line)                      # remove digits in the country name
        
            if previous_line.endswith('Republic of \n'):
                line = ' '.join([clean(previous_line), clean(line)])
                del countryNames[-1]
                countryNames.append(line)
                numRecords -= 1
            
            elif re.match(r"^and areas", line) == None:         # remove the lines named 'as areas'
                countryNames.append(clean(line))
                numRecords += 1
            previous_line = line  
    return countryNames, recordsPerPage
    
pdfTxtFile = './health.txt'
countryNames, recordsPerpage = extract(pdfTxtFile)

# Setup regular expression
regx = re.compile("^(\d{1,3}|–|.*\s+\d{2})\s?(x?|\*\*|‡|‡\*\*)$")

pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')

totalCols = 22
pageNum = -1
numRecords = 0
columnIndex = 0
isdata = False
data = {}
index = 0
count = 0

for i in range(totalCols):
    data[i] = []


for line in pdf_txt:
    if line.startswith('and areas'):
        pageNum += 1
        numRecords = recordsPerPage[pageNum]
        columnIndex = 0
        index = 0

        
    if pageNum < 3 and regx.match(line) != None and columnIndex < totalCols:
        line = re.sub("\D", "", line.strip('–'))
        data[columnIndex].append(line)
        index += 1
        if index % numRecords == 0:
            columnIndex += 1 
        
        
    if pageNum == 3 and regx.match(line) != None and 18 <= columnIndex and columnIndex < totalCols:
         line = re.sub("\D", "", line.strip('–'))
        
         count += 1

         if count >= recordsPerPage[3]:

             data[columnIndex].append(line)
             index += 1
             if index % numRecords == 0:
                 columnIndex += 1            
        
         if count >= recordsPerPage[pageNum] + recordsPerPage[pageNum] - 1:
             count = 0
            
            
        
    if pageNum == 3 and regx.match(line) != None and columnIndex < 18:
        line = re.sub("\D", "", line.strip('–'))
        data[columnIndex].append(line)
        index += 1
        
        if index % numRecords == 0:
            columnIndex += 1 
        
  
               
  
df = pd.DataFrame(data, index = countryNames)
df.to_csv('health.csv')
##### Total Countries:
len(countryNames)
##### Countries in each page:
recordsPerPage
##### Total Columns:
totalCols