# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in

from IPython.core.display import display, HTML

import json



# Description:

# This code seeks the words in mKeyWordList below in each of the papers provided.

# The keywords were derived in the task outlined.  The keyword list can

# be modified to search for any terms.  Each group of text is search for both

# COVID and the search term.  If you do not want COVID included, that can

# be removed around line 59 where you see "and lText.find('COVID') != -1": 



# Input:

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter)

# will list all files under the input directory



# Output:

# Results are created in the /kaggle/working directory in a file called Results.html

# After this cell is run, a link to the output is created and launches

# the display of that content in a new tab.  This file can also be downloaded.

import os



mOutputResults = {}

mTask = '1.  Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.'

mKeyWordList = ['incubation', 'contagious', 'shedding', 'surface', 'diagnostics', 'model', 'immune', 'transmission', 'environment', 'protection', 'recover']

mTextHeader = 'Text Excerpt:'

mDir = '/kaggle/input'



def iterateFiles():

    for dirname, _, filenames in os.walk(mDir):

        for filename in filenames:

            pathFileName = os.path.join(dirname, filename)

            if pathFileName.__contains__('.json'):

                with open(pathFileName) as f:

                    data = json.load(f)

                    findTerms(data, pathFileName)

                    

def findTerms(iData, iPathFile):

    try:

        for lKeyword in mKeyWordList:

            lFinalStr = '<br>'

            lFinalStr += '<h3>Paper ID:</h3>'

            lFinalStr += '<p>' + iData['paper_id'] + '</p>'

            lFinalStr += '<br>'

            lFinalStr += '<h3>Title:</h3>'

            lFinalStr += '<p>' + iData['metadata']['title'] + '</p>'

            lAuthStr = ''

            for lAuthor in iData['metadata']['authors']:

                lAuthStr += lAuthor['first'] + ' ' + lAuthor['last'] + '(email: ' + lAuthor['email'] + '), '

            lFinalStr += '<br>'

            lFinalStr += '<h3>Authors (First and Last Name):</h3>'

            lFinalStr += '<p>' + lAuthStr.rstrip(", ") + '</p>'

            lFinalStr += '<br>'

            lFinalStr += '<h3>Reference File:</h3>'

            lFinalStr += '<p>' + iPathFile + '</p>'        

            lFinalStr += '<br>'



            for key in iData['body_text']:

                lText = key['text'].upper()

                lUpperKey = lKeyword.upper()

                if lText.find(lUpperKey) != -1 and lText.find('COVID') != -1:

                    lFinalStr += '<h3>' + mTextHeader + '</h3>'

                    lFinalStr += '<h3>Section: ' + key['section'] + '</h3>'

                    lFinalStr += '<p>' + key['text'].replace(lKeyword.lower(), '<mark>' + lKeyword.lower() + '</mark>').replace(lKeyword.capitalize(), '<mark>' + lKeyword.capitalize() + '</mark>') + '</p>'

            if lFinalStr.find(mTextHeader) != -1:

                if lKeyword in mOutputResults.keys():

                    mOutputResults[lKeyword] = mOutputResults.get(lKeyword) + '\n' + lFinalStr

                else:

                    mOutputResults[lKeyword] = lFinalStr

    except:

        pass





def ouputResults():

    lOutputFile = 'Results.html'

    display(HTML('<h1>Results</h1>'))

    display(HTML('<p>Results have been output to the link below:</p>'))

    display(HTML('<a href="' + lOutputFile + '" target="_blank">Results - Click here...</a>'))

    #write file

    with open(lOutputFile, 'w', encoding='UTF-8') as f:

        f.write('<html>')

        f.write('<head><meta charset="UTF-8"><style> mark {background-color: yellow; color: black;}</style></head>')

        f.write('<body>')

        f.write('<h1>Task:</h1>')

        f.write(mTask)

        f.write('<h1>Approach:</h1>')

        f.write('<p>Searched each text block in the documents for COVID and the terms below.  The resulting text blocks are grouped by the terms below.</p>')

        f.write('<p>')

        f.write('<ul style="list-style-type:circle;">')

        for lItem in mKeyWordList:

            f.write('<li>'+ lItem + '</li>')

        f.write('</ul>')

        f.write('</p>')

        lCount = 1

        for k, v in mOutputResults.items():

            f.write('<hr>')

            f.write('<h2>' + str(lCount) + '. Search Terms: COVID and <mark>' + k.capitalize() + '</mark></h2>')

            f.write('<hr>')

            f.write(v)

            lCount +=1

        f.write('<body>')

        f.write('</html>')



# Any results you write to the current directory are saved as output.



display(HTML('<hr>'))

display(HTML('<p>Beginning search.......</p>'))

iterateFiles()

display(HTML('<hr>'))

ouputResults()
