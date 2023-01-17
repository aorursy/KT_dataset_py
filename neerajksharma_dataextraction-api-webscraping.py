#import requests module

import requests
# store the URI in the url variable.

url = 'https://api.data.gov/ed/collegescorecard/v1/schools?api_key=7g1Yt0xxEVGxrd8hovbQunznrPE4oxyJivQsoL4h'
#using get command, returns response object

result = requests.get(url)
# Get the status code

result.status_code
#Get the complete results in the text format.

result.text
# Improt the module to read html code and capture the data.

from bs4 import BeautifulSoup
#HTML Created. Please refer the result in the next command for exact page. 

html_string = """

<!doctyp html>

<html lang="en">

<head>

    <title>Doing Data Science With Python</title>

</head>

<body>

    <h1 style="color:#F15B2A;">Data Extraction: WebScrapping</h1>

    <p id="author">Author : Neeraj Sharma</p>

    <p id="description">This notebook will help you to learn webscraping.</p>

    

    <h3 style="color:#404040;">Where does Data Scientist Spends their time?</h3>

    <table id="workdistribution" style="width:100%">

        <tr>

            <th>Work</th>

            <th>% of time</th>

        </tr>

        <tr>

            <td>Data Extraction</td>

            <td>20</td>

        </tr>

          <tr>

            <td>Data Organize</td>

            <td>60</td>

        </tr>

          <tr>

            <td>Building Model and Evaluation</td>

            <td>10</td>

        </tr>

          <tr>

            <td>Presentation and Other tasks</td>

            <td>10</td>

        </tr>

    </table>

    </body>

    </html>

"""
#Display HTML page in the juyper notedbook

from IPython.core.display import display, HTML

display(HTML(html_string))
#use beautiful soup to read html string and create an object

ps = BeautifulSoup(html_string)
# print the value of ps

print(ps)
# print the body from html

body = ps.find(name="body")

print(body)
#use text attribute to get the content of the tag

print(body.find(name="h1").text)
#print the value of <p> tag

print(body.find(name='p'))
#print the value of all paragraphs tag

print(body.findAll(name='p'))
#loop through  each element in <p> tag and print them one by one

for p in body.findAll(name='p'):

    print(p.text)
# add attributes author also in selection process

print(body.findAll(name='p', attrs={"id":"author"}))
#print attributes description along with paragraph

print(body.findAll(name='p', attrs={"id":"description"}))
# Read and print columns of the table. this can be later stored in the variable and create df to work further. 

#body

body = ps.find(name='body')

#module table 

module_table = body.find(name='table', attrs={"id":"workdistribution"})

#iterate through each row in the table (skipping the first row)

for row in module_table.findAll(name='tr')[1:]:

    #module title

    title = row.findAll(name='td')[0].text

    #module duration

    duration = int(row.findAll(name='td')[1].text)

    print(title, duration)