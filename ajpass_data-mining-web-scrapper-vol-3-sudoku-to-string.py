# All libraries we will need

import pandas as pd # To store data as a dataframe

import requests # to get the data of an url

from bs4 import BeautifulSoup # to parse the html data and find what we want

import re # Regular expressions library, it may be useful 

print('Setup complete!')
# We need the url of the page we are gonna scrape 

url = 'https://www.nytimes.com/puzzles/sudoku/hard'

response = requests.get(url) # Get content of page
response.text
# Parse the webpage text as html

page_html = BeautifulSoup(response.text, 'html.parser') 

page_html
# Get only script tag

scriptsContainer = page_html.find('script', attrs={'type':'text/javascript'}) #Script tag, contains all sudokus all difficulties, all hints, all solutions

scriptsContainer
# Get content of script

withNoTag = scriptsContainer.contents[0]

withNoTag
import json
# Split the 'window.gameData = ' part, and get json string 

hardSudoku = withNoTag.split('.gameData = ')[1]

hardSudoku

# Make json object

jsonOjb = json.loads(hardSudoku)

jsonOjb

# To access [diffculty] [the data] [the actual puzzle]

sudokuHard = jsonOjb['hard']['puzzle_data']['puzzle']

sudokuMedium = jsonOjb['medium']['puzzle_data']['puzzle']

sudokuEasy = jsonOjb['easy']['puzzle_data']['puzzle']
sudokuHard
# JSON object to string without commas, as is the format the sudoku importer accepts

def jsonPuzzleToString(obj):

    sudokuStr = ''

    for i in obj:

        sudokuStr += str(i)

    return sudokuStr
sudokuHardStr = jsonPuzzleToString(sudokuHard)

sudokuMediumStr = jsonPuzzleToString(sudokuMedium)

sudokuEasyStr = jsonPuzzleToString(sudokuEasy)

sudokuHardStr
sudokuMediumStr
sudokuEasyStr