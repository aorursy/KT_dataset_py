# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install python-chess
import chess

import chess.pgn

import io

from IPython.display import clear_output

import re

import datetime

import csv

import sys
pgn = open("/kaggle/input/pgn-lichess/chess-dataset.pgn")
start = datetime.datetime.now()
dictionary_position_score = {}

number_of_positions = 0

number_of_games_parsed = 0

flag = True
with open('fen_to_stockfish_evaluation.csv', 'w') as f:    

    while number_of_positions < 10**7 and (datetime.datetime.now() - start).seconds < 7*3600:

        

        game = chess.pgn.read_game(pgn)

        if "%eval" in str(game).split("\n\n")[1]:



            #getting scores

            subStr = re.findall(r'%eval(.+?)\]',str(game))

            subStrCorrected = [-99999 if i.startswith(" #") else i for i in subStr]

            scores = [float(score) for score in subStrCorrected]



            #get positions

            board = game.board()

            score_index = 0        

            for move in game.mainline_moves():

                if score_index >= len(scores):

                    break

                input = board.fen()

                if input not in dictionary_position_score and scores[score_index] != float(-99999):

                    f.write("%s, %s\n" % (input, scores[score_index]))            

                    number_of_positions = number_of_positions+1

                    if number_of_positions % 10000 == 0 and flag:  

                        os.system('echo '+str(number_of_positions))

                        flag = False

                    flag = True

                dictionary_position_score[input] = scores[score_index]

                score_index = score_index + 1

                board.push(move)



            number_of_games_parsed = number_of_games_parsed + 1



            clear_output(wait=True)

            print("number of positions = {} | number of games parsed = {} / 1592460".format(str(number_of_positions) ,str(number_of_games_parsed)))
sys.__stdout__.write("your message")