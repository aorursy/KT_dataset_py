# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#data=pd.read_csv('../input/sudoku.csv')

import numpy as np

quizzes = np.zeros((1000000, 81), np.int32)

solutions = np.zeros((1000000, 81), np.int32)

for i, line in enumerate(open('../input/sudoku.csv', 'r').read().splitlines()[1:]):

    quiz, solution = line.split(",")

    for j, q_s in enumerate(zip(quiz, solution)):

        q, s = q_s

        quizzes[i, j] = q

        solutions[i, j] = s

quizzes = quizzes.reshape((-1, 9, 9))

solutions = solutions.reshape((-1, 9, 9))
