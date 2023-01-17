import csv

with open('../input/numbers.csv', newline='') as csvfile:

  numReader = csv.reader(csvfile, delimiter=',')

  for row in numReader:

    print(', '.join(row))
with open('../input/numbers.csv', newline='') as csvfile:    

  numReader = csv.reader(csvfile, delimiter=',')

  next(numReader)

  for row in numReader:

    print(sum(map(int, row)))
import pandas as pd

matrix = pd.read_csv('../input/numbers.csv').values

matrix.transpose()