## import math related lib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3


## define functions
def con_open(sql_file):
	con_handle = sqlite3.connect(sql_file)
	return con_handle

def con_close(con_handle):
	con_handle.close()

def config_plot(xlabel, ylabel, title, grid = True):
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if (grid):
		plt.grid()


## operation code

Data = con_open('../input/database.sqlite') 
# open database

cmd = 'SELECT DISTINCT Year FROM NationalNames'
year = pd.read_sql(cmd, Data)
# get the list of year in database

cmd1 = 'SELECT * FROM NationalNames WHERE Year = ? AND Gender = ? AND Count > ?'
cmd2 = 'SELECT * FROM NationalNames WHERE Year = ? AND Gender = ? AND Count < ?'
cmd3 = 'SELECT SUM(Count) AS SUM_C FROM NationalNames WHERE Year = ? AND Gender = ?'
nb_15000 = []
nb_1000 = []
nb_500 = []
nb = []
for row in year['Year']:
	param = (str(row), 'M', '15000')
	select = pd.read_sql(cmd1, Data, params = param)
	nb_15000.append(len(select))
	# count the number of boy names used over 15000 times each year
	param = (str(row), 'M', '1000')
	select = pd.read_sql(cmd1, Data, params = param)
	nb_1000.append(len(select)-nb_15000[-1])
	# count the number of boy names used over 1000 times each year
	param = (str(row), 'M', '500')
	select = pd.read_sql(cmd2, Data, params = param)
	nb_500.append(len(select))
	# count the number of boy names used less 500 times each year
	param = (str(row), 'M')
	select = pd.read_sql(cmd3, Data, params = param)
	nb.append(select['SUM_C'])
	# count the number of boy names used over 10000 times each year



plt.figure(1)
plt.plot(year, nb, 'b-')
config_plot('Year', '# of baby', '# of boys each year', True)
plt.figure(2)
plt.plot(year, nb_15000, 'r-')
config_plot('Year', '# of names', 'names used over 15000 times', True)
plt.figure(3)
plt.plot(year, nb_1000, 'y-')
config_plot('Year', '# of names', 'names used over 1000 times', True)
plt.figure(4)
plt.plot(year, nb_500, 'g-')
config_plot('Year', '# of names', 'names used less than 500 times', True)


plt.show()
