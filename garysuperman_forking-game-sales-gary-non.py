# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data_set = pd.read_csv("../input/vgsales.csv")
NintendoJP = data_set.loc[data_set['Publisher'] == 'Nintendo']
Nintendo2005 = NintendoJP.loc[NintendoJP['Year'] == 2005.0]
Nintendo2006 = NintendoJP.loc[NintendoJP['Year'] == 2006.0]
Nintendo2007 = NintendoJP.loc[NintendoJP['Year'] == 2007.0]
Nintendo2008 = NintendoJP.loc[NintendoJP['Year'] == 2008.0]
Nintendo2009 = NintendoJP.loc[NintendoJP['Year'] == 2009.0]
Nintendo2010 = NintendoJP.loc[NintendoJP['Year'] == 2010.0]
Nintendo2011 = NintendoJP.loc[NintendoJP['Year'] == 2011.0]
Nintendo2012 = NintendoJP.loc[NintendoJP['Year'] == 2012.0]
Nintendo2013 = NintendoJP.loc[NintendoJP['Year'] == 2013.0]
Nintendo2005_Sales = np.sum(Nintendo2005['Global_Sales'])
Nintendo2006_Sales = np.sum(Nintendo2006['Global_Sales'])
Nintendo2007_Sales = np.sum(Nintendo2007['Global_Sales'])
Nintendo2008_Sales = np.sum(Nintendo2008['Global_Sales'])
Nintendo2009_Sales = np.sum(Nintendo2009['Global_Sales'])
Nintendo2010_Sales = np.sum(Nintendo2010['Global_Sales'])
Nintendo2011_Sales = np.sum(Nintendo2011['Global_Sales'])
Nintendo2012_Sales = np.sum(Nintendo2012['Global_Sales'])
Nintendo2013_Sales = np.sum(Nintendo2013['Global_Sales'])

Microsoft = data_set.loc[data_set['Publisher'] == 'Microsoft Game Studios']
Microsoft2005 = Microsoft.loc[Microsoft['Year'] == 2005.0]
Microsoft2006 = Microsoft.loc[Microsoft['Year'] == 2006.0]
Microsoft2007 = Microsoft.loc[Microsoft['Year'] == 2007.0]
Microsoft2008 = Microsoft.loc[Microsoft['Year'] == 2008.0]
Microsoft2009 = Microsoft.loc[Microsoft['Year'] == 2009.0]
Microsoft2010 = Microsoft.loc[Microsoft['Year'] == 2010.0]
Microsoft2011 = Microsoft.loc[Microsoft['Year'] == 2011.0]
Microsoft2012 = Microsoft.loc[Microsoft['Year'] == 2012.0]
Microsoft2013 = Microsoft.loc[Microsoft['Year'] == 2013.0]
Microsoft2005_Sales = np.sum(Microsoft2005['Global_Sales'])
Microsoft2006_Sales = np.sum(Microsoft2006['Global_Sales'])
Microsoft2007_Sales = np.sum(Microsoft2007['Global_Sales'])
Microsoft2008_Sales = np.sum(Microsoft2008['Global_Sales'])
Microsoft2009_Sales = np.sum(Microsoft2009['Global_Sales'])
Microsoft2010_Sales = np.sum(Microsoft2010['Global_Sales'])
Microsoft2011_Sales = np.sum(Microsoft2011['Global_Sales'])
Microsoft2012_Sales = np.sum(Microsoft2012['Global_Sales'])
Microsoft2013_Sales = np.sum(Microsoft2013['Global_Sales'])

Ubisoft = data_set.loc[data_set['Publisher'] == 'Ubisoft']
Ubisoft2005 = Ubisoft.loc[Ubisoft['Year'] == 2005.0]
Ubisoft2006 = Ubisoft.loc[Ubisoft['Year'] == 2006.0]
Ubisoft2007 = Ubisoft.loc[Ubisoft['Year'] == 2007.0]
Ubisoft2008 = Ubisoft.loc[Ubisoft['Year'] == 2008.0]
Ubisoft2009 = Ubisoft.loc[Ubisoft['Year'] == 2009.0]
Ubisoft2010 = Ubisoft.loc[Ubisoft['Year'] == 2010.0]
Ubisoft2011 = Ubisoft.loc[Ubisoft['Year'] == 2011.0]
Ubisoft2012 = Ubisoft.loc[Ubisoft['Year'] == 2012.0]
Ubisoft2013 = Ubisoft.loc[Ubisoft['Year'] == 2013.0]
Ubisoft2005_Sales = np.sum(Ubisoft2005['Global_Sales'])
Ubisoft2006_Sales = np.sum(Ubisoft2006['Global_Sales'])
Ubisoft2007_Sales = np.sum(Ubisoft2007['Global_Sales'])
Ubisoft2008_Sales = np.sum(Ubisoft2008['Global_Sales'])
Ubisoft2009_Sales = np.sum(Ubisoft2009['Global_Sales'])
Ubisoft2010_Sales = np.sum(Ubisoft2010['Global_Sales'])
Ubisoft2011_Sales = np.sum(Ubisoft2011['Global_Sales'])
Ubisoft2012_Sales = np.sum(Ubisoft2012['Global_Sales'])
Ubisoft2013_Sales = np.sum(Ubisoft2013['Global_Sales'])

Sony = data_set.loc[data_set['Publisher'] == 'Sony Computer Entertainment']
Sony2005 = Sony.loc[Sony['Year'] == 2005.0]
Sony2006 = Sony.loc[Sony['Year'] == 2006.0]
Sony2007 = Sony.loc[Sony['Year'] == 2007.0]
Sony2008 = Sony.loc[Sony['Year'] == 2008.0]
Sony2009 = Sony.loc[Sony['Year'] == 2009.0]
Sony2010 = Sony.loc[Sony['Year'] == 2010.0]
Sony2011 = Sony.loc[Sony['Year'] == 2011.0]
Sony2012 = Sony.loc[Sony['Year'] == 2012.0]
Sony2013 = Sony.loc[Sony['Year'] == 2013.0]
Sony2005_Sales = np.sum(Sony2005['Global_Sales'])
Sony2006_Sales = np.sum(Sony2006['Global_Sales'])
Sony2007_Sales = np.sum(Sony2007['Global_Sales'])
Sony2008_Sales = np.sum(Sony2008['Global_Sales'])
Sony2009_Sales = np.sum(Sony2009['Global_Sales'])
Sony2010_Sales = np.sum(Sony2010['Global_Sales'])
Sony2011_Sales = np.sum(Sony2011['Global_Sales'])
Sony2012_Sales = np.sum(Sony2012['Global_Sales'])
Sony2013_Sales = np.sum(Sony2013['Global_Sales'])

N = pd.Series([Nintendo2005_Sales, Nintendo2006_Sales, Nintendo2007_Sales, Nintendo2008_Sales, Nintendo2009_Sales, Nintendo2010_Sales, Nintendo2011_Sales, Nintendo2012_Sales, Nintendo2013_Sales], [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])
M =  pd.Series([Microsoft2005_Sales,Microsoft2006_Sales, Microsoft2007_Sales, Microsoft2008_Sales, Microsoft2009_Sales, Microsoft2010_Sales, Microsoft2011_Sales, Microsoft2012_Sales, Microsoft2013_Sales], [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])
U = pd.Series([Ubisoft2005_Sales, Ubisoft2006_Sales, Ubisoft2007_Sales, Ubisoft2008_Sales, Ubisoft2009_Sales, Ubisoft2010_Sales, Ubisoft2011_Sales, Ubisoft2012_Sales, Ubisoft2013_Sales], [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])
S = pd.Series([Sony2005_Sales, Sony2006_Sales, Sony2007_Sales, Sony2008_Sales, Sony2009_Sales, Sony2010_Sales, Sony2011_Sales, Sony2012_Sales, Sony2013_Sales], [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])

plt.plot(N, 'r--', M, '--', U, 'b--', S, 'g--')
plt.ylabel('Global Sales')
plt.show()
Action = data_set.loc[data_set['Genre'] == 'Action']
Adventure = data_set.loc[data_set['Genre'] == 'Adventure']
Fighting = data_set.loc[data_set['Genre'] == 'Fighting']
Misc = data_set.loc[data_set['Genre'] == 'Misc']
Platform = data_set.loc[data_set['Genre'] == 'Platform']
Puzzle = data_set.loc[data_set['Genre'] == 'Puzzle']
Racing = data_set.loc[data_set['Genre'] == 'Racing']
RolePlaying = data_set.loc[data_set['Genre'] == 'Role-Playing']
Shooter = data_set.loc[data_set['Genre'] == 'Shooter']
Simulation = data_set.loc[data_set['Genre'] == 'Simulation']
Sports = data_set.loc[data_set['Genre'] == 'Sports']
Strategy = data_set.loc[data_set['Genre'] == 'Strategy']

Action_Sales = np.mean(Action['Global_Sales'])
Adventure_Sales = np.mean(Adventure['Global_Sales'])
Fighting_Sales = np.mean(Fighting['Global_Sales'])
Misc_Sales = np.mean(Misc['Global_Sales'])
Platform_Sales = np.mean(Platform['Global_Sales'])
Puzzle_Sales = np.mean(Puzzle['Global_Sales'])
Racing_Sales = np.mean(Racing['Global_Sales'])
RolePlaying_Sales = np.mean(RolePlaying['Global_Sales'])
Shooter_Sales = np.mean(Shooter['Global_Sales'])
Simulation_Sales = np.mean(Simulation['Global_Sales'])
Sports_Sales = np.mean(Sports['Global_Sales'])
Strategy_Sales = np.mean(Strategy['Global_Sales'])

y = [Action_Sales, Adventure_Sales, Fighting_Sales, Misc_Sales, Platform_Sales, Puzzle_Sales, Racing_Sales, RolePlaying_Sales, Shooter_Sales, Simulation_Sales, Sports_Sales, Strategy_Sales]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.bar(x, y, color="blue")
plt.show()

XOne = data_set.loc[data_set['Platform'] == 'XOne']
PS4 = data_set.loc[data_set['Platform'] == 'PS4']
WiiU = data_set.loc[data_set['Platform'] == 'WiiU']
PC = data_set.loc[data_set['Platform'] == 'PC']

