# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
suicide = pd.read_csv("../input/who_suicide_statistics.csv")
suicide.info()
aze = suicide[suicide.country == "Azerbaijan"]






aze_1981 = aze[aze.year == 1981]

aze_1981_suicides = aze_1981.suicides_no

suicide_no_1981 = 0

for i in aze_1981_suicides :
    suicide_no_1981 = i + suicide_no_1981

suicide_no_1981


aze_1982 = aze[aze.year == 1982]

aze_1982_suicides = aze_1982.suicides_no

suicide_no_1982 = 0

for i in aze_1982_suicides :
    suicide_no_1982 = i + suicide_no_1982
    
suicide_no_1982



aze_1983 = aze[aze.year == 1983]

aze_1983_suicides = aze_1983.suicides_no

suicide_no_1983 = 0

for i in aze_1983_suicides :
    suicide_no_1983 = i + suicide_no_1983

suicide_no_1983
    
    

aze_1984 = aze[aze.year == 1984]

aze_1984_suicides = aze_1984.suicides_no

suicide_no_1984 = 0

for i in aze_1984_suicides :
    suicide_no_1984 = i + suicide_no_1984

suicide_no_1984
    
    

aze_1985 = aze[aze.year == 1985]

aze_1985_suicides = aze_1985.suicides_no

suicide_no_1985 = 0

for i in aze_1985_suicides :
    suicide_no_1985 = i + suicide_no_1985

suicide_no_1985



aze_1986 = aze[aze.year == 1986]

aze_1986_suicides = aze_1986.suicides_no

suicide_no_1986 = 0

for i in aze_1986_suicides :
    suicide_no_1986 = i + suicide_no_1986

suicide_no_1986


aze_1987 = aze[aze.year == 1987]

aze_1987_suicides = aze_1987.suicides_no

suicide_no_1987 = 0

for i in aze_1987_suicides :
    suicide_no_1987 = i + suicide_no_1987

suicide_no_1987 
    
    
    

aze_1988 = aze[aze.year == 1988]

aze_1988_suicides = aze_1988.suicides_no

suicide_no_1988 = 0

for i in aze_1988_suicides :
    suicide_no_1988 = i + suicide_no_1988

suicide_no_1988
    



aze_1989 = aze[aze.year == 1989]

aze_1989_suicides = aze_1989.suicides_no

suicide_no_1989 = 0

for i in aze_1989_suicides :
    suicide_no_1989 = i + suicide_no_1989
    
suicide_no_1989
    
    
    

aze_1990 = aze[aze.year == 1990]

aze_1990_suicides = aze_1990.suicides_no

suicide_no_1990 = 0

for i in aze_1990_suicides :
    suicide_no_1990 = i + suicide_no_1990
suicide_no_1990






aze_1991 = aze[aze.year == 1991]

aze_1991_suicides = aze_1991.suicides_no

suicide_no_1991 = 0

for i in aze_1991_suicides :
    suicide_no_1991 = i + suicide_no_1991
suicide_no_1991



aze_1992 = aze[aze.year == 1992]

aze_1992_suicides = aze_1992.suicides_no

suicide_no_1992 = 0

for i in aze_1992_suicides :
    suicide_no_1992 = i + suicide_no_1992
suicide_no_1992



aze_1993 = aze[aze.year == 1993]

aze_1993_suicides = aze_1993.suicides_no

suicide_no_1993 = 0

for i in aze_1993_suicides :
    suicide_no_1993 = i + suicide_no_1993
suicide_no_1993





aze_1994 = aze[aze.year == 1994]

aze_1994_suicides = aze_1994.suicides_no

suicide_no_1994 = 0

for i in aze_1994_suicides :
    suicide_no_1994 = i + suicide_no_1994
suicide_no_1994







aze_1995 = aze[aze.year == 1995]

aze_1995_suicides = aze_1995.suicides_no

suicide_no_1995 = 0

for i in aze_1995_suicides :
    suicide_no_1995 = i + suicide_no_1995
suicide_no_1995







aze_1996 = aze[aze.year == 1996]

aze_1996_suicides = aze_1996.suicides_no

suicide_no_1996 = 0

for i in aze_1996_suicides :
    suicide_no_1996 = i + suicide_no_1996
suicide_no_1996






aze_1997 = aze[aze.year == 1997]

aze_1997_suicides = aze_1997.suicides_no

suicide_no_1997 = 0

for i in aze_1997_suicides :
    suicide_no_1997 = i + suicide_no_1997
suicide_no_1997






aze_1998 = aze[aze.year == 1998]

aze_1998_suicides = aze_1998.suicides_no

suicide_no_1998 = 0

for i in aze_1998_suicides :
    suicide_no_1998 = i + suicide_no_1998
suicide_no_1998





aze_1999 = aze[aze.year == 1999]

aze_1999_suicides = aze_1999.suicides_no

suicide_no_1999 = 0

for i in aze_1999_suicides :
    suicide_no_1999 = i + suicide_no_1999
suicide_no_1999







aze_2000 = aze[aze.year == 2000]

aze_2000_suicides = aze_2000.suicides_no

suicide_no_2000 = 0

for i in aze_2000_suicides :
    suicide_no_2000 = i + suicide_no_2000
suicide_no_2000







aze_2000 = aze[aze.year == 2000]

aze_2000_suicides = aze_2000.suicides_no

suicide_no_2000 = 0

for i in aze_2000_suicides :
    suicide_no_2000 = i + suicide_no_2000
suicide_no_2000







aze_2001 = aze[aze.year == 2001]

aze_2001_suicides = aze_2001.suicides_no

suicide_no_2001 = 0

for i in aze_2001_suicides :
    suicide_no_2001 = i + suicide_no_2001
suicide_no_2001







aze_2002 = aze[aze.year == 2002]

aze_2002_suicides = aze_2002.suicides_no

suicide_no_2002 = 0

for i in aze_2002_suicides :
    suicide_no_2002 = i + suicide_no_2002
suicide_no_2002








aze_2003 = aze[aze.year == 2003]

aze_2003_suicides = aze_2003.suicides_no

suicide_no_2003 = 0

for i in aze_2003_suicides :
    suicide_no_2003 = i + suicide_no_2003
suicide_no_2003








aze_2004 = aze[aze.year == 2004]

aze_2004_suicides = aze_2004.suicides_no

suicide_no_2004 = 0

for i in aze_2004_suicides :
    suicide_no_2004 = i + suicide_no_2004
suicide_no_2004








aze_2005 = aze[aze.year == 2005]

aze_2005_suicides = aze_2005.suicides_no

suicide_no_2005 = 0

for i in aze_2005_suicides :
    suicide_no_2005 = i + suicide_no_2005
suicide_no_2005










aze_2006 = aze[aze.year == 2006]

aze_2006_suicides = aze_2006.suicides_no

suicide_no_2006 = 0

for i in aze_2006_suicides :
    suicide_no_2006 = i + suicide_no_2006
suicide_no_2006








aze_2007 = aze[aze.year == 2007]

aze_2007_suicides = aze_2007.suicides_no

suicide_no_2007 = 0

for i in aze_2007_suicides :
    suicide_no_2007 = i + suicide_no_2007
suicide_no_2007






dead_number = pd.DataFrame({"Dead" : [suicide_no_1981,suicide_no_1982,suicide_no_1983,suicide_no_1984,suicide_no_1985,suicide_no_1986,suicide_no_1987,suicide_no_1988,suicide_no_1989,suicide_no_1990,suicide_no_1991,suicide_no_1992,suicide_no_1993,suicide_no_1994,suicide_no_1995,suicide_no_1996,suicide_no_1997,suicide_no_1998,suicide_no_1999,suicide_no_2000,suicide_no_2001,suicide_no_2002,suicide_no_2003,suicide_no_2004,suicide_no_2005,suicide_no_2006,suicide_no_2007]})

year = aze.year.unique()

year = [1981, 1982, 1983,1984,1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,2006, 2007]

year = pd.DataFrame(year)




fig, ax = plt.subplots()
ax.plot(year, dead_number,)

ax.set(xlabel='İl', ylabel='Ölüm',
       title='İllik intihar qrafiki')
ax.grid(500)

plt.show()




