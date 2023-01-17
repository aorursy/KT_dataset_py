import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("../input/uk_stats.csv")

data.head()
plt.figure(figsize=(18,8))

plt.plot(data["Year"], data["GDP"], label='GDP')

plt.annotate("Suez Crisis",xy=(1956,1.7),xytext=(1950,0.5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Pound Devalued",xy=(1967,2.8),xytext=(1962,0.5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Oil Crisis",xy=(1973,6.5),xytext=(1967,6),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("UK Joins EEC",xy=(1973,6.5),xytext=(1965,5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Brexit Referendum 1.0",xy=(1975,-1.0),xytext=(1962,-3),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Oil Crisis II",xy=(1979,3.7),xytext=(1981,5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Winter of discontent",xy=(1978,4.2),xytext=(1976,5.5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Falklands Crisis",xy=(1982,2),xytext=(1984,1),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("US Fed Reserve acts to lower inflation",xy=(1988,5.7),xytext=(1991,6.1),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Black Monday",xy=(1987,5.4),xytext=(1981,6.3),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Black Friday",xy=(1989,2.6),xytext=(1991,4.3),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Gulf War",xy=(1990,0.7),xytext=(1984,-1.8),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Maastricht Treaty",xy=(1992,0.4),xytext=(1993,-1.8),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Black Wednesday",xy=(1992,0.4),xytext=(1995,1),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Dot Com peak",xy=(2000,3.4),xytext=(1995,5.5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("9-11",xy=(2001,3),xytext=(1996,2.1),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Gulf War II",xy=(2003,3.3),xytext=(2005,5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Financial Crisis",xy=(2007,2.4),xytext=(2010,4),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("2012 Olympics",xy=(2012,1.5),xytext=(2013,0.5),arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate("Brexit Referendum 2.0",(2016,1.9),xytext=(2018,3),arrowprops=dict(facecolor='black',shrink=0.02))



#Leaders

plt.annotate("Thatcher",(1982,-4),xytext=(1982,-3),arrowprops=dict(facecolor='blue',shrink=0.02))

plt.annotate("Major",(1992,-4.2),xytext=(1992,-3),arrowprops=dict(facecolor='blue',shrink=0.02))

plt.annotate("Blair",(2000,-4),xytext=(2000,-3),arrowprops=dict(facecolor='red',shrink=0.02))

plt.annotate("Brown",(2008,-4.2),xytext=(2005,-3),arrowprops=dict(facecolor='red',shrink=0.02))

plt.annotate("Cameron",(2012,-4),xytext=(2012,-3),arrowprops=dict(facecolor='blue',shrink=0.02))

plt.annotate("May",(2017,-4.2),xytext=(2017,-3),arrowprops=dict(facecolor='blue',shrink=0.02))

plt.annotate("Bojo",(2020,-4),xytext=(2020,-3),arrowprops=dict(facecolor='blue',shrink=0.02))



#Who's in power (red: Labour, blue: Conservative)

plt.plot([1948,1951], [-4,-4], 'r-')

plt.plot([1951,1964], [-4,-4], 'b-')

plt.plot([1964,1970], [-4,-4], 'r-')

plt.plot([1970,1974], [-4,-4], 'b-')

plt.plot([1974,1976], [-4,-4], 'r-')

plt.plot([1976,1979], [-4.2,-4.2], 'r-')

plt.plot([1979,1990], [-4,-4], 'b-')

plt.plot([1990,1997], [-4.2,-4.2], 'b-')

plt.plot([1997,2007], [-4,-4], 'r-')

plt.plot([2007,2010], [-4.2,-4.2], 'r-')

plt.plot([2010,2016], [-4,-4], 'b-')

plt.plot([2016,2019], [-4.2,-4.2], 'b-')

plt.plot([2019,2020], [-4,-4], 'b--')



#Trendlines

plt.plot([1988,2020], [5.9,2.1], 'g--')



#Plot general

plt.xlabel('Year')

plt.ylabel('%')

plt.title("UK GDP (% Change Year on Year)")

#plt.legend()

plt.grid(True)

plt.axhline(y=0,color='k')

plt.show()
plt.figure(figsize=(18,7))

#plt.plot(data["Year"], data["GDP"], label='GDP')

plt.plot(data["Year"], data["Inflation"], label='Inflation')

plt.plot(data["Year"], data["Unemployment"], label='Unemployment')



#Plot general

plt.xlabel('Year')

plt.ylabel('%')

plt.title("% Change Year on Year")

plt.legend()

plt.grid(True)

plt.axhline(y=0,color='k')

plt.show()