# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
#data = pd.read_csv('../input/Police_Department_Incidents_-_Previous_Year__2016_.csv')

data1999 = pd.read_csv('../input/nse-1999-to-2000/NIFTY50all01-04-1999-TO-30-03-2000.csv')

data1999.shape
data1999.head()
print(os.listdir("../input/nse-2000-to-2001"))
data2000 = pd.read_csv('../input/nse-2000-to-2001/NIFTY50all01-04-2000-TO-30-03-2001.csv')

data2000.head()
merged = data1999.append(data2000)

merged.shape
print(os.listdir("../input/nse-2000-to-2019"))
data2001 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2001-TO-30-03-2002.csv')

data2002 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2002-TO-30-03-2003.csv')

data2003 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2003-TO-30-03-2004.csv')

data2004 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2004-TO-30-03-2005.csv')

data2005 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2005-TO-30-03-2006.csv')

data2006 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2006-TO-30-03-2007.csv')

data2007 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2007-TO-30-03-2008.csv')

data2008 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2008-TO-30-03-2009.csv')

data2009 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2009-TO-30-03-2010.csv')

data2010 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2010-TO-30-03-2011.csv')

data2011 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2011-TO-30-03-2012.csv')

data2012 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2012-TO-30-03-2013.csv')

data2013 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2013-TO-30-03-2014.csv')

data2014 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2014-TO-30-03-2015.csv')

data2015 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2015-TO-30-03-2016.csv')

data2016 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2016-TO-30-03-2017.csv')

data2017 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2017-TO-30-03-2018.csv')

data2018 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all01-04-2018-TO-30-03-2019.csv')

data2019 = pd.read_csv('../input/nse-2000-to-2019/NIFTY50all31-03-2019-TO-05-06-2019.csv')
mergedPePbYield = merged.append(data2001).append(data2002).append(data2003).append(data2004).append(data2005).append(data2006).append(data2007).append(data2008).append(data2009).append(data2010).append(data2011).append(data2012).append(data2013).append(data2014).append(data2015).append(data2016).append(data2017).append(data2018).append(data2019)

mergedPePbYield.shape
mergedPePbYield.head()
mergedPePbYield.corr()
print(os.listdir("../input/nifty50-realdata"))
nse1999 = pd.read_csv('../input/nifty50-realdata/nifty50-1999-2000.csv')

nse2000 = pd.read_csv('../input/nifty50-realdata/nifty50-2000-2001.csv')

nse2001 = pd.read_csv('../input/nifty50-realdata/nifty50-2001-2002.csv')

nse2002 = pd.read_csv('../input/nifty50-realdata/nifty50-2002-2003.csv')

nse2003 = pd.read_csv('../input/nifty50-realdata/nifty50-2003-2004.csv')

nse2004 = pd.read_csv('../input/nifty50-realdata/nifty50-2004-2005.csv')

nse2005 = pd.read_csv('../input/nifty50-realdata/nifty50-2005-2006.csv')

nse2006 = pd.read_csv('../input/nifty50-realdata/nifty50-2006-2007.csv')

nse2007 = pd.read_csv('../input/nifty50-realdata/nifty50-2007-2008.csv')

nse2008 = pd.read_csv('../input/nifty50-realdata/nifty50-2008-2009.csv')

nse2009 = pd.read_csv('../input/nifty50-realdata/nifty50-2009-2010.csv')

nse2010 = pd.read_csv('../input/nifty50-realdata/nifty50-2010-2011.csv')

nse2011 = pd.read_csv('../input/nifty50-realdata/nifty50-2011-2012.csv')

nse2012 = pd.read_csv('../input/nifty50-realdata/nifty50-2012-2013.csv')

nse2013 = pd.read_csv('../input/nifty50-realdata/nifty50-2013-2014.csv')

nse2014 = pd.read_csv('../input/nifty50-realdata/nifty50-2014-2015.csv')

nse2015 = pd.read_csv('../input/nifty50-realdata/nifty50-2015-2016.csv')

nse2016 = pd.read_csv('../input/nifty50-realdata/nifty50-2016-2017.csv')

nse2017 = pd.read_csv('../input/nifty50-realdata/nifty50-2017-2018.csv')

nse2018 = pd.read_csv('../input/nifty50-realdata/nifty50-2018-2019.csv')

nse2019 = pd.read_csv('../input/nifty50-realdata/nifty50-2019-2019.csv')
mergedNseData = nse1999.append(nse2000).append(nse2001).append(nse2002).append(nse2003).append(nse2004).append(nse2005).append(nse2006).append(nse2007).append(nse2008).append(nse2009).append(nse2010).append(nse2011).append(nse2012).append(nse2013).append(nse2014).append(nse2015).append(nse2016).append(nse2017).append(nse2018).append(nse2019)

mergedNseData.shape
mergedFinal = mergedPePbYield.merge(mergedNseData, on='Date')

mergedFinal.shape
mergedFinal.corr()
mergedFinal.isnull().sum()
mergedFinal.hist()
mergedFinal.plot()
mergedFinal.plot(kind='scatter',x='P/E',y='Close')
mergedFinal.plot(kind='scatter',x='P/E',y='P/B')
mergedFinal.plot(kind='scatter',x='Low',y='High')